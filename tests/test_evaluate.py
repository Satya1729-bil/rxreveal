import importlib.util
import re
from pathlib import Path

from rxreveal.evaluate import (
    has_term,
    load_questions,
    patients_mentioned,
    render_markdown,
    run_eval,
    says_not_found,
    summarize,
)
from rxreveal.index import VectorIndex
from rxreveal.ingest import chunk_pages

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    spec = importlib.util.spec_from_file_location("make_sample_pdf", ROOT / "scripts" / "make_sample_pdf.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records():
    """The sample PDF's records, parsed into fields, keyed by patient number."""
    records = {}
    for raw in _generator().make_records(30):
        text = raw.replace("<br/>", "\n")
        pid = int(re.match(r"Patient (\d+):", text).group(1))
        fields = dict(re.findall(r"^(?:Patient \d+: )?(\w[\w -]*?): (.*)$", text, re.MULTILINE))
        fields["text"] = text
        fields["temp"] = float(re.search(r"Temperature - ([\d.]+)", text).group(1))
        records[pid] = fields
    return records


# ---------------------------------------------------------------- the answer key is correct
def test_answer_key_names_match_the_generator():
    key, records = load_questions(), _records()
    assert {int(k): v for k, v in key["patients"].items()} == {p: r["Name"] for p, r in records.items()}


def test_fact_answers_are_in_the_right_record():
    records = _records()
    for q in load_questions()["questions"]:
        if q["type"] == "fact":
            assert all(has_term(records[q["patient"]]["text"], term) for term in q["expect"]), q["id"]


def test_list_answers_match_the_generator():
    records = _records()
    rules = {
        "list-01": lambda r: r["Disease"] == "Pneumonia",
        "list-02": lambda r: r["Disease"] == "Asthma",
        "list-03": lambda r: r["Disease"] == "Arthritis",
        "list-04": lambda r: r["Disease"] == "Tuberculosis",
        "list-05": lambda r: r["temp"] > 103,
        "list-06": lambda r: "Insulin" in r["Treatment"],
    }
    lists = [q for q in load_questions()["questions"] if q["type"] == "list"]
    assert {q["id"] for q in lists} == set(rules)
    for q in lists:
        assert q["patients"] == sorted(p for p, r in records.items() if rules[q["id"]](r)), q["id"]


def test_out_of_scope_questions_really_have_no_answer():
    all_text = " ".join(r["text"] for r in _records().values()).lower()
    assert "patient 45" not in all_text
    assert "blood type" not in all_text
    assert "malaria" not in all_text


# ---------------------------------------------------------------- scoring
def test_scoring_helpers():
    patients = {1: "Ana Diaz", 2: "Ben Ray", 3: "Cy Wu"}
    assert patients_mentioned("- Ana Diaz [1]\n- Patient 3 [2]", patients) == {1, 3}
    assert has_term("Follow-up in six weeks", ["6 week", "six week"])
    assert says_not_found("I could not find this in the documents.")
    assert says_not_found("No patients in the documents have malaria.")
    assert not says_not_found("Patient 2 has diabetes [1].")


class ScriptedLLM:
    def __init__(self, replies):
        self.replies = replies

    def generate(self, system, prompt):
        question = prompt.rsplit("Question: ", 1)[1].strip()
        return self.replies[question]


def test_run_eval_scores_each_question_type(sample_pages, embedder):
    index = VectorIndex.build(chunk_pages(sample_pages, source="records.pdf"), embedder)
    key = {
        "patients": {"1": "Ana Diaz", "2": "Ben Ray", "3": "Cy Wu"},
        "questions": [
            {"id": "fact-01", "type": "fact", "question": "What disease does Cy Wu have asthma",
             "patient": 3, "expect": ["Asthma"]},
            {"id": "list-01", "type": "list", "question": "Which patients have tuberculosis",
             "patients": [1]},
            {"id": "list-02", "type": "list", "question": "Which patients have diabetes",
             "patients": [2]},
            {"id": "none-01", "type": "not_found", "question": "Which patients have tuberculosis malaria"},
        ],
    }
    llm = ScriptedLLM({
        "What disease does Cy Wu have asthma": "Cy Wu has asthma [1].",
        "Which patients have tuberculosis": "- Ana Diaz [1]",
        "Which patients have diabetes": "- Ben Ray [1]\n- Ana Diaz [2]",
        "Which patients have tuberculosis malaria": "I could not find this in the documents.",
    })
    results = run_eval(key, index, embedder, llm, top_k=3, min_score=0.0)
    by_id = {r.id: r for r in results}

    assert by_id["fact-01"].passed and by_id["fact-01"].retrieval_ok and by_id["fact-01"].cited
    assert by_id["list-01"].passed and by_id["list-01"].recall == 1.0
    assert not by_id["list-02"].passed and "wrongly added Ana Diaz" in by_id["list-02"].detail
    assert by_id["none-01"].passed

    summary = summarize(results)
    assert summary["correct"] == 3 and summary["total"] == 4
    report = render_markdown(results, summary, "Test run.")
    assert "| Questions answered correctly | 3/4 (75%) |" in report
    assert "| list-02 |" in report
