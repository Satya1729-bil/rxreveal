"""Measure how well RxReveal answers questions about the bundled sample PDF.

The sample PDF (data/sample_medical_records.pdf) is made by scripts/make_sample_pdf.py
from a fixed random seed, so every answer is known in advance. eval/questions.json holds
that answer key, and tests/test_evaluate.py checks the key against the generator.

Usage:
    python -m rxreveal.evaluate
    python -m rxreveal.evaluate --top-k 10 --out eval/results.md

Needs GEMINI_API_KEY (search and answers both use Gemini by default). On the free tier,
keep the default pause between questions so you stay under the per-minute limit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import get_settings
from .index import EmbedderError, VectorIndex, make_embedder
from .ingest import load_folder
from .llm import GeminiLLM, LLMError
from .rag import answer_question

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = ROOT / "eval" / "questions.json"
DEFAULT_REPORT = ROOT / "eval" / "results.md"

# Phrases that count as a correct "the documents do not say" answer.
NOT_FOUND_PHRASES = (
    "could not find",
    "no patient",
    "none of the",
    "not mentioned",
    "not contain",
    "does not include",
    "do not include",
    "no information",
    "not provided",
    "not available in the documents",
)

RECORD_START = re.compile(r"^\s*Patient\s+(\d+)\s*:", re.IGNORECASE)
CITATION = re.compile(r"\[\d+\]")


@dataclass
class QuestionResult:
    id: str
    type: str
    question: str
    passed: bool
    answer: str
    detail: str
    retrieved: list[int] = field(default_factory=list)
    retrieval_ok: bool | None = None
    cited: bool | None = None
    recall: float | None = None


# ---------------------------------------------------------------- scoring helpers
def load_questions(path: Path = DEFAULT_QUESTIONS) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def has_term(answer: str, term: str | list[str]) -> bool:
    """True if the answer contains the term (or any of its listed alternatives)."""
    options = [term] if isinstance(term, str) else term
    text = answer.lower()
    return any(option.lower() in text for option in options)


def patients_mentioned(answer: str, patients: dict[int, str]) -> set[int]:
    """Patient ids the answer refers to, by full name or as "Patient <n>"."""
    found = set()
    for pid, name in patients.items():
        if re.search(rf"\b{re.escape(name)}\b", answer, re.IGNORECASE):
            found.add(pid)
        elif re.search(rf"\bPatient\s+{pid}\b", answer, re.IGNORECASE):
            found.add(pid)
    return found


def retrieved_patients(sources) -> list[int]:
    ids = []
    for source in sources:
        match = RECORD_START.match(source.chunk.text)
        if match:
            ids.append(int(match.group(1)))
    return ids


def says_not_found(answer: str) -> bool:
    text = answer.lower()
    return any(phrase in text for phrase in NOT_FOUND_PHRASES)


def check_data_matches_key(index: VectorIndex, patients: dict[int, str]) -> list[str]:
    """Problems found when comparing the indexed records with the answer key."""
    names = {}
    for chunk in index.chunks:
        match = RECORD_START.match(chunk.text)
        if match:
            names[int(match.group(1))] = chunk.text
    problems = []
    for pid, name in patients.items():
        text = names.get(pid)
        if text is None:
            problems.append(f"Patient {pid} is not in the index.")
        elif name.lower() not in text.lower():
            problems.append(f"Patient {pid} is not {name} in the indexed data.")
    return problems


# ---------------------------------------------------------------- one question
def score(q: dict, result, patients: dict[int, str]) -> QuestionResult:
    answer = result.text
    retrieved = retrieved_patients(result.sources)
    kind = q["type"]
    base = dict(id=q["id"], type=kind, question=q["question"], answer=answer, retrieved=retrieved,
                cited=bool(CITATION.search(answer)) if result.mode == "llm" else None)

    if kind == "not_found":
        passed = result.mode == "not_found" or says_not_found(answer)
        detail = "Said the documents do not answer it" if passed else "Gave an answer that is not in the documents"
        base["cited"] = None  # a refusal does not need a citation
        return QuestionResult(passed=passed, detail=detail, **base)

    if result.mode != "llm":
        reason = result.error or result.mode
        return QuestionResult(passed=False, detail=f"No generated answer ({reason})", **base)

    if kind == "fact":
        missing = [t if isinstance(t, str) else t[0] for t in q["expect"] if not has_term(answer, t)]
        retrieval_ok = q["patient"] in retrieved
        passed = not missing
        detail = "All expected details present" if passed else "Missing: " + ", ".join(missing)
        if not retrieval_ok:
            detail += f"; Patient {q['patient']} was not retrieved"
        return QuestionResult(passed=passed, detail=detail, retrieval_ok=retrieval_ok, **base)

    if kind == "list":
        expected = set(q["patients"])
        found = patients_mentioned(answer, patients)
        hits, extra = found & expected, found - expected
        recall = len(hits) / len(expected)
        passed = hits == expected and not extra
        parts = [f"Named {len(hits)} of {len(expected)}"]
        if extra:
            parts.append("wrongly added " + ", ".join(patients[p] for p in sorted(extra)))
        retrieved_hits = len(expected & set(retrieved))
        parts.append(f"{retrieved_hits} of {len(expected)} records retrieved")
        return QuestionResult(passed=passed, detail="; ".join(parts), recall=recall,
                              retrieval_ok=retrieved_hits == len(expected), **base)

    raise ValueError(f"Unknown question type: {kind}")


# ---------------------------------------------------------------- run + report
def run_eval(key: dict, index, embedder, llm, top_k: int, min_score: float,
             pause: float = 0.0) -> list[QuestionResult]:
    patients = {int(k): v for k, v in key["patients"].items()}
    results = []
    for n, q in enumerate(key["questions"]):
        if n and pause:
            time.sleep(pause)
        result = answer_question(q["question"], index=index, embedder=embedder, llm=llm,
                                 top_k=top_k, min_score=min_score)
        results.append(score(q, result, patients))
    return results


def summarize(results: list[QuestionResult]) -> dict:
    def part(kind):
        group = [r for r in results if r.type == kind]
        return sum(r.passed for r in group), len(group)

    facts = [r for r in results if r.type == "fact"]
    answered = [r for r in results if r.cited is not None]
    lists = [r for r in results if r.type == "list"]
    return {
        "correct": sum(r.passed for r in results),
        "total": len(results),
        "fact": part("fact"),
        "list": part("list"),
        "not_found": part("not_found"),
        "fact_retrieval": (sum(bool(r.retrieval_ok) for r in facts), len(facts)),
        "cited": (sum(bool(r.cited) for r in answered), len(answered)),
        "list_recall": (sum(r.recall or 0 for r in lists) / len(lists)) if lists else None,
    }


def _pct(a: int, b: int) -> str:
    return f"{a}/{b} ({round(100 * a / b)}%)" if b else "n/a"


def render_markdown(results: list[QuestionResult], summary: dict, settings_line: str) -> str:
    s = summary
    lines = [
        "# RxReveal evaluation",
        "",
        settings_line,
        "",
        "| Measure | Result |",
        "|---|---|",
        f"| Questions answered correctly | {_pct(s['correct'], s['total'])} |",
        f"| Single-patient lookups | {_pct(*s['fact'])} |",
        f"| \"List all patients who...\" questions | {_pct(*s['list'])} |",
        f"| Questions the documents cannot answer (should refuse) | {_pct(*s['not_found'])} |",
        f"| Right record retrieved for lookups | {_pct(*s['fact_retrieval'])} |",
        f"| Answers with a [n] citation | {_pct(*s['cited'])} |",
    ]
    if s["list_recall"] is not None:
        lines.append(f"| Average share of patients named in list answers | {round(100 * s['list_recall'])}% |")
    lines += ["", "## Each question", "", "| ID | Question | Result | Detail |", "|---|---|---|---|"]
    for r in results:
        mark = "Pass" if r.passed else "Fail"
        question = r.question.replace("|", "\\|")
        lines.append(f"| {r.id} | {question} | {mark} | {r.detail} |")
    return "\n".join(lines) + "\n"


def _load_or_build_index(settings, embedder) -> VectorIndex:
    if VectorIndex.exists(settings.index_dir):
        index = VectorIndex.load(settings.index_dir)
        if index.matches(embedder):
            return index
    chunks = load_folder(settings.data_dir, size=settings.chunk_size, overlap=settings.chunk_overlap)
    index = VectorIndex.build(chunks, embedder)
    index.save(settings.index_dir)
    return index


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Evaluate RxReveal on the sample PDF.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--min-score", type=float, default=settings.min_score)
    parser.add_argument("--pause", type=float, default=4.0,
                        help="Seconds to wait between questions (free-tier rate limits).")
    args = parser.parse_args(argv)

    if not settings.gemini_api_key:
        print("GEMINI_API_KEY is not set. Add it to .env and run again.", file=sys.stderr)
        return 1
    try:
        embedder = make_embedder(settings)
        llm = GeminiLLM(api_key=settings.gemini_api_key, model=settings.gemini_model)
        index = _load_or_build_index(settings, embedder)
    except (EmbedderError, LLMError, ValueError) as exc:
        print(f"Could not start the evaluation: {exc}", file=sys.stderr)
        return 1

    key = load_questions(args.questions)
    problems = check_data_matches_key(index, {int(k): v for k, v in key["patients"].items()})
    if problems:
        print("The indexed documents do not match the answer key:", file=sys.stderr)
        for problem in problems[:5]:
            print(f"  - {problem}", file=sys.stderr)
        print("Index only data/sample_medical_records.pdf and run again.", file=sys.stderr)
        return 1

    results = run_eval(key, index, embedder, llm, args.top_k, args.min_score, pause=args.pause)
    summary = summarize(results)
    settings_line = (
        f"Run on {time.strftime('%Y-%m-%d')} with answers by `{settings.gemini_model}`, "
        f"search by `{embedder.name}`, top_k={args.top_k}, min_score={args.min_score}."
    )
    report = render_markdown(results, summary, settings_line)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
