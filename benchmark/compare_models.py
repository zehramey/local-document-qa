"""Runs the eval_questions.json question set through /questions once per
model and writes the results to benchmark/results/ as JSON and CSV.

This only measures what the API can measure on its own (timing, tokens,
whether citations were rejected) — it does NOT judge whether an answer is
actually correct. That still needs a human to read the `answer` column
against the source document; see README.md's evaluation notes.

Usage:
    python benchmark/compare_models.py --models qwen/qwen3.5-4b,gemini-flash-latest
"""

import argparse
import csv
import json
import re
import string
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx

BENCHMARK_DIR = Path(__file__).parent

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")


def _normalize_answer_text(text: str) -> str:
    """Standard SQuAD normalization: lowercase, strip punctuation/articles,
    collapse whitespace — so "It's Mercury." and "mercury" are the same
    answer for scoring purposes."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def _exact_match(prediction: str, gold_answers: list[str]) -> float:
    if not gold_answers:
        return 0.0
    normalized_prediction = _normalize_answer_text(prediction)
    return float(any(normalized_prediction == _normalize_answer_text(g) for g in gold_answers))


def _f1(prediction: str, gold_answers: list[str]) -> float:
    """Token-overlap F1 against each gold answer, keeping the best score —
    the same metric SQuAD's own eval script uses."""
    if not gold_answers:
        return 0.0
    prediction_tokens = _normalize_answer_text(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_answer_text(gold).split()
        if not prediction_tokens or not gold_tokens:
            score = float(prediction_tokens == gold_tokens)
        else:
            num_same = sum((Counter(prediction_tokens) & Counter(gold_tokens)).values())
            if num_same == 0:
                score = 0.0
            else:
                precision = num_same / len(prediction_tokens)
                recall = num_same / len(gold_tokens)
                score = 2 * precision * recall / (precision + recall)
        best = max(best, score)
    return best


def _documents_from_eval_set(eval_set: dict[str, object]) -> list[dict[str, object]]:
    """Normalizes both eval-set shapes to a list of {document_id, questions}.

    Legacy shape (eval_questions.json): a single top-level document_id +
    questions. Multi-document shape (eval_questions_squad.json, see
    prepare_squad_eval.py): a "documents" list, since SQuAD questions are
    each scoped to their own paragraph/document rather than one shared doc.
    """
    if "documents" in eval_set:
        return eval_set["documents"]  # type: ignore[return-value]
    return [{"document_id": eval_set["document_id"], "questions": eval_set["questions"]}]


def run(api_base_url: str, models: list[str], questions_path: Path) -> list[dict[str, object]]:
    eval_set = json.loads(questions_path.read_text(encoding="utf-8"))
    documents = _documents_from_eval_set(eval_set)

    results: list[dict[str, object]] = []
    total = len(models) * sum(len(doc["questions"]) for doc in documents)
    done = 0

    for model_id in models:
        for doc in documents:
            document_id = doc["document_id"]
            for q in doc["questions"]:
                done += 1
                print(f"[{done}/{total}] {model_id} :: {document_id[:12]} :: {q['id']}", flush=True)
                start = time.monotonic()
                try:
                    response = httpx.post(
                        f"{api_base_url}/questions",
                        json={
                            "document_id": document_id,
                            "question": q["question"],
                            "model_id": model_id,
                        },
                        timeout=240.0,
                    )
                    client_elapsed = time.monotonic() - start
                except httpx.HTTPError as exc:
                    results.append(
                        {
                            "model_id": model_id,
                            "document_id": document_id,
                            "question_id": q["id"],
                            "question": q["question"],
                            "expected_answerable": q["expected_answerable"],
                            "error": f"connection error: {exc}",
                        }
                    )
                    continue

                if response.status_code != 200:
                    results.append(
                        {
                            "model_id": model_id,
                            "document_id": document_id,
                            "question_id": q["id"],
                            "question": q["question"],
                            "expected_answerable": q["expected_answerable"],
                            "error": f"HTTP {response.status_code}: {response.text[:300]}",
                        }
                    )
                    continue

                body = response.json()
                metrics = body.get("metrics") or {}
                # Only present for eval sets built by prepare_squad_eval.py;
                # eval_questions.json's hand-authored set has no reference
                # answer text to score against.
                gold_answers = q.get("gold_answers", [])
                exact_match = None
                f1_score = None
                if q["expected_answerable"] and gold_answers:
                    exact_match = _exact_match(body["answer"], gold_answers)
                    f1_score = _f1(body["answer"], gold_answers)
                results.append(
                    {
                        "model_id": model_id,
                        "document_id": document_id,
                        "question_id": q["id"],
                        "question": q["question"],
                        "expected_answerable": q["expected_answerable"],
                        "answerable": body["answerable"],
                        "answerability_correct": body["answerable"] == q["expected_answerable"],
                        "answer": body["answer"],
                        "gold_answers": gold_answers,
                        "exact_match": exact_match,
                        "f1_score": f1_score,
                        "citation_count": len(body["citations"]),
                        "rejected_citation_count": len(body["rejected_citation_ids"]),
                        "prompt_tokens": metrics.get("prompt_tokens"),
                        "output_tokens": metrics.get("output_tokens"),
                        "total_duration_seconds": metrics.get("total_duration_seconds"),
                        "tokens_per_second": metrics.get("tokens_per_second"),
                        "client_elapsed_seconds": round(client_elapsed, 2),
                        "error": None,
                        # exact_match/f1_score cover "was the answer text
                        # right"; this is still worth filling in by hand for
                        # nuance auto-scoring can't catch (citation quality,
                        # a correct answer phrased so differently it scores
                        # low on F1, refusals that are technically right but
                        # unhelpful).
                        "human_correct": "",
                        "human_notes": "",
                    }
                )
    return results


def write_outputs(results: list[dict[str, object]]) -> tuple[Path, Path]:
    results_dir = BENCHMARK_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    json_path = results_dir / f"results_{stamp}.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = results_dir / f"results_{stamp}.csv"
    # Error rows have fewer keys than success rows (see run()); union of all
    # keys, in first-seen order, so a leading error row can't cause later
    # full rows to fail as "extra" fields.
    fieldnames: list[str] = []
    for row in results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return json_path, csv_path


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def print_summary(results: list[dict[str, object]]) -> None:
    """Per-model aggregate: error rate, answerability accuracy, and mean
    EM/F1 over the subset of rows that had gold_answers to score against."""
    model_ids = sorted({row["model_id"] for row in results})
    print("\nSummary:")
    for model_id in model_ids:
        rows = [row for row in results if row["model_id"] == model_id]
        errors = [row for row in rows if row.get("error")]
        scored = [row for row in rows if row.get("exact_match") is not None]
        answerability = [row for row in rows if row.get("answerability_correct") is not None]

        error_rate = len(errors) / len(rows)
        answerability_accuracy = _mean(
            [1.0 if row["answerability_correct"] else 0.0 for row in answerability]
        )
        mean_em = _mean([row["exact_match"] for row in scored])
        mean_f1 = _mean([row["f1_score"] for row in scored])

        print(f"  {model_id}:")
        print(f"    error rate:              {error_rate:.0%} ({len(errors)}/{len(rows)})")
        if answerability_accuracy is not None:
            print(f"    answerability accuracy:  {answerability_accuracy:.0%}")
        if mean_em is not None:
            print(f"    mean EM / F1 (n={len(scored)}):    {mean_em:.2f} / {mean_f1:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated model_ids to compare.")
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument(
        "--questions-file", default=str(BENCHMARK_DIR / "eval_questions.json"), type=Path
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    results = run(args.api_base_url, models, Path(args.questions_file))
    json_path, csv_path = write_outputs(results)

    print(f"\nWrote {len(results)} rows to:\n  {json_path}\n  {csv_path}")
    print_summary(results)


if __name__ == "__main__":
    main()
