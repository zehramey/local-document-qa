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
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BENCHMARK_DIR = Path(__file__).parent


def run(api_base_url: str, models: list[str], questions_path: Path) -> list[dict[str, object]]:
    eval_set = json.loads(questions_path.read_text(encoding="utf-8"))
    document_id = eval_set["document_id"]
    questions = eval_set["questions"]

    results: list[dict[str, object]] = []
    total = len(models) * len(questions)
    done = 0

    for model_id in models:
        for q in questions:
            done += 1
            print(f"[{done}/{total}] {model_id} :: {q['id']}", flush=True)
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
                        "question_id": q["id"],
                        "question": q["question"],
                        "expected_answerable": q["expected_answerable"],
                        "error": f"HTTP {response.status_code}: {response.text[:300]}",
                    }
                )
                continue

            body = response.json()
            metrics = body.get("metrics") or {}
            results.append(
                {
                    "model_id": model_id,
                    "question_id": q["id"],
                    "question": q["question"],
                    "expected_answerable": q["expected_answerable"],
                    "answerable": body["answerable"],
                    "answer": body["answer"],
                    "citation_count": len(body["citations"]),
                    "rejected_citation_count": len(body["rejected_citation_ids"]),
                    "prompt_tokens": metrics.get("prompt_tokens"),
                    "output_tokens": metrics.get("output_tokens"),
                    "total_duration_seconds": metrics.get("total_duration_seconds"),
                    "tokens_per_second": metrics.get("tokens_per_second"),
                    "client_elapsed_seconds": round(client_elapsed, 2),
                    "error": None,
                    # Filled in by hand after reading the answer.
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
    fieldnames = list(results[0].keys()) if results else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    return json_path, csv_path


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


if __name__ == "__main__":
    main()
