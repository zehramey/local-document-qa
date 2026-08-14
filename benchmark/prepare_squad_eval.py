"""Builds a multi-document eval set from SQuAD 2.0 for compare_models.py.

SQuAD 2.0 (CC BY-SA 4.0, https://rajpurkar.github.io/SQuAD-explorer/) pairs
each question with `is_impossible` — an adversarially-written unanswerable
variant when true — which maps directly onto this project's
`expected_answerable` field. Each SQuAD paragraph becomes one uploaded
document (via POST /documents on a running API instance) with its own
questions, unlike eval_questions.json's single-document Little Prince set.

Each question also carries `gold_answers` — SQuAD's own annotator answer
strings, deduplicated, empty for unanswerable questions (see
_gold_answers()) — so compare_models.py can score answerable questions
with EM/F1 instead of relying only on a human reading every row.

This is English-only by design: it isolates "can the model do grounded RAG
at all" from "can the model do it in Turkish", which a machine-translated
Turkish variant (e.g. SQuAD-TR) would conflate — translation noise would
look identical to a genuine Turkish-comprehension failure.

Re-run this once per machine before benchmarking: document_ids are assigned
by your local Qdrant instance at upload time, so a committed
eval_questions_squad.json from another machine won't resolve here.

Usage:
    python benchmark/prepare_squad_eval.py --num-documents 6
"""

import argparse
import json
import random
from pathlib import Path

import httpx

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_LICENSE = "CC BY-SA 4.0"


def _download_squad(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SQuAD 2.0 dev set from {SQUAD_URL} ...")
    response = httpx.get(SQUAD_URL, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    path.write_bytes(response.content)


def _eligible_paragraphs(squad_data: dict, min_questions: int) -> list[dict]:
    """Paragraphs with enough questions and a real answerable/unanswerable mix."""
    eligible = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            qas = paragraph["qas"]
            possible = [q for q in qas if not q["is_impossible"]]
            impossible = [q for q in qas if q["is_impossible"]]
            if len(qas) >= min_questions and possible and impossible:
                eligible.append(
                    {
                        "article_title": article["title"],
                        "context": paragraph["context"],
                        "possible": possible,
                        "impossible": impossible,
                    }
                )
    return eligible


def _gold_answers(q: dict) -> list[str]:
    """Deduplicated reference answer strings for EM/F1-style auto-scoring.

    Empty for is_impossible questions on purpose: SQuAD 2.0's "correct
    answer" for those *is* the absence of an answer, which this project
    already represents via expected_answerable=False and NOT_FOUND_PHRASE
    (see app/services/rag_prompt.py) rather than any answer text.
    """
    if q["is_impossible"]:
        return []
    seen: list[str] = []
    for answer in q["answers"]:
        text = answer["text"].strip()
        if text and text not in seen:
            seen.append(text)
    return seen


def _select_questions(paragraph: dict, max_questions: int, rng: random.Random) -> list[dict]:
    half = max_questions // 2
    possible = rng.sample(paragraph["possible"], k=min(half, len(paragraph["possible"])))
    impossible = rng.sample(paragraph["impossible"], k=min(half, len(paragraph["impossible"])))
    selected = possible + impossible
    rng.shuffle(selected)
    return selected


def _upload_document(api_base_url: str, article_title: str, index: int, context: str) -> dict:
    filename = f"squad_{index:02d}_{article_title.replace(' ', '_')}.txt"
    response = httpx.post(
        f"{api_base_url}/documents",
        files={"file": (filename, context.encode("utf-8"))},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def build_eval_set(
    api_base_url: str,
    squad_path: Path,
    num_documents: int,
    min_questions_per_paragraph: int,
    max_questions_per_document: int,
    seed: int,
) -> dict:
    if not squad_path.exists():
        _download_squad(squad_path)

    squad_data = json.loads(squad_path.read_text(encoding="utf-8"))
    eligible = _eligible_paragraphs(squad_data, min_questions_per_paragraph)
    if len(eligible) < num_documents:
        raise ValueError(
            f"Only {len(eligible)} eligible paragraphs found, need {num_documents}."
        )

    rng = random.Random(seed)
    chosen = rng.sample(eligible, k=num_documents)

    documents = []
    for index, paragraph in enumerate(chosen, start=1):
        print(f"[{index}/{num_documents}] uploading paragraph from '{paragraph['article_title']}'")
        upload_result = _upload_document(
            api_base_url, paragraph["article_title"], index, paragraph["context"]
        )
        questions = _select_questions(paragraph, max_questions_per_document, rng)
        documents.append(
            {
                "document_id": upload_result["document_id"],
                "document_filename": upload_result["filename"],
                "source_article": paragraph["article_title"],
                "questions": [
                    {
                        "id": q["id"],
                        "question": q["question"],
                        "expected_answerable": not q["is_impossible"],
                        "gold_answers": _gold_answers(q),
                    }
                    for q in questions
                ],
            }
        )

    return {
        "source": f"SQuAD 2.0 dev-v2.0 ({SQUAD_LICENSE}), {SQUAD_URL}",
        "seed": seed,
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default="http://localhost:8000")
    parser.add_argument("--squad-path", default=str(DATA_DIR / "squad_dev_v2.json"), type=Path)
    parser.add_argument("--num-documents", default=6, type=int)
    parser.add_argument("--min-questions-per-paragraph", default=4, type=int)
    parser.add_argument("--max-questions-per-document", default=6, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--output", default=str(BENCHMARK_DIR / "eval_questions_squad.json"), type=Path
    )
    args = parser.parse_args()

    eval_set = build_eval_set(
        api_base_url=args.api_base_url,
        squad_path=args.squad_path,
        num_documents=args.num_documents,
        min_questions_per_paragraph=args.min_questions_per_paragraph,
        max_questions_per_document=args.max_questions_per_document,
        seed=args.seed,
    )

    args.output.write_text(json.dumps(eval_set, ensure_ascii=False, indent=2), encoding="utf-8")
    total_questions = sum(len(d["questions"]) for d in eval_set["documents"])
    print(
        f"\nWrote {len(eval_set['documents'])} documents / {total_questions} questions to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
