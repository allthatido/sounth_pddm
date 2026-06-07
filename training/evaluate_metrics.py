import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_references(path: Path, limit: int) -> list[str]:
    refs = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit > 0 and len(refs) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            messages = obj.get("messages") or []
            assistant = ""
            for message in reversed(messages):
                if message.get("role") == "assistant":
                    assistant = (message.get("content") or "").strip()
                    break
            refs.append(assistant)
    return refs


def first_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        for item in value:
            result = first_string(item)
            if result:
                return result
    if isinstance(value, dict):
        for key in [
            "response",
            "prediction",
            "generated_text",
            "output",
            "answer",
            "content",
            "text",
        ]:
            result = first_string(value.get(key))
            if result:
                return result
    return ""


def extract_prediction(obj: dict) -> str:
    for key in [
        "response",
        "prediction",
        "generated_text",
        "output",
        "answer",
        "content",
        "text",
    ]:
        pred = first_string(obj.get(key))
        if pred:
            return pred.strip()
    messages = obj.get("messages") or obj.get("conversation") or []
    if isinstance(messages, list):
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = first_string(message.get("content"))
                if content:
                    return content.strip()
    return ""


def load_predictions(path: Path, limit: int) -> list[str]:
    preds = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit > 0 and len(preds) >= limit:
                break
            if not line.strip():
                continue
            preds.append(extract_prediction(json.loads(line)))
    return preds


def compute_bleu(preds: list[str], refs: list[str]) -> dict:
    import sacrebleu

    score = sacrebleu.corpus_bleu(preds, [refs])
    return {
        "bleu": score.score,
        "bleu_precisions": score.precisions,
        "bleu_bp": score.bp,
    }


def compute_rouge(preds: list[str], refs: list[str]) -> dict:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    totals = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for pred, ref in zip(preds, refs):
        scores = scorer.score(ref, pred)
        for key in totals:
            totals[key] += scores[key].fmeasure
    count = max(1, len(preds))
    return {key: value / count for key, value in totals.items()}


def compute_moverscore(preds: list[str], refs: list[str]) -> dict:
    try:
        from moverscore_v2 import get_idf_dict, word_mover_score
    except Exception as exc:
        return {
            "moverscore": None,
            "moverscore_error": f"Could not import moverscore_v2: {exc}",
        }

    idf_refs = get_idf_dict(refs)
    idf_preds = get_idf_dict(preds)
    scores = word_mover_score(
        refs,
        preds,
        idf_refs,
        idf_preds,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
    )
    clean_scores = [score for score in scores if isinstance(score, float) and not math.isnan(score)]
    return {
        "moverscore": sum(clean_scores) / max(1, len(clean_scores)),
        "moverscore_count": len(clean_scores),
    }


def score_pair(name: str, refs: list[str], preds: list[str]) -> dict:
    count = min(len(refs), len(preds))
    refs = refs[:count]
    preds = preds[:count]
    non_empty = sum(1 for pred in preds if pred.strip())
    metrics = {
        "name": name,
        "count": count,
        "non_empty_predictions": non_empty,
    }
    metrics.update(compute_bleu(preds, refs))
    metrics.update(compute_rouge(preds, refs))
    metrics.update(compute_moverscore(preds, refs))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--references", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refs = load_references(Path(args.references), args.limit)
    baseline = load_predictions(Path(args.baseline), args.limit)
    trained = load_predictions(Path(args.trained), args.limit)

    result = {
        "references": str(args.references),
        "baseline_predictions": str(args.baseline),
        "trained_predictions": str(args.trained),
        "baseline": score_pair("baseline", refs, baseline),
        "trained": score_pair("trained", refs, trained),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
