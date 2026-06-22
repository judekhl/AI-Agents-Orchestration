"""
quality_eval.py — Evaluate and compare output quality across scenarios.

Reads output_text fields from results/raw/*_metrics.json files and
produces a quality comparison in results/processed/quality_scores.json.

Metrics:
  - Output length (tokens)
  - ROUGE-1 / ROUGE-L vs a reference answer (if provided)
  - Simple coherence flags: repetition ratio, empty output, truncation

Usage:
    python src/quality_eval.py --input-dir results/raw/ \
        --output results/processed/quality_scores.json \
        --reference-file experiments/prompts/reference_answer.txt
"""

import argparse
import json
import re
from pathlib import Path


def _repetition_ratio(text: str) -> float:
    """Fraction of tokens that are duplicates of the immediately preceding token."""
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    return round(repeats / (len(tokens) - 1), 4)


def _is_truncated(text: str) -> bool:
    """Heuristic: output ends mid-sentence (no terminal punctuation)."""
    stripped = text.strip()
    return bool(stripped) and stripped[-1] not in ".!?\"'"


def evaluate_output(output_text: str, reference: str = "") -> dict:
    scores = {
        "output_length_chars": len(output_text),
        "output_length_tokens_approx": len(output_text.split()),
        "is_empty": len(output_text.strip()) == 0,
        "repetition_ratio": _repetition_ratio(output_text),
        "looks_truncated": _is_truncated(output_text),
    }

    if reference:
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            result = scorer.score(reference, output_text)
            scores["rouge1_f"] = round(result["rouge1"].fmeasure, 4)
            scores["rougeL_f"] = round(result["rougeL"].fmeasure, 4)
        except ImportError:
            scores["rouge_note"] = "rouge-score not installed; run: pip install rouge-score"

    return scores


def load_output_from_metrics(path: Path) -> dict:
    """Extract scenario label and output_text from a metrics JSON file."""
    with open(path) as f:
        data = json.load(f)
    return {
        "scenario": data.get("scenario", path.stem),
        "model_id": data.get("model_id", "unknown"),
        "output_text": data.get("extra", {}).get("output_text", ""),
        "prompt": data.get("extra", {}).get("prompt", ""),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate output quality across scenarios.")
    parser.add_argument("--input-dir", default="results/raw/")
    parser.add_argument("--output", default="results/processed/quality_scores.json")
    parser.add_argument("--reference-file", default="", help="Path to reference answer .txt")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    metric_files = sorted(input_dir.glob("*_metrics.json"))

    if not metric_files:
        print(f"No *_metrics.json files found in {input_dir}")
        print("Run experiments first, then run quality_eval.py")
        return

    reference = ""
    if args.reference_file and Path(args.reference_file).exists():
        reference = Path(args.reference_file).read_text(encoding="utf-8").strip()

    results = []
    for path in metric_files:
        try:
            entry = load_output_from_metrics(path)
            quality = evaluate_output(entry["output_text"], reference)
            results.append({
                "scenario": entry["scenario"],
                "model_id": entry["model_id"],
                "quality": quality,
            })
            print(f"  {entry['scenario']}: "
                  f"length={quality['output_length_tokens_approx']} tokens, "
                  f"repetition={quality['repetition_ratio']}, "
                  f"truncated={quality['looks_truncated']}")
        except Exception as e:
            print(f"  SKIPPED {path.name}: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nQuality scores saved: {out_path}")


if __name__ == "__main__":
    main()
