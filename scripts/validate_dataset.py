
"""Dataset schema validator for PromptPressure Eval Suite.

Usage:
    python scripts/validate_dataset.py evals_dataset.json
"""

import json
import sys
from collections import Counter
from pathlib import Path

REQUIRED_KEYS = {"category", "input", "expected_behavior", "eval_criteria"}
OPTIONAL_KEYS = {"notes"}
ALLOWED_KEYS = REQUIRED_KEYS | OPTIONAL_KEYS


def validate_entry(entry: dict, idx: int) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_KEYS - entry.keys()
    extra = set(entry.keys()) - ALLOWED_KEYS
    if missing:
        errors.append(f"Entry {idx}: missing keys {sorted(missing)}")

    if extra:
        errors.append(f"Entry {idx}: extra keys {sorted(extra)}")

    # Basic sanity: ensure non‑empty string values
    for key in REQUIRED_KEYS:
        if not isinstance(entry.get(key), str) or not entry[key].strip():
            errors.append(f"Entry {idx}: '{key}' must be a non‑empty string")

    return errors


def main(path: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        print(f"✗ Dataset file not found: {path}", file=sys.stderr)

        sys.exit(1)

    try:
        data = json.loads(file_path.read_text(encoding="utf‑8"))
    except json.JSONDecodeError as exc:
        print(f"✗ JSON parse error: {exc}", file=sys.stderr)

        sys.exit(1)

    if not isinstance(data, list):
        print("✗ Top‑level JSON must be a list of entries", file=sys.stderr)

        sys.exit(1)

    all_errors: list[str] = []
    prompt_counter = Counter()

    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            all_errors.append(f"Entry {idx}: must be an object / dict")

            continue
        all_errors.extend(validate_entry(entry, idx))
        prompt_counter[entry.get("input", "")] += 1

    duplicates = [p for p, cnt in prompt_counter.items() if cnt > 1]
    if duplicates:
        all_errors.append(f"✗ Duplicate input prompts detected: {len(duplicates)} duplicates")


    if all_errors:
        print("\n".join(all_errors), file=sys.stderr)

        print(f"\n✗ Validation failed with {len(all_errors)} error(s).", file=sys.stderr)

        sys.exit(1)

    print(f"✓ Dataset {path} is valid — {len(data)} entries, no duplicates.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <dataset_path>", file=sys.stderr)

        sys.exit(1)
    main(sys.argv[1])
