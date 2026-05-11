"""
Dataset builder: downloads CodeSearchNet Python split and prepares
a labeled bug-detection dataset.

Labels:
  0 = no_bug
  1 = null_dereference
  2 = off_by_one
  3 = resource_leak
  4 = logic_error

Strategy:
  - Clean samples from CodeSearchNet → label 0
  - Rule-based injection of synthetic bugs → labels 1-4
  - This gives us a balanced dataset without needing manual annotation.
"""

import random
import re
from typing import List, Dict, Tuple
from datasets import load_dataset, Dataset
import pandas as pd

random.seed(42)

BUG_CLASSES = {
    0: "no_bug",
    1: "null_dereference",
    2: "off_by_one",
    3: "resource_leak",
    4: "logic_error",
}

# ── Synthetic bug injectors ──────────────────────────────────────────────────

def inject_null_dereference(code: str) -> str:
    """Remove a None check to simulate null dereference."""
    patterns = [
        r'if\s+\w+\s+is\s+not\s+None\s*:\n\s*',
        r'if\s+\w+\s*!=\s*None\s*:\n\s*',
        r'if\s+\w+\s*:\n\s*',
    ]
    for pat in patterns:
        match = re.search(pat, code)
        if match:
            return code[:match.start()] + code[match.end():]
    # Fallback: append a None access
    lines = code.split('\n')
    lines.append('    _ = None.attribute  # injected')
    return '\n'.join(lines)


def inject_off_by_one(code: str) -> str:
    """Change range(len(x)) to range(len(x)+1) or len(x)-1."""
    if 'range(len(' in code:
        return code.replace('range(len(', 'range(len(', 1).replace(
            'range(len(', 'range(1 + len(', 1)
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if 'for ' in line and ' in ' in line:
            lines[i] = line + '  # off-by-one risk'
            break
    return '\n'.join(lines)


def inject_resource_leak(code: str) -> str:
    """Replace 'with open' pattern with bare open (no close)."""
    if 'with open(' in code:
        return code.replace('with open(', 'f = open(', 1).replace(
            ') as f:', ')', 1)
    lines = code.split('\n')
    lines.append('    leaked = open("/tmp/leak.txt", "w")  # injected')
    return '\n'.join(lines)


def inject_logic_error(code: str) -> str:
    """Flip a comparison operator."""
    replacements = [('==', '!='), ('>', '<'), ('<', '>'),
                    ('>=', '<='), ('<=', '>=')]
    for old, new in replacements:
        if f' {old} ' in code:
            return code.replace(f' {old} ', f' {new} ', 1)
    lines = code.split('\n')
    lines.append('    if True == False: pass  # injected logic error')
    return '\n'.join(lines)


INJECTORS = {
    1: inject_null_dereference,
    2: inject_off_by_one,
    3: inject_resource_leak,
    4: inject_logic_error,
}


# ── Main builder ─────────────────────────────────────────────────────────────

def build_dataset(max_samples_per_class: int = 2000) -> pd.DataFrame:
    """
    Load CodeSearchNet Python, extract clean functions,
    then inject synthetic bugs for classes 1-4.
    Returns a balanced DataFrame.
    """
    print("Loading CodeSearchNet (Python)...")
    raw = load_dataset(
        "code_search_net",
        "python",
        split="train",
        trust_remote_code=True
    )

    # Extract clean function bodies
    clean_codes = []
    for item in raw:
        code = item.get("func_code_string", "")
        if 50 < len(code) < 2000 and 'def ' in code:
            clean_codes.append(code)
        if len(clean_codes) >= max_samples_per_class * 5:
            break

    random.shuffle(clean_codes)
    print(f"Collected {len(clean_codes)} clean functions")

    rows = []

    # Class 0: clean code
    for code in clean_codes[:max_samples_per_class]:
        rows.append({"code": code, "label": 0,
                     "bug_type": "no_bug", "language": "python"})

    # Classes 1-4: injected bugs
    pool = clean_codes[max_samples_per_class:]
    per_class = max_samples_per_class
    for label, injector in INJECTORS.items():
        subset = pool[:per_class]
        pool = pool[per_class:]
        for code in subset:
            try:
                buggy = injector(code)
                rows.append({"code": buggy, "label": label,
                             "bug_type": BUG_CLASSES[label], "language": "python"})
            except Exception:
                pass

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nDataset built: {len(df)} total samples")
    print(df["bug_type"].value_counts())
    return df


def save_dataset(df: pd.DataFrame, path: str = "data/processed/dataset.csv"):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved to {path}")


if __name__ == "__main__":
    df = build_dataset(max_samples_per_class=2000)
    save_dataset(df)