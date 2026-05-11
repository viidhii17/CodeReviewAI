"""
Inference module — loads trained model and predicts bug type for a code snippet.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.parser.code_parser import parse_code

MODEL_PATH = "models/checkpoints/best_model"
MAX_LENGTH = 512
LABEL_NAMES = ["no_bug", "null_dereference", "off_by_one", "resource_leak", "logic_error"]

SEVERITY = {
    "no_bug": "none",
    "null_dereference": "high",
    "off_by_one": "medium",
    "resource_leak": "high",
    "logic_error": "medium",
}

SUGGESTIONS = {
    "no_bug": "No issues detected. Code looks clean.",
    "null_dereference": "Add a None check before accessing this object. Use 'if obj is not None:' guard.",
    "off_by_one": "Check loop bounds carefully. Consider using 'range(len(x))' instead of 'range(len(x)+1)'.",
    "resource_leak": "Use 'with open(...) as f:' to ensure the file is always closed.",
    "logic_error": "Review comparison operators (==, >, <). Consider adding unit tests for edge cases.",
}

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is None:
        print(f"Loading model from {MODEL_PATH}...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
        if torch.cuda.is_available():
            _model = _model.cuda()
        print("Model loaded.")
    return _model, _tokenizer


def predict_chunk(code: str):
    """Predict bug type for a single code snippet."""
    model, tokenizer = load_model()
    inputs = tokenizer(code, return_tensors="pt", truncation=True,
                       padding="max_length", max_length=MAX_LENGTH)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    bug_type = LABEL_NAMES[pred_idx]

    return {
        "bug_type": bug_type,
        "confidence": float(probs[pred_idx]),
        "severity": SEVERITY[bug_type],
        "suggestion": SUGGESTIONS[bug_type],
        "probabilities": {name: float(probs[i]) for i, name in enumerate(LABEL_NAMES)},
    }


def review_code(source_code: str, language: str = "python"):
    """Full review: parse into chunks, predict each, return all findings."""
    chunks = parse_code(source_code, language)
    findings = []

    for chunk in chunks:
        result = predict_chunk(chunk.code)
        findings.append({
            "function_name": chunk.function_name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "num_lines": chunk.num_lines,
            "complexity": chunk.complexity,
            **result,
        })

    # Sort by severity: high → medium → none
    severity_order = {"high": 0, "medium": 1, "none": 2}
    findings.sort(key=lambda x: severity_order.get(x["severity"], 3))
    return findings