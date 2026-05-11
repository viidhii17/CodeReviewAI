"""
Chunker: converts CodeChunk objects into model-ready feature dicts.
"""

from typing import List, Dict, Any
from src.parser.code_parser import CodeChunk


def chunk_to_features(chunk: CodeChunk) -> Dict[str, Any]:
    """Convert a CodeChunk to a flat dict for model input / dataset row."""
    return {
        "code": chunk.code,
        "language": chunk.language,
        "function_name": chunk.function_name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "complexity": chunk.complexity,
        "has_loops": int(chunk.has_loops),
        "has_try_except": int(chunk.has_try_except),
        "num_params": chunk.num_params,
        "num_lines": chunk.num_lines,
    }


def chunks_to_dataset_rows(chunks: List[CodeChunk], label: int = 0) -> List[Dict[str, Any]]:
    """Convert a list of chunks to dataset rows with an optional label."""
    rows = []
    for chunk in chunks:
        row = chunk_to_features(chunk)
        row["label"] = label
        rows.append(row)
    return rows