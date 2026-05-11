"""
Tree-sitter based code parser.
Parses Python and JavaScript source code into AST-aware chunks.
"""

import re
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class CodeChunk:
    code: str
    language: str
    function_name: str
    start_line: int
    end_line: int
    complexity: int  # cyclomatic complexity estimate
    has_loops: bool
    has_try_except: bool
    num_params: int
    num_lines: int


def estimate_complexity(code: str) -> int:
    """Estimate cyclomatic complexity by counting branches."""
    keywords = ['if ', 'elif ', 'else:', 'for ', 'while ',
                'except', 'and ', 'or ', 'case ']
    return 1 + sum(code.count(kw) for kw in keywords)


def extract_python_functions(source_code: str) -> List[CodeChunk]:
    """Extract function-level chunks from Python source using regex."""
    chunks = []
    lines = source_code.split('\n')

    func_pattern = re.compile(r'^(def\s+(\w+)\s*\(([^)]*)\)\s*:)', re.MULTILINE)
    matches = list(func_pattern.finditer(source_code))

    for i, match in enumerate(matches):
        start_pos = match.start()
        end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(source_code)

        func_code = source_code[start_pos:end_pos].strip()
        func_name = match.group(2)
        params_str = match.group(3)
        num_params = len([p for p in params_str.split(',') if p.strip()]) if params_str.strip() else 0

        start_line = source_code[:start_pos].count('\n') + 1
        end_line = start_line + func_code.count('\n')

        chunk = CodeChunk(
            code=func_code,
            language="python",
            function_name=func_name,
            start_line=start_line,
            end_line=end_line,
            complexity=estimate_complexity(func_code),
            has_loops=any(kw in func_code for kw in ['for ', 'while ']),
            has_try_except='try:' in func_code or 'except' in func_code,
            num_params=num_params,
            num_lines=func_code.count('\n') + 1
        )
        chunks.append(chunk)

    return chunks


def extract_javascript_functions(source_code: str) -> List[CodeChunk]:
    """Extract function-level chunks from JavaScript source using regex."""
    chunks = []

    patterns = [
        re.compile(r'function\s+(\w+)\s*\(([^)]*)\)\s*\{', re.MULTILINE),
        re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>\s*\{', re.MULTILINE),
        re.compile(r'(?:const|let|var)\s+(\w+)\s*=\s*function\s*\(([^)]*)\)\s*\{', re.MULTILINE),
    ]

    for pattern in patterns:
        for match in pattern.finditer(source_code):
            func_name = match.group(1)
            params_str = match.group(2)
            num_params = len([p for p in params_str.split(',') if p.strip()]) if params_str.strip() else 0

            # Find matching closing brace
            start = match.start()
            brace_count = 0
            end = start
            for j, ch in enumerate(source_code[match.end() - 1:], start=match.end() - 1):
                if ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = j + 1
                        break

            func_code = source_code[start:end].strip()
            start_line = source_code[:start].count('\n') + 1

            chunk = CodeChunk(
                code=func_code,
                language="javascript",
                function_name=func_name,
                start_line=start_line,
                end_line=start_line + func_code.count('\n'),
                complexity=estimate_complexity(func_code),
                has_loops=any(kw in func_code for kw in ['for(', 'for (', 'while(', 'while (']),
                has_try_except='try {' in func_code or 'try{' in func_code,
                num_params=num_params,
                num_lines=func_code.count('\n') + 1
            )
            chunks.append(chunk)

    return chunks


def parse_code(source_code: str, language: str) -> List[CodeChunk]:
    """Main entry point. Parse source code into function chunks."""
    language = language.lower().strip()
    if language == "python":
        chunks = extract_python_functions(source_code)
    elif language in ("javascript", "js"):
        chunks = extract_javascript_functions(source_code)
    else:
        # Fallback: treat entire code as one chunk
        chunks = [CodeChunk(
            code=source_code,
            language=language,
            function_name="<module>",
            start_line=1,
            end_line=source_code.count('\n') + 1,
            complexity=estimate_complexity(source_code),
            has_loops=any(kw in source_code for kw in ['for ', 'while ']),
            has_try_except='try' in source_code,
            num_params=0,
            num_lines=source_code.count('\n') + 1
        )]
    return chunks


if __name__ == "__main__":
    # Quick test
    sample = '''
def divide(a, b):
    result = a / b
    return result

def process_list(items):
    for i in range(len(items) + 1):
        print(items[i])

def read_file(path):
    f = open(path)
    data = f.read()
    return data
'''
    chunks = parse_code(sample, "python")
    for c in chunks:
        print(f"Function: {c.function_name} | Lines: {c.num_lines} | Complexity: {c.complexity}")
        print(f"  Has loops: {c.has_loops} | Has try/except: {c.has_try_except}")
        print()