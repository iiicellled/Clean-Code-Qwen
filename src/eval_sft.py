from __future__ import annotations

import argparse
import ast
import gc
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


SYSTEM_PROMPT = (
    "You are a concise Python code generation assistant. Given a function "
    "specification, output only correct, readable Python code. Keep necessary "
    "edge cases and avoid Markdown or explanations."
)
CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
ASSERT_RE = re.compile(r"^\s*-?\s*(assert\s+.+)$", re.MULTILINE)
SIGNATURE_RE = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")
EXACT_SIGNATURE_RE = re.compile(
    r"exact signature:\s*\n\s*(def\s+[A-Za-z_]\w*\s*\([^)]*\))",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base and SFT models on simple Python code generation hard metrics."
    )
    parser.add_argument("--base-model", default="models/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--adapter", default="output_models/qwen-coder-simplifier-lora")
    parser.add_argument(
        "--tasks",
        default="data/python_code_simplification/sft/python_simple_coder_valid.jsonl.jsonl",
    )
    parser.add_argument(
        "--tests",
        default="data/python_code_simplification/sft/python_simple_coder_valid_tests.jsonl",
        help="Required JSONL with task_id and asserts fields for functional evaluation.",
    )
    parser.add_argument("--output-dir", default="output_results/sft-evaluation")
    parser.add_argument("--models", nargs="+", choices=("base", "sft"), default=["base", "sft"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"instruction"} - item.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(
                {
                    "id": f"valid_{line_number:05d}",
                    "source_id": item.get("id"),
                    "instruction": str(item["instruction"]).strip(),
                    "output": str(item.get("output", "")).strip(),
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No tasks found in {path}")
    return rows


def load_manual_tests(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Functional test file not found: {path}")
    tests = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            task_id = item.get("task_id") or item.get("id")
            asserts = item.get("asserts", [])
            if not task_id or not isinstance(asserts, list):
                raise ValueError(f"{path}:{line_number} must contain task_id and asserts list")
            cleaned_asserts = [str(test).strip() for test in asserts if str(test).strip()]
            if not cleaned_asserts:
                raise ValueError(f"{path}:{line_number} has no functional asserts")
            task_id = str(task_id)
            if task_id in tests:
                raise ValueError(f"{path}:{line_number} duplicate task_id: {task_id}")
            tests[task_id] = cleaned_asserts
    if not tests:
        raise ValueError(f"No functional tests found in {path}")
    return tests


def validate_test_coverage(tasks: list[dict[str, Any]], tests_by_id: dict[str, list[str]]) -> None:
    task_ids = {task["id"] for task in tasks}
    test_ids = set(tests_by_id)
    missing = sorted(task_ids - test_ids)
    extra = sorted(test_ids - task_ids)
    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(f"Functional tests missing {len(missing)} task(s): {sample}")
    if extra:
        sample = ", ".join(extra[:10])
        raise ValueError(f"Functional tests contain {len(extra)} unknown task_id(s): {sample}")

def extract_asserts(instruction: str) -> list[str]:
    return [match.strip().rstrip("`") for match in ASSERT_RE.findall(instruction)]


def expected_signature(instruction: str) -> tuple[str, list[str]] | None:
    exact = EXACT_SIGNATURE_RE.search(instruction)
    if not exact:
        return None
    match = SIGNATURE_RE.search(exact.group(1))
    if not match:
        return None
    args = []
    for arg in match.group(2).split(","):
        name = arg.strip()
        if not name:
            continue
        name = name.split(":", 1)[0].split("=", 1)[0].strip()
        if name in {"*", "/"}:
            continue
        if name.startswith("**"):
            name = name[2:]
        elif name.startswith("*"):
            name = name[1:]
        args.append(name)
    return match.group(1), args

def generated_signature(code: str) -> tuple[str, list[str]] | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
            args.extend(arg.arg for arg in node.args.kwonlyargs)
            if node.args.vararg:
                args.append(node.args.vararg.arg)
            if node.args.kwarg:
                args.append(node.args.kwarg.arg)
            return node.name, args
    return None


def import_model_dependencies():
    try:
        import torch
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model evaluation dependencies are missing. Activate the training environment "
            "and run `pip install -r requirements.txt`."
        ) from exc
    return torch, AutoPeftModelForCausalLM, AutoModelForCausalLM, AutoTokenizer


def model_dtype(torch):
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_model(kind: str, args: argparse.Namespace):
    torch, peft_model_class, base_model_class, tokenizer_class = import_model_dependencies()
    model_path = args.base_model if kind == "base" else args.adapter
    tokenizer = tokenizer_class.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_class = base_model_class if kind == "base" else peft_model_class
    model = model_class.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=model_dtype(torch),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction: str, max_new_tokens: int) -> str:
    torch, _, _, _ = import_model_dependencies()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def generation_path(output_dir: Path, kind: str) -> Path:
    return output_dir / f"{kind}_generations.jsonl"


def load_cached_generations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cached = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                cached[item["task_id"]] = item
    return cached


def generate_model_outputs(kind: str, tasks: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path) -> None:
    path = generation_path(output_dir, kind)
    cached = {} if args.overwrite else load_cached_generations(path)
    pending = [task for task in tasks if task["id"] not in cached]
    if not pending:
        print(f"[{kind}] Reusing {len(tasks)} cached generations from {path}")
        return

    print(f"[{kind}] Loading model; {len(pending)} task(s) pending")
    model, tokenizer = load_model(kind, args)
    mode = "w" if args.overwrite else "a"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(pending, start=1):
            started = time.perf_counter()
            response = generate(model, tokenizer, task["instruction"], args.max_new_tokens)
            item = {
                "task_id": task["id"],
                "response": response,
                "generation_seconds": round(time.perf_counter() - started, 3),
            }
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[{kind}] {index}/{len(pending)} {task['id']}")

    del model, tokenizer
    gc.collect()
    torch, _, _, _ = import_model_dependencies()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_code(response: str) -> str:
    blocks = CODE_BLOCK_RE.findall(response)
    if blocks:
        return max(blocks, key=len).strip()
    code = response.strip()
    if code.lower().startswith("python\n"):
        code = code.split("\n", 1)[1].strip()
    return code


def syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def non_empty_lines(code: str) -> int:
    return sum(1 for line in code.splitlines() if line.strip())


def is_code_only(response: str, code: str) -> bool:
    if "```" in response:
        return False
    stripped = code.strip()
    if not stripped:
        return False
    prose_markers = ("Here is", "Explanation", "This function", "The code")
    return not any(marker.lower() in stripped.lower() for marker in prose_markers)


def import_count(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0
    return sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


def run_asserts(code: str, asserts: list[str], timeout: float) -> dict[str, Any]:
    if not asserts:
        return {"tested": False, "passed": False, "status": "no_tests", "error": ""}
    if not syntax_valid(code):
        return {"tested": True, "passed": False, "status": "invalid_syntax", "error": ""}
    program = code + "\n\n# Functional assertions\n" + "\n".join(asserts) + "\n"
    try:
        with tempfile.TemporaryDirectory(prefix="simple_coder_eval_") as temp_dir:
            script = Path(temp_dir) / "candidate.py"
            script.write_text(program, encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
    except subprocess.TimeoutExpired:
        return {"tested": True, "passed": False, "status": "timeout", "error": f">{timeout}s"}
    if completed.returncode == 0:
        return {"tested": True, "passed": True, "status": "passed", "error": ""}
    return {
        "tested": True,
        "passed": False,
        "status": "failed_asserts",
        "error": (completed.stderr or completed.stdout).strip()[-2000:],
    }


def score_model(kind: str, tasks: list[dict[str, Any]], tests_by_id: dict[str, list[str]], args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    generations = load_cached_generations(generation_path(output_dir, kind))
    results = []
    for task in tasks:
        generated = generations.get(task["id"])
        if generated is None:
            raise RuntimeError(f"Missing {kind} generation for {task['id']}")
        response = generated["response"]
        code = extract_code(response)
        expected = expected_signature(task["instruction"])
        actual = generated_signature(code)
        signature_specified = expected is not None
        signature_match = signature_specified and actual == expected
        name_match = signature_specified and actual is not None and actual[0] == expected[0]
        asserts = tests_by_id[task["id"]]
        functional = run_asserts(code, asserts, args.timeout)
        result = {
            "task_id": task["id"],
            "source_id": task.get("source_id"),
            "response": response,
            "code": code,
            "generation_seconds": generated["generation_seconds"],
            "syntax_valid": syntax_valid(code),
            "has_function": actual is not None,
            "signature_specified": signature_specified,
            "name_match": name_match,
            "signature_match": signature_match,
            "code_only": is_code_only(response, code),
            "imports": import_count(code),
            "lines": non_empty_lines(code),
            "chars": len(code.strip()),
            "assert_count": len(asserts),
            "functional_tested": functional["tested"],
            "functional_passed": functional["passed"],
            "functional_status": functional["status"],
            "functional_error": functional["error"],
            "expected_signature": expected,
            "generated_signature": actual,
        }
        results.append(result)

    total = len(results)
    tested = sum(item["functional_tested"] for item in results)
    functional_passed = sum(item["functional_passed"] for item in results)
    signature_specified = sum(item["signature_specified"] for item in results)
    name_match = sum(item["name_match"] for item in results)
    signature_match = sum(item["signature_match"] for item in results)
    return {
        "model": kind,
        "total": total,
        "syntax_valid": sum(item["syntax_valid"] for item in results),
        "syntax_valid_rate": sum(item["syntax_valid"] for item in results) / total,
        "has_function": sum(item["has_function"] for item in results),
        "has_function_rate": sum(item["has_function"] for item in results) / total,
        "signature_specified": signature_specified,
        "name_match": name_match,
        "name_match_rate": name_match / signature_specified if signature_specified else None,
        "signature_match": signature_match,
        "signature_match_rate": signature_match / signature_specified if signature_specified else None,
        "code_only": sum(item["code_only"] for item in results),
        "code_only_rate": sum(item["code_only"] for item in results) / total,
        "functional_tested": tested,
        "functional_passed": functional_passed,
        "functional_pass_rate": functional_passed / tested if tested else None,
        "avg_lines": mean(item["lines"] for item in results),
        "avg_chars": mean(item["chars"] for item in results),
        "avg_imports": mean(item["imports"] for item in results),
        "avg_generation_seconds": mean(item["generation_seconds"] for item in results),
        "functional_status_counts": dict(Counter(item["functional_status"] for item in results)),
        "results": results,
    }

def format_rate(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def write_markdown(reports: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Simple Coder Evaluation",
        "",
        "Hard metrics do not require reference answers. Functional pass rate is reported only",
        "on tasks with manually supplied or prompt-extracted assertions.",
        "",
        "| Model | Syntax | Has Fn | Name* | Signature* | Code Only | Functional | Avg Lines | Avg Chars | Avg Imports |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        functional = (
            f"{report['functional_passed']}/{report['functional_tested']} "
            f"({format_rate(report['functional_pass_rate'])})"
            if report["functional_tested"]
            else "-"
        )
        lines.append(
            f"| {report['model']} | "
            f"{report['syntax_valid']}/{report['total']} ({report['syntax_valid_rate']:.1%}) | "
            f"{report['has_function']}/{report['total']} ({report['has_function_rate']:.1%}) | "
            f"{report['name_match']}/{report['total']} ({report['name_match_rate']:.1%}) | "
            f"{report['signature_match']}/{report['total']} ({report['signature_match_rate']:.1%}) | "
            f"{report['code_only']}/{report['total']} ({report['code_only_rate']:.1%}) | "
            f"{functional} | "
            f"{report['avg_lines']:.2f} | {report['avg_chars']:.1f} | {report['avg_imports']:.2f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(reports: list[dict[str, Any]]) -> None:
    print("\nModel       Syntax   HasFn    Name*    Signature* CodeOnly Functional AvgLines AvgChars")
    print("-----------------------------------------------------------------------------------")
    for report in reports:
        functional = format_rate(report["functional_pass_rate"])
        print(
            f"{report['model']:<11} "
            f"{report['syntax_valid_rate']:>7.1%} "
            f"{report['has_function_rate']:>7.1%} "
            f"{report['name_match_rate']:>7.1%} "
            f"{report['signature_match_rate']:>9.1%} "
            f"{report['code_only_rate']:>8.1%} "
            f"{functional:>10} "
            f"{report['avg_lines']:>8.2f} "
            f"{report['avg_chars']:>8.1f}"
        )


def main() -> None:
    args = parse_args()
    tasks = load_tasks(Path(args.tasks), args.limit)
    tests_by_id = load_manual_tests(Path(args.tests))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "Warning: generated code is executed in a timed subprocess when assertions are available. "
        "Use trusted validation tests only."
    )
    for kind in args.models:
        generate_model_outputs(kind, tasks, args, output_dir)

    reports = [score_model(kind, tasks, tests_by_id, args, output_dir) for kind in args.models]
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "tasks": str(Path(args.tasks)),
                "tests": str(Path(args.tests)),
                "models": reports,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_markdown(reports, output_dir / "report.md")
    print_summary(reports)
    print(f"\nJSON report: {output_dir / 'report.json'}")
    print(f"Markdown report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
