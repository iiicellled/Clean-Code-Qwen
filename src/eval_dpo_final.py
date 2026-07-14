from __future__ import annotations

import argparse
import ast
import gc
import json
import re
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


SYSTEM_PROMPT = (
    "You are a concise Python code generation assistant. Given a function "
    "specification, output only correct, readable Python code. Correctness and "
    "edge cases are more important than being extremely short. Avoid Markdown "
    "or explanations."
)
CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare base, SFT, and DPO models on DPO final validation hard metrics."
    )
    parser.add_argument("--base-model", default="models/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--sft-adapter", default="output_models/qwen-coder-simplifier-lora")
    parser.add_argument("--dpo-adapter", default="output_models/qwen-coder-simplifier-dpo-lora")
    parser.add_argument(
        "--tasks",
        default="data/python_code_simplification/dpo/python_simple_coder_dpo_final_valid.jsonl",
    )
    parser.add_argument("--output-dir", default="output_results/dpo-final-evaluation")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("base", "sft", "dpo"),
        default=["base", "sft", "dpo"],
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_tasks(path: Path, limit: int | None) -> list[dict[str, Any]]:
    tasks = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"prompt", "chosen", "rejected"} - item.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            tasks.append(
                {
                    "id": f"dpo_final_{line_number:05d}",
                    "source_id": item.get("id"),
                    "prompt": str(item["prompt"]).strip(),
                    "chosen": str(item["chosen"]).strip(),
                    "rejected": str(item["rejected"]).strip(),
                    "pair_type": str(item.get("pair_type", "unknown")),
                    "error_type": str(item.get("error_type", "unknown")),
                }
            )
            if limit is not None and len(tasks) >= limit:
                break
    if not tasks:
        raise ValueError(f"No DPO final validation tasks found in {path}")
    return tasks


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


def model_path(kind: str, args: argparse.Namespace) -> str:
    if kind == "base":
        return args.base_model
    if kind == "sft":
        return args.sft_adapter
    if kind == "dpo":
        return args.dpo_adapter
    raise ValueError(f"Unknown model kind: {kind}")


def load_model(kind: str, args: argparse.Namespace):
    torch, peft_model_class, base_model_class, tokenizer_class = import_model_dependencies()
    path = model_path(kind, args)
    tokenizer = tokenizer_class.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_class = base_model_class if kind == "base" else peft_model_class
    model = model_class.from_pretrained(
        path,
        device_map="auto",
        torch_dtype=model_dtype(torch),
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    torch, _, _, _ = import_model_dependencies()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
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


def generate_model_outputs(
    kind: str, tasks: list[dict[str, Any]], args: argparse.Namespace, output_dir: Path
) -> None:
    path = generation_path(output_dir, kind)
    cached = {} if args.overwrite else load_cached_generations(path)
    pending = [task for task in tasks if task["id"] not in cached]
    if not pending:
        print(f"[{kind}] Reusing {len(tasks)} cached generations from {path}")
        return

    print(f"[{kind}] Loading model from {model_path(kind, args)}; {len(pending)} task(s) pending")
    model, tokenizer = load_model(kind, args)
    mode = "w" if args.overwrite else "a"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, task in enumerate(pending, start=1):
            started = time.perf_counter()
            response = generate(model, tokenizer, task["prompt"], args.max_new_tokens)
            item = {
                "task_id": task["id"],
                "source_id": task.get("source_id"),
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


def parse_module(code: str) -> ast.Module | None:
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = [arg.arg for arg in node.args.posonlyargs + node.args.args]
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return args


def first_interface_signature(code: str) -> tuple[str, Any] | None:
    tree = parse_module(code)
    if tree is None:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "function", node.name, function_args(node)
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append((child.name, function_args(child)))
            return "class", node.name, methods
    return None

def import_count(code: str) -> int:
    tree = parse_module(code)
    if tree is None:
        return 0
    return sum(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree))


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


def score_model(
    kind: str, tasks: list[dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    generations = load_cached_generations(generation_path(output_dir, kind))
    results = []
    for task in tasks:
        generated = generations.get(task["id"])
        if generated is None:
            raise RuntimeError(f"Missing {kind} generation for {task['id']}")

        response = generated["response"]
        code = extract_code(response)
        chosen_code = task["chosen"]
        expected_signature = first_interface_signature(chosen_code)
        generated_signature = first_interface_signature(code)
        syntax_valid = parse_module(code) is not None
        has_interface = generated_signature is not None
        name_match = (
            expected_signature is not None
            and generated_signature is not None
            and generated_signature[0] == expected_signature[0]
        )
        signature_match = expected_signature is not None and generated_signature == expected_signature
        lines = non_empty_lines(code)
        chars = len(code.strip())
        chosen_lines = non_empty_lines(chosen_code)
        chosen_chars = len(chosen_code.strip())

        results.append(
            {
                "task_id": task["id"],
                "source_id": task.get("source_id"),
                "pair_type": task["pair_type"],
                "error_type": task["error_type"],
                "prompt": task["prompt"],
                "chosen": chosen_code,
                "rejected": task["rejected"],
                "response": response,
                "code": code,
                "generation_seconds": generated["generation_seconds"],
                "syntax_valid": syntax_valid,
                "has_interface": has_interface,
                "name_match": name_match,
                "signature_match": signature_match,
                "expected_signature": expected_signature,
                "generated_signature": generated_signature,
                "code_only": is_code_only(response, code),
                "imports": import_count(code),
                "lines": lines,
                "chars": chars,
                "chosen_lines": chosen_lines,
                "chosen_chars": chosen_chars,
                "line_delta_vs_chosen": lines - chosen_lines,
                "char_delta_vs_chosen": chars - chosen_chars,
            }
        )

    total = len(results)
    categories = {}
    for field in ("pair_type", "error_type"):
        buckets = {}
        for value in sorted({item[field] for item in results}):
            items = [item for item in results if item[field] == value]
            count = len(items)
            buckets[value] = {
                "total": count,
                "syntax_valid_rate": sum(item["syntax_valid"] for item in items) / count,
                "name_match_rate": sum(item["name_match"] for item in items) / count,
                "signature_match_rate": sum(item["signature_match"] for item in items) / count,
                "code_only_rate": sum(item["code_only"] for item in items) / count,
                "avg_lines": mean(item["lines"] for item in items),
                "avg_chars": mean(item["chars"] for item in items),
            }
        categories[field] = buckets

    return {
        "model": kind,
        "total": total,
        "syntax_valid": sum(item["syntax_valid"] for item in results),
        "syntax_valid_rate": sum(item["syntax_valid"] for item in results) / total,
        "has_interface": sum(item["has_interface"] for item in results),
        "has_interface_rate": sum(item["has_interface"] for item in results) / total,
        "name_match": sum(item["name_match"] for item in results),
        "name_match_rate": sum(item["name_match"] for item in results) / total,
        "signature_match": sum(item["signature_match"] for item in results),
        "signature_match_rate": sum(item["signature_match"] for item in results) / total,
        "code_only": sum(item["code_only"] for item in results),
        "code_only_rate": sum(item["code_only"] for item in results) / total,
        "avg_lines": mean(item["lines"] for item in results),
        "avg_chars": mean(item["chars"] for item in results),
        "avg_imports": mean(item["imports"] for item in results),
        "avg_line_delta_vs_chosen": mean(item["line_delta_vs_chosen"] for item in results),
        "avg_char_delta_vs_chosen": mean(item["char_delta_vs_chosen"] for item in results),
        "avg_generation_seconds": mean(item["generation_seconds"] for item in results),
        "status_counts": {
            "syntax_invalid": sum(not item["syntax_valid"] for item in results),
            "missing_interface": sum(not item["has_interface"] for item in results),
            "name_mismatch": sum(not item["name_match"] for item in results),
            "signature_mismatch": sum(not item["signature_match"] for item in results),
            "not_code_only": sum(not item["code_only"] for item in results),
        },
        "by": categories,
        "results": results,
    }


def write_markdown(reports: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# DPO Final Validation Evaluation",
        "",
        "This evaluation does not execute functional assertions. It measures syntax, format,",
        "and whether the first generated top-level interface matches the `chosen` answer.",
        "",
        "| Model | Syntax | Has Interface | Name | Signature | Code Only | Avg Lines | Avg Chars | Line Delta vs Chosen | Char Delta vs Chosen |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        lines.append(
            f"| {report['model']} | "
            f"{report['syntax_valid']}/{report['total']} ({report['syntax_valid_rate']:.1%}) | "
            f"{report['has_interface']}/{report['total']} ({report['has_interface_rate']:.1%}) | "
            f"{report['name_match']}/{report['total']} ({report['name_match_rate']:.1%}) | "
            f"{report['signature_match']}/{report['total']} ({report['signature_match_rate']:.1%}) | "
            f"{report['code_only']}/{report['total']} ({report['code_only_rate']:.1%}) | "
            f"{report['avg_lines']:.2f} | "
            f"{report['avg_chars']:.1f} | "
            f"{report['avg_line_delta_vs_chosen']:+.2f} | "
            f"{report['avg_char_delta_vs_chosen']:+.1f} |"
        )

    lines.extend(["", "## By Pair Type", ""])
    pair_types = sorted({k for report in reports for k in report["by"]["pair_type"]})
    lines.append("| Pair Type | " + " | ".join(report["model"] for report in reports) + " |")
    lines.append("|---|" + "---:|" * len(reports))
    for pair_type in pair_types:
        values = []
        for report in reports:
            metric = report["by"]["pair_type"].get(pair_type)
            values.append(
                f"sig {metric['signature_match_rate']:.1%}, code {metric['code_only_rate']:.1%}"
                if metric
                else "-"
            )
        lines.append(f"| {pair_type} | " + " | ".join(values) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(reports: list[dict[str, Any]]) -> None:
    print("\nModel       Syntax   HasInt   Name     Signature CodeOnly AvgLines AvgChars DeltaLine DeltaChar")
    print("--------------------------------------------------------------------------------------------")
    for report in reports:
        print(
            f"{report['model']:<11} "
            f"{report['syntax_valid_rate']:>7.1%} "
            f"{report['has_interface_rate']:>7.1%} "
            f"{report['name_match_rate']:>7.1%} "
            f"{report['signature_match_rate']:>9.1%} "
            f"{report['code_only_rate']:>8.1%} "
            f"{report['avg_lines']:>8.2f} "
            f"{report['avg_chars']:>8.1f} "
            f"{report['avg_line_delta_vs_chosen']:>+9.2f} "
            f"{report['avg_char_delta_vs_chosen']:>+9.1f}"
        )


def main() -> None:
    args = parse_args()
    tasks = load_tasks(Path(args.tasks), args.limit)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for kind in args.models:
        generate_model_outputs(kind, tasks, args, output_dir)

    reports = [score_model(kind, tasks, output_dir) for kind in args.models]
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "tasks": str(Path(args.tasks)),
                "base_model": args.base_model,
                "sft_adapter": args.sft_adapter,
                "dpo_adapter": args.dpo_adapter,
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
