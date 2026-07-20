from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "models/Qwen2.5-Coder-7B-Instruct"
DEFAULT_SFT_ADAPTER = "output_models/qwen-coder-simplifier-lora"
DEFAULT_DPO_ADAPTER = "output_models/qwen-coder-simplifier-dpo-lora"
DEFAULT_OUTPUT_DIR = "output_models/qwen-coder-simplifier-dpo-merged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge Qwen2.5-Coder base weights with LoRA adapters into a standalone "
            "Transformers/vLLM-loadable model directory."
        )
    )
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="Base model directory.")
    parser.add_argument("--sft-adapter", default=DEFAULT_SFT_ADAPTER, help="First SFT LoRA adapter directory.")
    parser.add_argument("--dpo-adapter", default=DEFAULT_DPO_ADAPTER, help="Second DPO LoRA adapter directory.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for the merged full model.")
    parser.add_argument(
        "--merge-strategy",
        choices=("final_adapter", "sequential"),
        default="final_adapter",
        help=(
            "final_adapter: merge only the DPO adapter into the base model. This is correct when "
            "DPO training continued from the SFT adapter and saved the updated final LoRA weights. "
            "sequential: merge SFT into base first, then merge DPO into that merged model. Use only "
            "when the DPO adapter is a true delta relative to the SFT-merged model."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="float16",
        help="Load/merge dtype. Use float16 for Tesla V100.",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device_map, e.g. auto, cpu, cuda:0.")
    parser.add_argument("--no-local-files-only", action="store_true", help="Allow downloads from Hugging Face Hub.")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing output directory.")
    return parser.parse_args()


def resolve_project_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.exists() or path.is_absolute():
        return path.resolve()

    # Local runs in this repo historically used output_models/, while the remote server uses outputs/.
    parts = path.parts
    if parts and parts[0] == "outputs":
        fallback = Path("output_models", *parts[1:])
        if fallback.exists():
            print(f"Path {path} not found; using local fallback {fallback}.")
            return fallback.resolve()

    return path.resolve()


def torch_dtype(name: str) -> torch.dtype | str:
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def require_dir(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def adapter_rank(adapter_path: Path) -> int | None:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        return None
    try:
        return int(json.loads(config_path.read_text(encoding="utf-8")).get("r"))
    except Exception:
        return None


def load_base_model(base_model: Path, args: argparse.Namespace) -> Any:
    print(f"Loading base model: {base_model}")
    return AutoModelForCausalLM.from_pretrained(
        str(base_model),
        torch_dtype=torch_dtype(args.dtype),
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.no_local_files_only,
        low_cpu_mem_usage=True,
    )


def merge_one_adapter(model: Any, adapter_path: Path, adapter_name: str, local_files_only: bool) -> Any:
    print(f"Loading {adapter_name} adapter: {adapter_path}")
    rank = adapter_rank(adapter_path)
    if rank is not None:
        print(f"{adapter_name} LoRA rank: {rank}")
    peft_model = PeftModel.from_pretrained(
        model,
        str(adapter_path),
        adapter_name=adapter_name,
        local_files_only=local_files_only,
        is_trainable=False,
    )
    print(f"Merging {adapter_name} adapter into model weights.")
    return peft_model.merge_and_unload()


def save_tokenizer(base_model: Path, output_dir: Path, args: argparse.Namespace) -> None:
    print(f"Saving tokenizer from base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model),
        trust_remote_code=args.trust_remote_code,
        local_files_only=not args.no_local_files_only,
        use_fast=True,
    )
    tokenizer.save_pretrained(str(output_dir))


def main() -> None:
    args = parse_args()
    base_model = resolve_project_path(args.base_model)
    sft_adapter = resolve_project_path(args.sft_adapter)
    dpo_adapter = resolve_project_path(args.dpo_adapter)
    output_dir = Path(args.output_dir).resolve()

    require_dir(base_model, "Base model")
    require_dir(dpo_adapter, "DPO adapter")
    if args.merge_strategy == "sequential":
        require_dir(sft_adapter, "SFT adapter")

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to reuse it.")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Merge strategy: {args.merge_strategy}")
    model = load_base_model(base_model, args)

    if args.merge_strategy == "sequential":
        model = merge_one_adapter(model, sft_adapter, "sft", not args.no_local_files_only)
        model = merge_one_adapter(model, dpo_adapter, "dpo", not args.no_local_files_only)
    else:
        print(
            "Using DPO adapter as the final adapter. This avoids double-counting SFT when DPO "
            "training continued from SFT LoRA weights."
        )
        model = merge_one_adapter(model, dpo_adapter, "dpo_final", not args.no_local_files_only)

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=args.safe_serialization)
    save_tokenizer(base_model, output_dir, args)
    print("Merge complete.")
    print(f"You can now serve this directory directly with vLLM: {output_dir}")


if __name__ == "__main__":
    main()
