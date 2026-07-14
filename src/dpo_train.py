from __future__ import annotations

import argparse
import inspect
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DPOTrainer

try:
    from trl import DPOConfig
except ImportError:
    DPOConfig = None


SYSTEM_PROMPT = (
    "You are a concise Python code generation assistant. Given a function "
    "specification, output only correct, readable Python code. Correctness and "
    "edge cases are more important than being extremely short. Avoid Markdown "
    "or explanations."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DPO training for concise Python code generation.")
    parser.add_argument("--config", type=str, default="configs/dpo_lora.yaml")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_compute_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported compute dtype: {name}")


def build_quantization_config(cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not cfg.get("load_in_4bit", False):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=get_compute_dtype(cfg.get("bnb_4bit_compute_dtype", "float16")),
        bnb_4bit_use_double_quant=True,
    )


def load_dpo_dataset(path: str | Path) -> Dataset:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"prompt", "chosen", "rejected"} - item.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(
                {
                    "id": str(item.get("id") or f"row_{line_number:05d}"),
                    "prompt": str(item["prompt"]).strip(),
                    "chosen": str(item["chosen"]).strip(),
                    "rejected": str(item["rejected"]).strip(),
                    "pair_type": str(item.get("pair_type", "")),
                    "error_type": str(item.get("error_type", "")),
                }
            )
    if not rows:
        raise ValueError(f"No DPO examples found in {path}")
    return Dataset.from_list(rows)


def format_dpo_dataset(dataset: Dataset, tokenizer: Any, cfg: dict[str, Any]) -> Dataset:
    system_prompt = cfg.get("system_prompt", SYSTEM_PROMPT)

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["prompt"]},
        ]
        chosen_messages = prompt_messages + [{"role": "assistant", "content": example["chosen"]}]
        rejected_messages = prompt_messages + [{"role": "assistant", "content": example["rejected"]}]

        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        chosen_full = tokenizer.apply_chat_template(
            chosen_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        rejected_full = tokenizer.apply_chat_template(
            rejected_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        return {
            "prompt": prompt,
            "chosen": strip_prompt_prefix(chosen_full, prompt),
            "rejected": strip_prompt_prefix(rejected_full, prompt),
        }

    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting DPO pairs with chat template",
    )


def strip_prompt_prefix(full_text: str, prompt: str) -> str:
    if full_text.startswith(prompt):
        return full_text[len(prompt) :].strip()
    marker = "<|im_start|>assistant"
    index = full_text.rfind(marker)
    if index >= 0:
        return full_text[index:].strip()
    return full_text.strip()


def build_training_arguments(cfg: dict[str, Any], output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.get("num_train_epochs", 1),
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": cfg.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 8),
        "learning_rate": cfg.get("learning_rate", 5e-6),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": cfg.get("logging_steps", 10),
        "save_steps": cfg.get("save_steps", 50),
        "eval_steps": cfg.get("eval_steps", 50),
        "save_strategy": "steps",
        "save_total_limit": cfg.get("save_total_limit", 2),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "report_to": cfg.get("report_to", "none"),
        "optim": "paged_adamw_8bit" if cfg.get("load_in_4bit", False) else "adamw_torch",
        "seed": cfg.get("seed", 42),
        "remove_unused_columns": False,
    }

    argument_class = DPOConfig if DPOConfig is not None else TrainingArguments
    signature = inspect.signature(argument_class.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"

    dpo_values = {
        "beta": cfg.get("beta", 0.1),
        "loss_type": cfg.get("loss_type", "sigmoid"),
        "max_length": cfg.get("max_length", 2048),
        "max_prompt_length": cfg.get("max_prompt_length", 1536),
        "max_target_length": cfg.get("max_target_length", 512),
    }
    for key, value in dpo_values.items():
        if key in signature.parameters:
            kwargs[key] = value

    return argument_class(**kwargs)


def make_compatible_adapter_path(adapter_path: str, output_dir: str) -> str:
    source = Path(adapter_path)
    config_path = source / "adapter_config.json"
    if not config_path.exists():
        return adapter_path

    target = Path(output_dir) / "sft_adapter_compat"
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if item.is_file():
            shutil.copy2(item, destination)
        elif item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    allowed_keys = set(inspect.signature(LoraConfig.__init__).parameters)
    allowed_keys.discard("self")
    filtered = {key: value for key, value in config.items() if key in allowed_keys}
    removed = sorted(set(config) - set(filtered))
    (target / "adapter_config.json").write_text(
        json.dumps(filtered, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if removed:
        print(
            "Filtered unsupported LoRA adapter_config key(s) for this PEFT version: "
            + ", ".join(removed)
        )
    return str(target)


def load_policy_and_reference(cfg: dict[str, Any]):
    adapter_path = make_compatible_adapter_path(cfg["sft_adapter_path"], cfg["output_dir"])
    common_kwargs = {
        "quantization_config": build_quantization_config(cfg),
        "device_map": "auto",
        "trust_remote_code": True,
    }

    policy_base = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **common_kwargs)
    reference_base = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **common_kwargs)

    policy_model = PeftModel.from_pretrained(policy_base, adapter_path, is_trainable=True)
    reference_model = PeftModel.from_pretrained(reference_base, adapter_path, is_trainable=False)

    policy_model.config.use_cache = False
    reference_model.config.use_cache = False
    if cfg.get("load_in_4bit", False):
        policy_model = prepare_model_for_kbit_training(policy_model)
    elif cfg.get("gradient_checkpointing", True) and hasattr(policy_model, "enable_input_require_grads"):
        policy_model.enable_input_require_grads()

    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)
    reference_model.eval()
    return policy_model, reference_model


def build_dpo_trainer(
    model: Any,
    ref_model: Any,
    training_args: TrainingArguments,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: dict[str, Any],
) -> DPOTrainer:
    kwargs = {
        "model": model,
        "ref_model": ref_model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
    }

    signature = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer

    optional_values = {
        "beta": cfg.get("beta", 0.1),
        "loss_type": cfg.get("loss_type", "sigmoid"),
        "max_length": cfg.get("max_length", 2048),
        "max_prompt_length": cfg.get("max_prompt_length", 1536),
        "max_target_length": cfg.get("max_target_length", 512),
    }
    for key, value in optional_values.items():
        if key in signature.parameters:
            kwargs[key] = value

    return DPOTrainer(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name_or_path"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    train_dataset = load_dpo_dataset(cfg["dpo_dataset_path"])
    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples:
        train_dataset = train_dataset.shuffle(seed=cfg.get("seed", 42)).select(
            range(min(max_train_samples, len(train_dataset)))
        )

    validation_path = cfg.get("validation_dataset_path")
    if validation_path:
        eval_dataset = load_dpo_dataset(validation_path)
    else:
        split = train_dataset.train_test_split(
            test_size=cfg.get("test_size", 0.05),
            seed=cfg.get("seed", 42),
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]

    train_dataset = format_dpo_dataset(train_dataset, tokenizer, cfg)
    eval_dataset = format_dpo_dataset(eval_dataset, tokenizer, cfg)
    print(f"DPO examples: train={len(train_dataset)}, eval={len(eval_dataset)}")

    model, ref_model = load_policy_and_reference(cfg)
    model.print_trainable_parameters()

    training_args = build_training_arguments(cfg, output_dir)
    trainer = build_dpo_trainer(
        model=model,
        ref_model=ref_model,
        training_args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg=cfg,
    )

    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        try:
            eval_metrics["eval_perplexity"] = math.exp(eval_metrics["eval_loss"])
        except OverflowError:
            eval_metrics["eval_perplexity"] = float("inf")
    trainer.save_metrics("eval", eval_metrics)
    trainer.save_state()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
