from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer


SYSTEM_PROMPT = (
    "You are a concise Python code generation assistant. Given a function "
    "specification, output only correct, readable Python code. Keep necessary "
    "edge cases and avoid Markdown or explanations."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT + LoRA training for concise Python code generation."
    )
    parser.add_argument("--config", type=str, default="configs/sft_lora.yaml")
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


def load_sft_dataset(cfg: dict[str, Any]) -> Dataset:
    local_dataset_path = cfg.get("local_dataset_path") or cfg.get("dataset_name")
    if local_dataset_path and Path(local_dataset_path).exists():
        rows = []
        with open(local_dataset_path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                missing = {"instruction", "output"} - item.keys()
                if missing:
                    raise ValueError(
                        f"{local_dataset_path}:{line_number} missing fields: {sorted(missing)}"
                    )
                rows.append(
                    {
                        "id": str(item.get("id", line_number)),
                        "instruction": str(item["instruction"]).strip(),
                        "output": str(item["output"]).strip(),
                    }
                )
        if not rows:
            raise ValueError(f"No SFT examples found in {local_dataset_path}")
        return Dataset.from_list(rows)

    return load_dataset(cfg["dataset_name"], split=cfg.get("dataset_split", "train"))


def load_jsonl_sft_dataset(path: str | Path) -> Dataset:
    rows = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"instruction", "output"} - item.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            rows.append(
                {
                    "id": str(item.get("id", line_number)),
                    "instruction": str(item["instruction"]).strip(),
                    "output": str(item["output"]).strip(),
                }
            )
    if not rows:
        raise ValueError(f"No SFT examples found in {path}")
    return Dataset.from_list(rows)


def format_dataset(dataset: Dataset, tokenizer: Any, cfg: dict[str, Any]) -> Dataset:
    system_prompt = cfg.get("system_prompt", SYSTEM_PROMPT)

    def format_example(example: dict[str, Any]) -> dict[str, str]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(example["instruction"]).strip()},
            {"role": "assistant", "content": str(example["output"]).strip()},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    return dataset.map(
        format_example,
        remove_columns=dataset.column_names,
        desc="Formatting instruction-output examples with chat template",
    )


def build_lora_config(cfg: dict[str, Any]) -> LoraConfig:
    return LoraConfig(
        r=cfg.get("lora_r", 16),
        lora_alpha=cfg.get("lora_alpha", 32),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules"),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def build_training_arguments(cfg: dict[str, Any], output_dir: Path) -> TrainingArguments:
    kwargs = {
        "output_dir": str(output_dir),
        "num_train_epochs": cfg.get("num_train_epochs", 1),
        "per_device_train_batch_size": cfg.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size": cfg.get("per_device_eval_batch_size", 1),
        "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", 8),
        "learning_rate": cfg.get("learning_rate", 2e-5),
        "warmup_ratio": cfg.get("warmup_ratio", 0.03),
        "weight_decay": cfg.get("weight_decay", 0.0),
        "lr_scheduler_type": cfg.get("lr_scheduler_type", "cosine"),
        "logging_steps": cfg.get("logging_steps", 10),
        "save_steps": cfg.get("save_steps", 100),
        "eval_steps": cfg.get("eval_steps", 100),
        "save_strategy": "steps",
        "save_total_limit": cfg.get("save_total_limit", 2),
        "load_best_model_at_end": cfg.get("load_best_model_at_end", True),
        "metric_for_best_model": cfg.get("metric_for_best_model", "eval_loss"),
        "greater_is_better": cfg.get("greater_is_better", False),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": cfg.get("gradient_checkpointing", True),
        "report_to": cfg.get("report_to", "none"),
        "optim": "paged_adamw_8bit" if cfg.get("load_in_4bit", False) else "adamw_torch",
        "seed": cfg.get("seed", 42),
        "remove_unused_columns": True,
    }

    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def build_completion_collator(tokenizer: Any, cfg: dict[str, Any]):
    if not cfg.get("assistant_only_loss", True):
        return None

    response_template = cfg.get("response_template", "<|im_start|>assistant\n")
    return DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(response_template, add_special_tokens=False),
        tokenizer=tokenizer,
    )


def build_sft_trainer(
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    cfg: dict[str, Any],
) -> SFTTrainer:
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": build_lora_config(cfg),
    }

    signature = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = tokenizer

    optional_values = {
        "dataset_text_field": "text",
        "max_seq_length": cfg.get("max_seq_length", 2048),
        "packing": cfg.get("packing", False),
        "data_collator": build_completion_collator(tokenizer, cfg),
    }
    for key, value in optional_values.items():
        if key in signature.parameters and value is not None:
            kwargs[key] = value

    return SFTTrainer(**kwargs)


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
    tokenizer.truncation_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=build_quantization_config(cfg),
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if cfg.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model)
    elif cfg.get("gradient_checkpointing", True) and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = load_sft_dataset(cfg)
    max_train_samples = cfg.get("max_train_samples")
    if max_train_samples:
        dataset = dataset.shuffle(seed=cfg.get("seed", 42)).select(
            range(min(max_train_samples, len(dataset)))
        )
    dataset = format_dataset(dataset, tokenizer, cfg)
    validation_dataset_path = cfg.get("validation_dataset_path")
    if validation_dataset_path:
        eval_dataset = format_dataset(load_jsonl_sft_dataset(validation_dataset_path), tokenizer, cfg)
        train_dataset = dataset
    else:
        split = dataset.train_test_split(test_size=cfg.get("test_size", 0.05), seed=cfg.get("seed", 42))
        train_dataset = split["train"]
        eval_dataset = split["test"]

    training_args = build_training_arguments(cfg, output_dir)
    trainer = build_sft_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        cfg=cfg,
    )

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

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
