#!/usr/bin/env python3

import argparse
import logging
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers import logging as transformers_logging
from transformers import set_seed

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.gpart import GPartConfig
from peft.tuners.unilora import UniLoRAConfig

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request and conclude with: "
    "This is the answer: ANSWER.\n\n"
)

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


def build_training_text(question: str, answer: str, use_system_prompt: bool):
    prefix = SYSTEM_PROMPT if use_system_prompt else ""
    supervised_text = f"Question: {question.strip()}\nAnswer: {answer.strip()}"
    return prefix, supervised_text


def tokenize_example(example, tokenizer, max_seq_length: int, use_system_prompt: bool):
    prefix_text, supervised_text = build_training_text(
        example["query"], example["response"], use_system_prompt
    )

    bos_ids = []
    if tokenizer.bos_token_id is not None:
        bos_ids = [tokenizer.bos_token_id]

    prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    supervised_ids = tokenizer(supervised_text, add_special_tokens=False)["input_ids"]
    eos_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = bos_ids + prefix_ids + supervised_ids + eos_ids
    labels = [-100] * (len(bos_ids) + len(prefix_ids)) + supervised_ids + eos_ids

    if len(input_ids) > max_seq_length:
        input_ids = input_ids[-max_seq_length:]
        labels = labels[-max_seq_length:]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def load_and_prepare_dataset(args, tokenizer):
    dataset = load_dataset(args.dataset_name, split="train")

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = dataset.map(
        lambda ex: tokenize_example(
            ex,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            use_system_prompt=args.use_system_prompt,
        ),
        remove_columns=dataset.column_names,
        num_proc=args.dataset_num_proc,
        desc="Tokenizing MetaMathQA",
    )
    return dataset


@dataclass
class CausalLMDataCollator:
    pad_token_id: int
    label_pad_token_id: int = -100

    def __call__(self, features):
        input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        attention_mask = [
            torch.tensor(f["attention_mask"], dtype=torch.long) for f in features
        ]
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]

        return {
            "input_ids": pad_sequence(
                input_ids, batch_first=True, padding_value=self.pad_token_id
            ),
            "attention_mask": pad_sequence(
                attention_mask, batch_first=True, padding_value=0
            ),
            "labels": pad_sequence(
                labels, batch_first=True, padding_value=self.label_pad_token_id
            ),
        }


def sanitize_name(value: str) -> str:
    value = value.strip().replace("/", "-")
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-._")
    return value.lower()


def format_float(value: float) -> str:
    return f"{value:g}".replace("+", "")


def get_trainable_params_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def write_output_dir_file(output_dir: str, output_dir_file: str) -> Path:
    output_path = Path(output_dir).resolve()
    path_file = Path(output_dir_file)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_file.write_text(f"{output_path}\n", encoding="utf-8")
    return path_file


def build_run_name(args, trainable_params_count: int) -> str:
    parts = [
        "metamath",
        sanitize_name(args.model_name),
        args.adapter_type,
    ]

    if args.adapter_type in {"lora", "unilora"}:
        parts.append(f"r{args.rank}")
    if args.adapter_type == "lora":
        parts.append(f"a{args.alpha}")
    if args.adapter_type in {"unilora", "gpart"}:
        parts.append(f"d{args.d}")

    parts.extend(
        [
            f"drop{format_float(args.dropout)}",
            f"lr{format_float(args.learning_rate)}",
            f"bs{args.per_device_train_batch_size}",
            f"ga{args.gradient_accumulation_steps}",
            f"ep{format_float(args.num_train_epochs)}",
            f"seq{args.max_seq_length}",
            "sysprompt" if args.use_system_prompt else "nosysprompt",
            f"seed{args.seed}",
            f"{trainable_params_count // 1000}k",
        ]
    )

    if args.max_samples is not None:
        parts.append(f"n{args.max_samples}")

    return "_".join(parts)


def build_adapter_config(args):
    if args.adapter_type == "lora":
        return LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=args.dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )

    if args.adapter_type == "unilora":
        return UniLoRAConfig(
            r=args.rank,
            theta_d_length=args.d,
            target_modules=TARGET_MODULES,
            unilora_dropout=args.dropout,
            init_theta_d_bound=args.init_bound,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )

    if args.adapter_type == "gpart":
        return GPartConfig(
            d=args.d,
            target_modules=TARGET_MODULES,
            gpart_dropout=args.dropout,
            init_bound=args.init_bound,
            assignment_backend=args.assignment_backend,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
        )

    raise ValueError(f"Unsupported adapter type: {args.adapter_type}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune pretrained causal LMs on MetaMathQA with LoRA, UniLoRA, or GPart.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Model")
    g.add_argument("--model_name", type=str, default="mistralai/Mistral-7B-v0.1")
    g.add_argument("--eos_token", type=str, default=None)

    g = p.add_argument_group("Adapter")
    g.add_argument(
        "--adapter_type",
        type=str,
        choices=["lora", "unilora", "gpart"],
        default="gpart",
    )
    g.add_argument("--rank", type=int, default=4)
    g.add_argument("--alpha", type=int, default=4)
    g.add_argument("--dropout", type=float, default=0.05)
    g.add_argument("--d", type=int, default=524288)
    g.add_argument("--init_bound", type=float, default=0.0)
    g.add_argument(
        "--assignment_backend",
        type=str,
        choices=[
            "materialized",
            "stateless",
            "legacy_streaming",
            "implicit_stateless_v1",
        ],
        default="materialized",
        help=(
            "GPart random-assignment backend. 'materialized' (default) preserves "
            "the seeded torch.randint mapping; 'stateless' derives assignments from "
            "the seed and canonical global parameter position. Deprecated aliases: "
            "'legacy_streaming' and 'implicit_stateless_v1'."
        ),
    )

    g = p.add_argument_group("Dataset")
    g.add_argument("--dataset_name", type=str, default="meta-math/MetaMathQA")
    g.add_argument("--max_samples", type=int, default=None)
    g.add_argument("--max_seq_length", type=int, default=2048)
    g.add_argument("--dataset_num_proc", type=int, default=8)
    g.add_argument("--use_system_prompt", action="store_true", default=False)

    g = p.add_argument_group("Training")
    g.add_argument("--output_root_dir", type=str, default="./experiments/outputs")
    g.add_argument(
        "--output_dir_file",
        type=str,
        default=None,
        help=(
            "Optional file that receives the absolute final adapter directory after "
            "training and all final saves succeed."
        ),
    )
    g.add_argument("--num_train_epochs", type=float, default=2)
    g.add_argument("--per_device_train_batch_size", type=int, default=2)
    g.add_argument("--gradient_accumulation_steps", type=int, default=8)
    g.add_argument("--gradient_checkpointing", action="store_true", default=False)
    g.add_argument("--learning_rate", type=float, default=2e-4)
    g.add_argument("--lr_scheduler_type", type=str, default="cosine")
    g.add_argument("--warmup_ratio", type=float, default=0.05)
    g.add_argument("--weight_decay", type=float, default=0.01)
    g.add_argument("--max_grad_norm", type=float, default=1.0)
    g.add_argument("--bf16", action="store_true", default=True)
    g.add_argument("--fp16", action="store_true")
    g.add_argument("--logging_steps", type=int, default=10)
    g.add_argument("--save_steps", type=int, default=250)
    g.add_argument("--save_total_limit", type=int, default=2)
    g.add_argument("--report_to", type=str, default="none")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--dataloader_num_workers", type=int, default=8)

    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    set_seed(args.seed)

    if args.bf16 and args.fp16:
        raise ValueError("Choose only one of --bf16 or --fp16.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        padding_side="right",
    )

    if args.eos_token is not None:
        tokenizer.eos_token = args.eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logging.info(f"Loading dataset: {args.dataset_name}")
    train_dataset = load_and_prepare_dataset(args, tokenizer)
    logging.info(f"Training examples: {len(train_dataset):,}")

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    logging.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False

    adapter_config = build_adapter_config(args)
    model = get_peft_model(model, adapter_config)

    trainable_params_count = get_trainable_params_count(model)
    run_name = build_run_name(args, trainable_params_count)
    output_dir = os.path.join(args.output_root_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Run name: {run_name}")
    logging.info(f"Output dir: {output_dir}")
    logging.info(
        f"Trainable parameters: {trainable_params_count:,} ({trainable_params_count // 1000}k)"
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        fp16=args.fp16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        eval_strategy="no",
        # save_safetensors=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=CausalLMDataCollator(pad_token_id=tokenizer.pad_token_id),
        processing_class=tokenizer,
    )

    logging.info("Starting training...")
    train_result = trainer.train()

    logging.info(f"Saving adapter and tokenizer to: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if args.output_dir_file:
        path_file = write_output_dir_file(output_dir, args.output_dir_file)
        logging.info(f"Output directory path written to: {path_file}")

    logging.info("Done.")


if __name__ == "__main__":
    main()
