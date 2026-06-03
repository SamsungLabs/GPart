#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import jsonlines
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from peft import PeftModel
from utils.math_eval_utils import (
    batch_data,
    extract_answer_number,
    last_boxed_only_string,
    math_equal,
    remove_boxed,
)

MAX_INT = 100000

TRAINING_SYSTEM_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request and conclude with: "
    "This is the answer: ANSWER.\n\n"
)

DATA_PATHS = {
    "gsm8k": "data/math_eval/gsm8k_test.jsonl",
    "math": "data/math_eval/MATH_test.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate math models with plain-text prompting aligned to train_metamath_pt.py "
            "(no chat template, no default system prompt)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Base model path (HF model id or local path).",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional PEFT adapter path to merge before evaluation.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math"],
        required=True,
        help="Evaluation dataset.",
    )
    parser.add_argument("--start", type=int, default=0, help="Start index.")
    parser.add_argument("--end", type=int, default=MAX_INT, help="End index.")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate after slicing.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Number of prompts per vLLM generate() call.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override generation length. Defaults: GSM8K=1024, MATH=2048.",
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Optional plain-text prefix added before 'Question: ...'. Empty by default.",
    )
    parser.add_argument(
        "--use_training_system_prompt",
        action="store_true",
        help="Use the exact training SYSTEM_PROMPT from train_metamath_pt.py.",
    )
    parser.add_argument(
        "--eos_token",
        type=str,
        default=None,
        help="Optional EOS token override for tokenizer.",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="Fraction of GPU memory to use in vLLM.",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Optional explicit max sequence length for vLLM.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Model dtype for HF merge and vLLM load.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--result_filename",
        type=str,
        default=None,
        help="Optional explicit output JSON filename.",
    )

    return parser.parse_args()


def get_torch_dtype(dtype_str: str):
    if dtype_str == "auto":
        return None
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_str}")


def ensure_tokenizer(model_or_path: str, eos_token: Optional[str] = None):
    tokenizer = AutoTokenizer.from_pretrained(
        model_or_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if eos_token is not None:
        tokenizer.eos_token = eos_token

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_effective_system_prompt(args) -> Optional[str]:
    if args.system_prompt.strip():
        return args.system_prompt
    if args.use_training_system_prompt:
        return TRAINING_SYSTEM_PROMPT
    return None


def build_eval_prompt(question: str, system_prompt: Optional[str]) -> str:
    prefix = system_prompt if system_prompt else ""
    return f"{prefix}Question: {question.strip()}\nAnswer:"


def get_default_max_new_tokens(dataset_name: str) -> int:
    return 1024 if dataset_name == "gsm8k" else 2048


def parse_gsm8k_ground_truth(item):
    ans_str = item["answer"].split("#### ")[1]
    return int(ans_str.replace(",", ""))


def parse_math_ground_truth(item):
    solution = item["output"]
    return remove_boxed(last_boxed_only_string(solution))


def extract_math_prediction(completion: str):
    markers = [
        "This is the answer:",
        "The answer is:",
        "Answer:",
        "The answer is",
    ]
    for marker in markers:
        if marker in completion:
            ans = completion.split(marker)[-1].strip()
            if ans.endswith("."):
                ans = ans[:-1].strip()
            return ans

    boxed = last_boxed_only_string(completion)
    if boxed:
        return remove_boxed(boxed)

    return None


def extract_prediction(dataset_name: str, completion: str):
    if dataset_name == "gsm8k":
        return extract_answer_number(completion)

    if dataset_name == "math":
        return extract_math_prediction(completion)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def is_prediction_correct(dataset_name: str, y_pred, gt):
    if y_pred is None:
        return False

    if dataset_name == "gsm8k":
        try:
            return float(y_pred) == float(gt) or math_equal(y_pred, gt)
        except Exception:
            return math_equal(y_pred, gt)

    if dataset_name == "math":
        return math_equal(y_pred, gt)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_eval_data(
    args, system_prompt: Optional[str]
) -> Tuple[List[str], List[str], List]:
    data_file = DATA_PATHS[args.dataset]
    print(f"Loading dataset from: {data_file}")

    prompts = []
    raw_questions = []
    ground_truths = []

    with open(data_file, "r", encoding="utf-8") as f:
        for item in jsonlines.Reader(f):
            if args.dataset == "gsm8k":
                question = item.get("question", item.get("query", "")).strip()
                gt = parse_gsm8k_ground_truth(item)
            elif args.dataset == "math":
                question = item.get(
                    "instruction", item.get("query", item.get("question", ""))
                ).strip()
                gt = parse_math_ground_truth(item)
            else:
                raise ValueError(f"Unsupported dataset: {args.dataset}")

            prompt = build_eval_prompt(question, system_prompt)

            raw_questions.append(question)
            prompts.append(prompt)
            ground_truths.append(gt)

    prompts = prompts[args.start : args.end]
    raw_questions = raw_questions[args.start : args.end]
    ground_truths = ground_truths[args.start : args.end]

    if args.max_samples is not None:
        prompts = prompts[: args.max_samples]
        raw_questions = raw_questions[: args.max_samples]
        ground_truths = ground_truths[: args.max_samples]

    print(f"Total samples for evaluation: {len(prompts)}")
    return prompts, raw_questions, ground_truths


def save_tokenizer(source_tokenizer, out_dir: str):
    source_tokenizer.save_pretrained(out_dir)


def merge_adapter_to_local_dir(args, tokenizer):
    temp_dir = tempfile.TemporaryDirectory()
    merged_root = Path(temp_dir.name)
    print(f"Created temporary directory for merged model: {merged_root}")

    print("Loading base model in HF for adapter merge...")
    torch_dtype = get_torch_dtype(args.dtype)
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        **model_kwargs,
    )

    print(f"Loading adapter with PEFT from: {args.adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, args.adapter_path)

    print("Merging adapter into base model...")
    merged_model = peft_model.merge_and_unload()

    print(f"Saving merged model to: {merged_root}")
    merged_model.save_pretrained(str(merged_root), safe_serialization=True)
    save_tokenizer(tokenizer, str(merged_root))

    return str(merged_root), temp_dir


def build_vllm(args, model_path: str) -> LLM:
    llm_kwargs = {
        "model": model_path,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
    }

    if args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    return LLM(**llm_kwargs)


def build_sampling_params(tokenizer, max_new_tokens: int) -> SamplingParams:
    stop_token_ids = (
        [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )
    return SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        stop_token_ids=stop_token_ids,
        stop=["Question", "[Question"],
    )


def get_save_path(args) -> str:
    if args.result_filename:
        return args.result_filename

    filename = f"evaluation_results_{args.dataset}_vllm_plain.json"
    if args.adapter_path:
        return os.path.join(args.adapter_path, filename)
    return os.path.join("results", filename)


def evaluate(args):
    system_prompt = get_effective_system_prompt(args)
    tokenizer = ensure_tokenizer(args.model_path, eos_token=args.eos_token)

    vllm_model_path = args.model_path
    temp_dir = None

    try:
        if args.adapter_path:
            vllm_model_path, temp_dir = merge_adapter_to_local_dir(args, tokenizer)
            tokenizer = ensure_tokenizer(vllm_model_path, eos_token=args.eos_token)

        print(f"Loading vLLM model from: {vllm_model_path}")
        llm = build_vllm(args, vllm_model_path)

        prompts, raw_questions, ground_truths = load_eval_data(args, system_prompt)
        batched_prompts = batch_data(prompts, batch_size=args.batch_size)
        batched_questions = batch_data(raw_questions, batch_size=args.batch_size)
        batched_ground_truths = batch_data(ground_truths, batch_size=args.batch_size)

        results = []
        invalid_outputs = []
        prediction_records = []

        max_new_tokens = (
            args.max_new_tokens
            if args.max_new_tokens is not None
            else get_default_max_new_tokens(args.dataset)
        )
        sampling_params = build_sampling_params(tokenizer, max_new_tokens)

        print("Running plain-text batched inference with vLLM...")
        start_time = time.time()
        pbar = tqdm(total=len(batched_prompts), desc="Generating (Acc: 0.00%)")

        for batch_prompts, batch_questions, batch_gts in zip(
            batched_prompts, batched_questions, batched_ground_truths
        ):
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            decoded = [out.outputs[0].text if out.outputs else "" for out in outputs]

            for prompt, question, completion, gt in zip(
                batch_prompts, batch_questions, decoded, batch_gts
            ):
                y_pred = extract_prediction(args.dataset, completion)
                is_correct = is_prediction_correct(args.dataset, y_pred, gt)
                results.append(is_correct)

                record = {
                    "question": question,
                    "prompt": prompt,
                    "completion": completion,
                    "prediction": y_pred,
                    "ground_truth": gt,
                    "correct": bool(is_correct),
                }
                prediction_records.append(record)

                if y_pred is None:
                    invalid_outputs.append(
                        {
                            "question": question,
                            "prompt": prompt,
                            "output": completion,
                            "answer": gt,
                        }
                    )

                if args.max_samples is not None and args.max_samples <= 10:
                    print("QUESTION:", question)
                    print("PROMPT:", prompt)
                    print("COMPLETION:", completion)
                    print("Y_PRED:", y_pred)
                    print("GT:", gt)
                    print("CORRECT:", is_correct)
                    print()

            current_acc = sum(results) / len(results) if results else 0.0
            pbar.set_description(f"Generating (Acc: {current_acc * 100:.2f}%)")
            pbar.update(1)

        pbar.close()

        end_time = time.time()
        processing_time = end_time - start_time
        acc = sum(results) / len(results) if results else 0.0

        print("\n======================================")
        print(f"Model: {args.model_path}")
        print(f"Adapter: {args.adapter_path}")
        print(f"Dataset: {args.dataset}")
        print(f"Total Evaluated: {len(results)}")
        print(f"Invalid Outputs: {len(invalid_outputs)}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print(f"Processing Time: {processing_time:.2f} seconds")
        print("======================================")

        results_data = {
            "backend": "vllm",
            "prompt_format": "plain_text_question_answer",
            "dataset": args.dataset,
            "accuracy": acc,
            "total_evaluated": len(results),
            "invalid_outputs_count": len(invalid_outputs),
            "invalid_outputs": invalid_outputs,
            "predictions": prediction_records,
            "adapter_path": args.adapter_path,
            "model_path": args.model_path,
            "effective_model_path": vllm_model_path,
            "system_prompt": system_prompt,
            "used_training_system_prompt": bool(args.use_training_system_prompt),
            "start_index": args.start,
            "end_index": args.end,
            "max_samples": args.max_samples,
            "batch_size": args.batch_size,
            "max_new_tokens": max_new_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_model_len": args.max_model_len,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
        }

        save_path = get_save_path(args)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {save_path}")

    finally:
        if temp_dir is not None:
            print(f"Cleaning up temporary directory: {temp_dir.name}")
            temp_dir.cleanup()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
