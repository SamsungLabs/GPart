import argparse
import json
import logging
import os
import random

import accelerate
import evaluate
import numpy as np
import torch
import transformers
from datasets import ClassLabel, load_dataset
from torch.optim import AdamW
from torch.utils.data import Dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

import peft
from peft import GPartConfig, LoraConfig, get_peft_model
from peft.tuners.unilora import UniLoRAConfig

# Configure logging to display INFO-level messages from GPart grouping module
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

parser = argparse.ArgumentParser()
parser.add_argument("--head_lr", type=float, default=3e-3)
parser.add_argument("--base_lr", type=float, default=4e-3)
parser.add_argument("--output_prefix", type=str, default="")
parser.add_argument("--rank", type=int, default=4)
parser.add_argument("--num_train_epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--dataset",
    type=str,
    default="fgvc",
    choices=[
        "oxfordpets",
        "fgvc",
        "standfordcars",
        "cifar10",
        "cifar100",
        "dtd",
        "flowers102",
        "eurosat",
        "resisc45",
    ],
)
parser.add_argument(
    "--model_size",
    type=str,
    default="base",
    choices=["base", "large"],
)
parser.add_argument(
    "--adapter_type",
    type=str,
    default="gpart",
    choices=["gpart", "unilora", "lora"],
)
parser.add_argument(
    "--d",
    type=int,
    default=None,
    help="Budget parameter: d for GPart, theta_d_length for UniLoRA. "
    "Defaults to 72000/144000 (base/large).",
)
parser.add_argument(
    "--init_bound",
    type=float,
    default=0.0,
    help="Initialization bound for theta_d (GPart: init_bound, UniLoRA: init_theta_d_bound).",
)
parser.add_argument(
    "--dropout",
    type=float,
    default=0.0,
    help="Dropout probability (GPart: gpart_dropout, UniLoRA: unilora_dropout).",
)
parser.add_argument(
    "--lora_alpha",
    type=int,
    default=8,
    help="LoRA alpha (scaling factor). Only used with --adapter_type lora.",
)
parser.add_argument(
    "--grouping_strategy",
    type=str,
    default="random",
    choices=["random", "signed_magnitude"],
)

args_custom = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(args_custom.seed)

if args_custom.model_size == "base":
    model_checkpoint = "google/vit-base-patch16-224-in21k"
else:
    model_checkpoint = "google/vit-large-patch16-224-in21k"
dataset_name = args_custom.dataset
print(f"📦 Loading dataset: {dataset_name}")

is_torchvision_dataset = False
label_column = "labels"

if dataset_name == "cifar10":
    train_val_ds = load_dataset("cifar10", split="train")
    train_valid_split = train_val_ds.train_test_split(
        test_size=0.1, seed=args_custom.seed
    )
    train_ds = train_valid_split["train"]
    val_ds = train_valid_split["test"]
    test_ds = load_dataset("cifar10", split="test")

elif dataset_name == "cifar100":
    train_val_ds = load_dataset("cifar100", split="train")
    train_valid_split = train_val_ds.train_test_split(
        test_size=0.1, seed=args_custom.seed
    )
    train_ds = train_valid_split["train"]
    val_ds = train_valid_split["test"]
    test_ds = load_dataset("cifar100", split="test")

elif dataset_name == "flowers102":
    train_ds = load_dataset("oxford_flowers102", split="train")
    val_ds = load_dataset("oxford_flowers102", split="validation")
    test_ds = load_dataset("oxford_flowers102", split="test")

elif dataset_name == "resisc45":
    train_ds = load_dataset("timm/resisc45", split="train")
    val_ds = load_dataset("timm/resisc45", split="validation")
    test_ds = load_dataset("timm/resisc45", split="test")

elif dataset_name == "oxfordpets":
    train_val_ds = load_dataset("timm/oxford-iiit-pet", split="train")
    train_valid_split = train_val_ds.train_test_split(
        test_size=0.1, seed=args_custom.seed
    )
    train_ds = train_valid_split["train"]
    val_ds = train_valid_split["test"]
    test_ds = load_dataset("timm/oxford-iiit-pet", split="test")

elif dataset_name == "standfordcars":
    train_val_ds = load_dataset("tanganke/stanford_cars", split="train")
    train_valid_split = train_val_ds.train_test_split(
        test_size=0.1, seed=args_custom.seed
    )
    train_ds = train_valid_split["train"]
    val_ds = train_valid_split["test"]
    test_ds = load_dataset("tanganke/stanford_cars", split="test")

elif dataset_name == "fgvc":
    from torchvision.datasets import FGVCAircraft

    train_ds = FGVCAircraft(
        root="data",
        split="train",
        annotation_level="variant",
        download=True,
    )
    val_ds = FGVCAircraft(
        root="data",
        split="val",
        annotation_level="variant",
        download=True,
    )
    test_ds = FGVCAircraft(
        root="data",
        split="test",
        annotation_level="variant",
        download=True,
    )
    is_torchvision_dataset = True

elif dataset_name == "dtd":
    ds = load_dataset("imagefolder", data_dir="data/dtd/images", split="train")
    ds = ds.shuffle(seed=args_custom.seed)
    train_val_test = ds.train_test_split(test_size=0.28, seed=args_custom.seed)
    val_test = train_val_test["test"].train_test_split(
        test_size=0.715, seed=args_custom.seed
    )

    train_ds = train_val_test["train"]
    val_ds = val_test["train"]
    test_ds = val_test["test"]

    print(len(train_ds), len(val_ds), len(test_ds))

elif dataset_name == "eurosat":
    train_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
    val_ds = load_dataset("blanchon/EuroSAT_RGB", split="validation")
    test_ds = load_dataset("blanchon/EuroSAT_RGB", split="test")

else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")


if is_torchvision_dataset:
    labels = train_ds.classes
    label2id = train_ds.class_to_idx
    id2label = {i: label for label, i in label2id.items()}
else:
    if "label" in train_ds.features:
        hf_label_column = "label"
    elif "fine_label" in train_ds.features:
        hf_label_column = "fine_label"
    else:
        raise ValueError(
            f"Cannot find label column in dataset: {train_ds.features.keys()}"
        )

    if isinstance(train_ds.features[hf_label_column], ClassLabel):
        labels = train_ds.features[hf_label_column].names
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for i, label in enumerate(labels)}
    else:
        unique_labels = sorted(set(train_ds[hf_label_column]))
        label2id = {str(i): i for i in unique_labels}
        id2label = {i: str(i) for i in unique_labels}

    label_column = hf_label_column

print(id2label[2])

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
print(image_processor)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)

test_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


if not is_torchvision_dataset:
    if "image" in train_ds.features:
        image_column = "image"
    elif "img" in train_ds.features:
        image_column = "img"
    else:
        raise ValueError(
            f"Can't find image column in dataset: {train_ds.features.keys()}"
        )

    def preprocess_train(example_batch):
        example_batch["pixel_values"] = [
            train_transforms(image.convert("RGB"))
            for image in example_batch[image_column]
        ]
        example_batch["labels"] = example_batch[label_column]
        return example_batch

    def preprocess_val(example_batch):
        example_batch["pixel_values"] = [
            val_transforms(image.convert("RGB"))
            for image in example_batch[image_column]
        ]
        example_batch["labels"] = example_batch[label_column]
        return example_batch

    def preprocess_test(example_batch):
        example_batch["pixel_values"] = [
            test_transforms(image.convert("RGB"))
            for image in example_batch[image_column]
        ]
        example_batch["labels"] = example_batch[label_column]
        return example_batch

    train_ds.set_transform(preprocess_train)
    val_ds.set_transform(preprocess_val)
    test_ds.set_transform(preprocess_test)

else:

    class TorchvisionFGVCDataset(Dataset):
        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform

        def __len__(self):
            return len(self.base_dataset)

        def __getitem__(self, idx):
            image, label = self.base_dataset[idx]
            image = image.convert("RGB")
            pixel_values = self.transform(image)
            return {
                "pixel_values": pixel_values,
                "labels": label,
            }

    train_ds = TorchvisionFGVCDataset(train_ds, train_transforms)
    val_ds = TorchvisionFGVCDataset(val_ds, val_transforms)
    test_ds = TorchvisionFGVCDataset(test_ds, test_transforms)


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
print_trainable_parameters(model)


def build_adapter_config(args):
    """Build the adapter config based on the chosen adapter type."""
    # Resolve default d if not provided
    if args.d is not None:
        d = args.d
    else:
        d = 72000 if args.model_size == "base" else 144000

    if args.adapter_type == "gpart":
        return GPartConfig(
            d=d,
            target_modules=["query", "value"],
            gpart_dropout=args.dropout,
            bias="none",
            inference_mode=False,
            grouping_strategy=args.grouping_strategy,
            init_bound=args.init_bound,
        )

    if args.adapter_type == "unilora":
        return UniLoRAConfig(
            r=args.rank,
            theta_d_length=d,
            target_modules=["query", "value"],
            unilora_dropout=args.dropout,
            bias="none",
            inference_mode=False,
            init_theta_d_bound=args.init_bound,
        )

    if args.adapter_type == "lora":
        return LoraConfig(
            r=args.rank,
            lora_alpha=args.lora_alpha,
            target_modules=["query", "value"],
            lora_dropout=args.dropout,
            bias="none",
            inference_mode=False,
        )

    raise ValueError(f"Unsupported adapter type: {args.adapter_type}")


config = build_adapter_config(args_custom)

lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

model_name = model_checkpoint.split("/")[-1]
batch_size = 64 if args_custom.model_size == "base" else 32

# Build output dir with adapter-specific info
adapter_info = f"{args_custom.adapter_type}"
if args_custom.adapter_type == "gpart":
    adapter_info += (
        f"-d{args_custom.d or (72000 if args_custom.model_size == 'base' else 144000)}"
    )
    if args_custom.grouping_strategy != "random":
        adapter_info += f"-grouping_{args_custom.grouping_strategy}"
elif args_custom.adapter_type == "unilora":
    adapter_info += f"-r{args_custom.rank}-d{args_custom.d or (72000 if args_custom.model_size == 'base' else 144000)}"
elif args_custom.adapter_type == "lora":
    adapter_info += f"-r{args_custom.rank}-alpha{args_custom.lora_alpha}"

# Hierarchical output dir: vit-{model_size}/{adapter_info}-epoch{N}/{dataset}/seed_{seed}
adapter_config_dir = f"{adapter_info}-epoch{args_custom.num_train_epochs}"
output_dir = (
    f"experiments/outputs/vit-{args_custom.model_size}"
    f"/{adapter_config_dir}"
    f"/{args_custom.dataset}"
    f"/seed_{args_custom.seed}"
)
os.makedirs(output_dir, exist_ok=True)


class CustomTrainer(Trainer):
    def create_optimizer(self):
        head_lr = args_custom.head_lr
        base_lr = args_custom.base_lr
        weight_decay = 0.01

        classifier_params = []
        base_params = []

        for name, param in self.model.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
                classifier_params.append(param)
                continue

            if not param.requires_grad:
                continue

            base_params.append(param)

        optimizer_grouped_parameters = [
            {"params": base_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": classifier_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        self.lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.4 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler


args = TrainingArguments(
    output_dir=output_dir,
    remove_unused_columns=False,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    num_train_epochs=args_custom.num_train_epochs,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
    report_to="none",
    seed=args_custom.seed,
)

metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["labels"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


trainer = CustomTrainer(
    model=lora_model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    processing_class=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)

train_results = trainer.train()

# Save the best model directly in the run directory (not in a checkpoint subdirectory)
lora_model.save_pretrained(output_dir)
image_processor.save_pretrained(output_dir)
print(f"Best model saved to: {output_dir}")

# Evaluate on test set and save results
test_results = trainer.evaluate(test_ds)
print("test results")
print(test_results)

# Save results.json in the run directory
results_path = os.path.join(output_dir, "results.json")
with open(results_path, "w") as f:
    json.dump(test_results, f, indent=2)
print(f"Results saved to: {results_path}")
