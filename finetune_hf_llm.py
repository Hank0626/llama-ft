import argparse
import copy
import os
import time

import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

OPTIM_BETAS = (0.9, 0.999)
OPTIM_EPS = 1e-8
OPTIM_WEIGHT_DECAY = 0.0


def get_number_of_params(model: nn.Module):
    state_dict = model.state_dict()
    return sum(p.numel() for p in state_dict.values())


def get_preprocessed_function_call(data_path, tokenizer, split):
    dataset = datasets.load_dataset(path=data_path, split="train")
    if split == "test":
        dataset = dataset.train_test_split(test_size=0.1)[split]

    def convert(sample):
        max_words = 512
        full = tokenizer.encode(sample["input"] + sample["output"])
        prompt = torch.tensor(tokenizer.encode(sample["input"]), dtype=torch.int64)
        full = torch.tensor(full + [tokenizer.eos_token_id], dtype=torch.int64)
        padding = max_words - full.shape[0]
        if padding > 0:
            full = torch.cat((full, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            full = full[:max_words]
        labels = copy.deepcopy(full)
        labels[: len(prompt)] = -1
        full_mask = full.ge(0)
        label_mask = labels.ge(0)
        full[~full_mask] = 0
        labels[~label_mask] = 0
        full_mask = full_mask.float()
        label_mask = label_mask.float()
        return {
            "input_ids": full,
            "labels": labels,
            "attention_mask": full_mask,
        }

    dataset = dataset.map(
        convert,
        remove_columns=list(dataset.features),
    )
    return dataset


def get_tokenizer(pretrained_path):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, legacy=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def training_function(kwargs: dict):
    print("training_function called")

    config = kwargs["config"]
    args = argparse.Namespace(**kwargs["args"])
    model_path = config["model_path"]
    data_path = config["data_path"]

    lr = config["lr"]
    num_epochs = int(config["num_epochs"])

    gradient_accumulation_steps = int(config["gradient_accumulation_steps"])

    tokenizer = get_tokenizer(pretrained_path=model_path)

    train_ds = get_preprocessed_function_call(
        data_path=data_path, tokenizer=tokenizer, split="train"
    )
    valid_ds = get_preprocessed_function_call(
        data_path=data_path, tokenizer=tokenizer, split="test"
    )

    train_ds_len = len(train_ds)

    print(f"Loading model from {model_path} ...")
    s = time.time()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # `use_cache=True` is incompatible with gradient checkpointing.
        use_cache=False,
    ).to("cuda")

    print(f"Done loading model in {time.time() - s} seconds.")
    model.resize_token_embeddings(len(tokenizer))
    print(f"Size of model: {get_number_of_params(model) / 1e9:.2f}b")
    print("Model initialized with pretrained weights. Training starting...")
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()

    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.parameters(),
        lr=lr,
        betas=OPTIM_BETAS,
        weight_decay=OPTIM_WEIGHT_DECAY,
        eps=OPTIM_EPS,
    )

    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=(train_ds_len * num_epochs) // gradient_accumulation_steps,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=args.batch_size_per_device,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=OPTIM_WEIGHT_DECAY,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_steps=1,
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        optimizers=(optimizer, lr_scheduler),
    )

    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mx",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )

    parser.add_argument(
        "--batch-size-per-device",
        "-bs",
        type=int,
        default=16,
        help="Batch size to use per device.",
    )

    parser.add_argument(
        "--eval-batch-size-per-device",
        type=int,
        default=64,
        help="Batch size to use per device (For evaluation).",
    )

    parser.add_argument(
        "--num-devices", "-nd", type=int, default=4, help="Number of devices to use."
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps."
    )
    parser.add_argument(
        "--data_path", type=str, help="Path to fine-tuning dataset path."
    )

    parser.add_argument(
        "--special_token_path", type=str, help="Path to token json file"
    )
    parser.add_argument(
        "--no-grad-ckpt",
        action="store_true",
        help="If passed, will not use gradient checkpointing.",
    )
    parser.add_argument("--output_dir", type=str, help="Path to output directory.")
    parser.add_argument(
        "--model_path", default="meta-llama/Llama-2-7b-chat-hf", type=str
    )
    parser.add_argument(
        "--num-epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--num-checkpoints-to-keep",
        type=int,
        help=(
            "Number of checkpoints to keep, if None, all checkpoints will be kept, "
            "if set to n>=1, the top n checkpoint with min. evaluation perplexity "
            "will be kept."
        ),
    )
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate to use.")

    parser.add_argument(
        "--ctx_len",
        type=int,
        default=512,
        help="Maximum context length for the model input sequences.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local rank passed from distributed launcher",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not args.output_dir:
        raise ValueError("--output_dir must be specified")

    # update the config with args so that we have access to them.
    config = vars(args)
    config.update(
        **{
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "seed": 42,
            "batch_size": args.batch_size_per_device,
            "gradient_accumulation_steps": args.grad_accum,
            "model_path": args.model_path,
            "block_size": args.ctx_len,
            "eval_batch_size": args.eval_batch_size_per_device,
            "data_path": args.data_path,
        }
    )

    os.environ["TUNE_RESULT_DIR"] = args.output_dir

    training_function({"config": config, "args": vars(args)})


if __name__ == "__main__":
    main()
