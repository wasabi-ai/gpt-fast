import torch
from datasets import load_dataset
from pathlib import Path
from tokenizer import get_tokenizer
from transformers import TrainingArguments, Trainer

import numpy as np
import evaluate
import lightning as L
import torch, torch.nn as nn, torch.utils.data as data, torch.nn.functional as F

import generate

torch.set_float32_matmul_precision("medium")

MAX_SEQ_LEN = 1024
MODEL_CHECKPOINT = Path("checkpoints/meta-llama/Meta-Llama-3-8B-Instruct/model.pth")
TOKENIZER_CHECKPOINT = MODEL_CHECKPOINT.parent / "tokenizer.model"

dataset = load_dataset("yelp_review_full")
tokenizer = get_tokenizer(
    tokenizer_model_path=TOKENIZER_CHECKPOINT, model_name="Meta-Llama-3-8B-Instruct"
)


def tokenize_function(examples):
    bos_id = tokenizer.bos_id()
    eos_id = tokenizer.eos_id()
    tokens = []
    for row in examples["text"]:
        tokens.append([bos_id] + tokenizer.encode(row))
        if len(tokens) > MAX_SEQ_LEN:
            tokens = tokens[:MAX_SEQ_LEN]

    for row in tokens:
        row.extend([eos_id] * (MAX_SEQ_LEN - len(row)))

    tokens = [torch.tensor(t, dtype=torch.int32) for t in tokens]
    return {"text": tokens}


tokenized_datasets = dataset.map(
    tokenize_function, batched=True, num_proc=32, batch_size=1
)
tokenized_datasets = tokenized_datasets.with_format("torch")
small_train_dataset = [
    row["text"]
    for row in tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
]
small_eval_dataset = [
    row["text"]
    for row in tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
]


model = generate._load_model(
    checkpoint_path=MODEL_CHECKPOINT,
    device="cuda",
    precision=torch.bfloat16,
    use_tp=False,
)

model.setup_caches(max_batch_size=1, max_seq_length=MAX_SEQ_LEN)

training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")


class LlamaTrainer(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        print(batch.shape, batch.dtype)
        x = batch
        y_hat = self.model(x)
        y = x.view(x.size(0), -1)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


llama_model = LlamaTrainer()
trainer = L.Trainer()
trainer.fit(
    llama_model,
    data.DataLoader(small_train_dataset, batch_size=1),
    data.DataLoader(small_eval_dataset, batch_size=1),
)
