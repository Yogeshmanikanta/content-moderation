import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.metrics import classification_report, confusion_matrix
import os
import time

# ── 0. Device setup ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 1. Load data ───────────────────────────────────────────────────────────
train_df = pd.read_csv("data/train_clean.csv")
val_df   = pd.read_csv("data/val_clean.csv")
test_df  = pd.read_csv("data/test_clean.csv")

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Use 20% of data for faster training on CPU
train_df = train_df.sample(frac=0.2, random_state=42)
val_df   = val_df.sample(frac=0.2, random_state=42)
test_df  = test_df.sample(frac=0.2, random_state=42)
print(f"Reduced — Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── 2. Tokenizer ───────────────────────────────────────────────────────────
MODEL_NAME  = "distilbert-base-uncased"
MAX_LEN     = 64
BATCH_SIZE  = 16
EPOCHS      = 2
LR          = 2e-5

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# ── 3. Dataset class ───────────────────────────────────────────────────────
class ToxicDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.texts  = df["comment_text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = ToxicDataset(train_df, tokenizer, MAX_LEN)
val_dataset   = ToxicDataset(val_df,   tokenizer, MAX_LEN)
test_dataset  = ToxicDataset(test_df,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

# ── 4. Model ───────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model = model.to(device)
print(f"\nModel loaded: {MODEL_NAME}")

# ── 5. Optimizer & Scheduler ───────────────────────────────────────────────
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# ── 6. Training loop ───────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(f"  Step {batch_idx+1}/{len(loader)} "
                  f"| Loss: {total_loss/(batch_idx+1):.4f} "
                  f"| Acc: {correct/total*100:.2f}%")

    return total_loss / len(loader), correct / total

# ── 7. Evaluation ──────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            total_loss += outputs.loss.item()
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    report   = classification_report(all_labels, all_preds,
                                     target_names=["clean", "toxic"])
    return avg_loss, report

# ── 8. Run training ────────────────────────────────────────────────────────
print("\n" + "="*55)
print("Starting training...")
print("="*55)

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print("-"*40)

    start = time.time()
    train_loss, train_acc = train_epoch(model, train_loader,
                                        optimizer, scheduler, device)
    elapsed = time.time() - start

    val_loss, val_report = evaluate(model, val_loader, device)

    print(f"\nEpoch {epoch} summary:")
    print(f"  Train Loss : {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
    print(f"  Val Loss   : {val_loss:.4f}")
    print(f"  Time       : {elapsed/60:.1f} min")
    print(f"\nValidation Report:\n{val_report}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        os.makedirs("models/distilbert", exist_ok=True)
        model.save_pretrained("models/distilbert")
        tokenizer.save_pretrained("models/distilbert")
        print(f"  ✅ Best model saved to models/distilbert/")

# ── 9. Final test evaluation ───────────────────────────────────────────────
print("\n" + "="*55)
print("Final Test Evaluation")
print("="*55)

test_loss, test_report = evaluate(model, test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"\nTest Report:\n{test_report}")
print("\n✅ Training complete!")