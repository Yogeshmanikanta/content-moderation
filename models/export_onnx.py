import torch
import time
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import onnx
import onnxruntime as ort
from pathlib import Path
import os

print("="*55)
print("Phase 4 — ONNX Export & Optimization")
print("="*55)

# ── 1. Load your trained model ─────────────────────────────────────────────
MODEL_PATH = "models/distilbert"
tokenizer  = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model      = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print(f"\n✅ Loaded trained model from {MODEL_PATH}")

# ── 2. Create dummy input for export ───────────────────────────────────────
dummy_text = "This is a sample comment for export"
dummy_enc  = tokenizer(
    dummy_text,
    max_length=64,
    padding="max_length",
    truncation=True,
    return_tensors="pt"
)

dummy_input_ids      = dummy_enc["input_ids"]
dummy_attention_mask = dummy_enc["attention_mask"]

# ── 3. Export to ONNX ──────────────────────────────────────────────────────
os.makedirs("models/onnx", exist_ok=True)
ONNX_PATH = "models/onnx/model.onnx"

print(f"\nExporting to ONNX...")
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),
    ONNX_PATH,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids":      {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits":         {0: "batch_size"}
    },
    opset_version=13,
    do_constant_folding=True
)
print(f"✅ ONNX model saved to {ONNX_PATH}")

# ── 4. Validate ONNX model ─────────────────────────────────────────────────
onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model validation passed")

# ── 5. Create ONNX Runtime session ────────────────────────────────────────
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

session = ort.InferenceSession(
    ONNX_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)
print("✅ ONNX Runtime session created")

# ── 6. Benchmark — PyTorch vs ONNX latency ────────────────────────────────
test_texts = [
    "You are so stupid and pathetic!",
    "Have a wonderful day, hope you feel better",
    "I will find you and hurt you badly",
    "Thank you for your help, really appreciate it",
    "Go kill yourself nobody likes you",
]

NUM_RUNS = 50
print(f"\n{'='*55}")
print(f"Latency Benchmark ({NUM_RUNS} runs each)")
print(f"{'='*55}")

# PyTorch latency
pytorch_times = []
with torch.no_grad():
    for text in test_texts:
        enc = tokenizer(text, max_length=64, padding="max_length",
                        truncation=True, return_tensors="pt")
        for _ in range(NUM_RUNS):
            start = time.perf_counter()
            out   = model(input_ids=enc["input_ids"],
                          attention_mask=enc["attention_mask"])
            end   = time.perf_counter()
            pytorch_times.append((end - start) * 1000)

avg_pytorch = np.mean(pytorch_times)
print(f"PyTorch average latency : {avg_pytorch:.2f} ms")

# ONNX latency
onnx_times = []
for text in test_texts:
    enc = tokenizer(text, max_length=64, padding="max_length",
                    truncation=True, return_tensors="pt")
    ort_inputs = {
        "input_ids":      enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy()
    }
    for _ in range(NUM_RUNS):
        start  = time.perf_counter()
        output = session.run(None, ort_inputs)
        end    = time.perf_counter()
        onnx_times.append((end - start) * 1000)

avg_onnx = np.mean(onnx_times)
print(f"ONNX average latency    : {avg_onnx:.2f} ms")
print(f"Speedup                 : {avg_pytorch/avg_onnx:.2f}x faster")

if avg_onnx < 50:
    print(f"\n✅ Target achieved! ONNX inference is under 50ms")
else:
    print(f"\n⚠️  Above 50ms on CPU — acceptable, GPU would be <10ms")

# ── 7. Test predictions ────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("Sample Predictions")
print(f"{'='*55}")

labels = ["CLEAN", "TOXIC"]

for text in test_texts:
    enc = tokenizer(text, max_length=64, padding="max_length",
                    truncation=True, return_tensors="pt")
    ort_inputs = {
        "input_ids":      enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy()
    }
    logits     = session.run(None, ort_inputs)[0]
    probs      = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
    pred_label = labels[np.argmax(probs)]
    confidence = np.max(probs) * 100

    print(f"\nText      : {text[:55]}...")
    print(f"Prediction: {pred_label} ({confidence:.1f}% confidence)")

# ── 8. Save session config for API use ────────────────────────────────────
print(f"\n{'='*55}")
print("Model sizes")
print(f"{'='*55}")

pytorch_size = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
) / 1024 / 1024

onnx_size = Path(ONNX_PATH).stat().st_size / 1024 / 1024
print(f"PyTorch model : {pytorch_size:.1f} MB")
print(f"ONNX model    : {onnx_size:.1f} MB")

print("\n✅ Phase 4 complete! ONNX model ready for API.")