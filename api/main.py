from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import onnxruntime as ort
from transformers import DistilBertTokenizer
import numpy as np
import torch
import time
import uvicorn
from datetime import datetime

# ── 1. App setup ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Real-Time Content Moderation API",
    description="Detects toxic content using fine-tuned DistilBERT + ONNX",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── 2. Load model on startup ───────────────────────────────────────────────
MODEL_PATH = "models/distilbert"
ONNX_PATH  = "models/onnx/model.onnx"
MAX_LEN    = 64
LABELS     = ["clean", "toxic"]

print("Loading tokenizer and ONNX model...")

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4

session = ort.InferenceSession(
    ONNX_PATH,
    sess_options=sess_options,
    providers=["CPUExecutionProvider"]
)

print("✅ Model loaded and ready!")

# ── 3. Request / Response schemas ─────────────────────────────────────────
class ModerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=512,
                      example="You are so stupid!")

class BatchModerationRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=32,
                             example=["Hello!", "You are awful!"])

class ModerationResult(BaseModel):
    text:        str
    label:       str
    confidence:  float
    is_toxic:    bool
    latency_ms:  float
    flagged_at:  str

class ModerationResponse(BaseModel):
    results:        List[ModerationResult]
    total_texts:    int
    toxic_count:    int
    clean_count:    int
    avg_latency_ms: float

# ── 4. Core prediction function ────────────────────────────────────────────
def predict(text: str):
    start = time.perf_counter()

    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    ort_inputs = {
        "input_ids":      enc["input_ids"].numpy(),
        "attention_mask": enc["attention_mask"].numpy()
    }

    logits     = session.run(None, ort_inputs)[0]
    probs      = torch.softmax(torch.tensor(logits), dim=1).numpy()[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(np.max(probs))
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "text":       text,
        "label":      LABELS[pred_idx],
        "confidence": round(confidence * 100, 2),
        "is_toxic":   pred_idx == 1,
        "latency_ms": round(latency_ms, 2),
        "flagged_at": datetime.utcnow().isoformat()
    }

# ── 5. Routes ──────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Content Moderation API is running!",
        "docs":    "Visit /docs for interactive API documentation"
    }

@app.get("/health")
def health():
    return {
        "status":  "healthy",
        "model":   "DistilBERT + ONNX",
        "version": "1.0.0"
    }

@app.post("/moderate", response_model=ModerationResponse)
def moderate_text(request: ModerationRequest):
    try:
        result  = predict(request.text)
        results = [ModerationResult(**result)]
        return ModerationResponse(
            results=results,
            total_texts=1,
            toxic_count=1 if result["is_toxic"] else 0,
            clean_count=0 if result["is_toxic"] else 1,
            avg_latency_ms=result["latency_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/moderate/batch", response_model=ModerationResponse)
def moderate_batch(request: BatchModerationRequest):
    try:
        results    = [predict(text) for text in request.texts]
        toxic      = sum(1 for r in results if r["is_toxic"])
        avg_lat    = round(np.mean([r["latency_ms"] for r in results]), 2)
        return ModerationResponse(
            results=[ModerationResult(**r) for r in results],
            total_texts=len(results),
            toxic_count=toxic,
            clean_count=len(results) - toxic,
            avg_latency_ms=avg_lat
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats():
    return {
        "model_name":    "DistilBERT-base-uncased (fine-tuned)",
        "optimization":  "ONNX Runtime",
        "avg_latency":   "~27ms",
        "precision":     "90%",
        "accuracy":      "93%",
        "max_length":    MAX_LEN,
        "labels":        LABELS
    }

# ── 6. Run ─────────────────────────────────────────────────────────────────
# Replace with this
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port)