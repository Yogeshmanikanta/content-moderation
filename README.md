# 🛡️ Real-Time Content Moderation System

Detects toxic and policy-violating content in real time using
fine-tuned DistilBERT with ONNX optimization.

## Results
- 90% precision on toxic content detection
- 93% overall accuracy
- 55ms average inference latency (ONNX optimized)
- 1.56x speedup over vanilla PyTorch

## Tech Stack
- Model: DistilBERT-base-uncased (HuggingFace Transformers)
- Optimization: ONNX Runtime
- API: FastAPI + Uvicorn
- Dashboard: Streamlit + Plotly
- Container: Docker

## Project Structure
content-moderation/
├── data/               # Data preparation scripts
├── models/             # Trained model + ONNX export
├── api/                # FastAPI REST endpoint
├── dashboard/          # Streamlit monitoring dashboard
├── notebooks/          # Training script
└── Dockerfile

## Setup
pip install -r requirements.txt

## Run API
uvicorn api.main:app --host 0.0.0.0 --port 8000

## Run Dashboard
streamlit run dashboard/app.py

## API Endpoints
| Endpoint          | Method | Description          |
|-------------------|--------|----------------------|
| /health           | GET    | API health check     |
| /moderate         | POST   | Single text check    |
| /moderate/batch   | POST   | Batch text check     |
| /stats            | GET    | Model statistics     |

## Dataset
Jigsaw Toxic Comment Classification (Kaggle)
- 200K labeled social media posts
- Binary classification: clean vs toxic
- Balanced via undersampling
