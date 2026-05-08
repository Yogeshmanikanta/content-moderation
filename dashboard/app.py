import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.express as px

# ── 1. Page config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Content Moderation Dashboard",
    page_icon="🛡️",
    layout="wide"
)

API_URL = "http://localhost:8000"

# ── 2. Session state ───────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "prefill" not in st.session_state:
    st.session_state.prefill = ""

# ── 3. Helper functions ────────────────────────────────────────────────────
def moderate_text(text):
    try:
        res = requests.post(
            f"{API_URL}/moderate",
            json={"text": text},
            timeout=10
        )
        return res.json()
    except Exception as e:
        return None

def moderate_batch(texts):
    try:
        res = requests.post(
            f"{API_URL}/moderate/batch",
            json={"texts": texts},
            timeout=30
        )
        return res.json()
    except Exception as e:
        return None

def check_api():
    try:
        res = requests.get(f"{API_URL}/health", timeout=5)
        return res.status_code == 200
    except:
        return False

def add_to_history(r):
    st.session_state.history.append({
        "text":       r["text"][:60] + "..." if len(r["text"]) > 60 else r["text"],
        "label":      r["label"],
        "confidence": r["confidence"],
        "is_toxic":   r["is_toxic"],
        "latency_ms": r["latency_ms"],
        "time":       datetime.now().strftime("%H:%M:%S")
    })

# ── 4. Header ──────────────────────────────────────────────────────────────
st.title("🛡️ Real-Time Content Moderation Dashboard")
st.markdown("Powered by **DistilBERT + ONNX** — Fine-tuned on 200K social media posts")
st.divider()

# ── 5. API status ──────────────────────────────────────────────────────────
api_ok = check_api()
if api_ok:
    st.success("API is online and ready", icon="✅")
else:
    st.error("❌ API is offline — make sure uvicorn is running on port 8000")
    st.stop()

# ── 6. Top metrics (recalculated every rerun) ──────────────────────────────
total   = len(st.session_state.history)
toxic   = sum(1 for r in st.session_state.history if r["is_toxic"])
clean   = total - toxic
avg_lat = round(
    sum(r["latency_ms"] for r in st.session_state.history) / total, 1
) if total > 0 else 0.0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Checked",    total)
col2.metric("Toxic Flagged",    toxic,
            delta=f"{toxic/total*100:.1f}%" if total else "0%")
col3.metric("Clean Passed",     clean)
col4.metric("Avg Latency (ms)", avg_lat)

st.divider()

# ── 7. Two column layout ───────────────────────────────────────────────────
left, right = st.columns([1.2, 0.8])

with left:
    st.subheader("🔍 Single Text Check")

    # FIX 1 — prefill text area from sample button
    user_input = st.text_area(
        "Enter text to moderate:",
        value=st.session_state.prefill,   # ← this was missing before
        placeholder="Type any comment here...",
        height=120,
        key="text_input"
    )

    if st.button("Moderate", type="primary", use_container_width=True):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                result = moderate_text(user_input.strip())

            if result and result.get("results"):
                r = result["results"][0]

                if r["is_toxic"]:
                    st.error(f"🚨 TOXIC — {r['confidence']}% confidence")
                else:
                    st.success(f"✅ CLEAN — {r['confidence']}% confidence")

                c1, c2 = st.columns(2)
                c1.metric("Label",   r["label"].upper())
                c2.metric("Latency", f"{r['latency_ms']} ms")

                st.progress(
                    r["confidence"] / 100,
                    text=f"Confidence: {r['confidence']}%"
                )

                add_to_history(r)

                # FIX 2 — clear prefill and rerun so metrics update
                st.session_state.prefill = ""
                st.rerun()
            else:
                st.error("Something went wrong — check your API is running")
        else:
            st.warning("Please enter some text first!")

    st.divider()

    # Batch moderation
    st.subheader("📋 Batch Text Check")
    batch_input = st.text_area(
        "Enter multiple texts (one per line):",
        placeholder="Comment 1\nComment 2\nComment 3",
        height=150
    )

    if st.button("Moderate Batch", use_container_width=True):
        texts = [t.strip() for t in batch_input.strip().split("\n") if t.strip()]
        if texts:
            with st.spinner(f"Analyzing {len(texts)} texts..."):
                result = moderate_batch(texts)

            if result:
                st.info(
                    f"Processed {result['total_texts']} texts — "
                    f"🚨 {result['toxic_count']} toxic | "
                    f"✅ {result['clean_count']} clean | "
                    f"⚡ {result['avg_latency_ms']}ms avg"
                )

                rows = []
                for r in result["results"]:
                    rows.append({
                        "Text":       r["text"][:50] + "..." if len(r["text"]) > 50 else r["text"],
                        "Label":      r["label"].upper(),
                        "Confidence": f"{r['confidence']}%",
                        "Latency":    f"{r['latency_ms']}ms",
                        "is_toxic":   r["is_toxic"]
                    })
                    add_to_history(r)

                df = pd.DataFrame(rows)
                st.dataframe(
                    df.drop(columns=["is_toxic"]),
                    use_container_width=True,
                    hide_index=True
                )
                # FIX 3 — rerun after batch so metrics update
                st.rerun()
        else:
            st.warning("Please enter at least one line of text!")

with right:
    st.subheader("📊 Live Analytics")

    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)

        pie_data = df_hist["label"].value_counts().reset_index()
        pie_data.columns = ["Label", "Count"]
        fig_pie = px.pie(
            pie_data,
            names="Label",
            values="Count",
            color="Label",
            color_discrete_map={"toxic": "#E24B4A", "clean": "#1D9E75"},
            title="Toxic vs Clean"
        )
        fig_pie.update_layout(
            margin=dict(t=40, b=0, l=0, r=0),
            height=260,
            showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        fig_lat = px.line(
            df_hist.reset_index(),
            x="index",
            y="latency_ms",
            title="Inference Latency (ms)",
            labels={"index": "Request #", "latency_ms": "Latency (ms)"}
        )
        fig_lat.add_hline(
            y=50, line_dash="dash",
            line_color="red",
            annotation_text="50ms target"
        )
        fig_lat.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=220)
        st.plotly_chart(fig_lat, use_container_width=True)

    else:
        st.info("Start moderating text to see live analytics here!")

# ── 8. History table ───────────────────────────────────────────────────────
st.divider()
st.subheader("📜 Moderation History")

if st.session_state.history:
    df_show = pd.DataFrame(st.session_state.history)
    df_show = df_show.drop(columns=["is_toxic"])
    df_show.columns = ["Text", "Label", "Confidence %", "Latency ms", "Time"]
    st.dataframe(df_show[::-1], use_container_width=True, hide_index=True)

    if st.button("🗑️ Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.info("No moderation history yet — start checking some text above!")

# ── 9. Sidebar ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model Info")
    st.markdown("""
    **Model:** DistilBERT-base-uncased  
    **Optimization:** ONNX Runtime  
    **Precision:** 90% (toxic class)  
    **Accuracy:** 93% overall  
    **Latency:** ~27ms avg  
    **Max tokens:** 64  
    """)

    st.divider()
    st.header("Test Samples")
    samples = [
        "You are so stupid and ugly!",
        "Have a wonderful day!",
        "I will hurt you badly",
        "Thanks for your help!",
        "Go kill yourself",
    ]

    # FIX 4 — sample buttons now actually fill the text area
    for s in samples:
        if st.button(s[:35] + "...", use_container_width=True, key=f"sample_{s[:10]}"):
            st.session_state.prefill = s
            st.rerun()