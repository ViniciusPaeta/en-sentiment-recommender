"""
Streamlit UI for manual testing.

- Sentiment tab: enter text, see positive/negative label
- Recommender tab: enter query, see top-K similar reviews

Important: When running with `streamlit run`, ensure the project
root is in PYTHONPATH:
    export PYTHONPATH=$PWD
"""

import sys
from pathlib import Path

# Add project root to PYTHONPATH when run via `streamlit run`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import streamlit as st  # noqa: E402
from src.models.infer import predict  # noqa: E402
from src.recommender.recommend import recommend  # noqa: E402

st.title("EN Sentiment + Content-Based Recommender")

with st.expander("Sentiment", expanded=True):
    txt = st.text_area("Text", height=140)
    if st.button("Predict"):
        if txt.strip():
            label = predict(txt)[0]
            st.success(f"Label: {'positive' if label==1 else 'negative'}")
        else:
            st.warning("Please enter some text.")

with st.expander("Recommender", expanded=False):
    query = st.text_input("Query", placeholder="e.g., space adventure with friendship")
    topk = st.slider("Top K", 1, 10, 5)
    if st.button("Search"):
        if query.strip():
            res = recommend(query, topk)
            for i, (idx, score, txt) in enumerate(res, 1):
                st.write(f"**#{i}** — score={score:.4f}")
                st.write(txt[:600])
                st.write("---")
        else:
            st.warning("Please enter a query first.")
