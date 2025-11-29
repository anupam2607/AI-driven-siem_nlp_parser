# src/app_streamlit.py
import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

# ✅ FIX: Ensure project root is added to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ✅ Imports
from src.nlp_infer import nlp_output_from_text
from src.query_generator import generate_elastic_query, generate_kql_string
from src.siem_connector import simulated_siem_search
from src.rag_retriever import rag_retrieve
from src.agentic_ai import summarize_logs

# ✅ Streamlit setup
st.set_page_config(page_title="SIEM -  Agentic ai  + rag  Dashboard", layout="wide")
st.title("🛡️ SIEM-  Agentic ai + rag  Dashboard")

# -------------------------------
# Load dataset
# -------------------------------
CSV = os.path.join(PROJECT_ROOT, "data.csv")
if not os.path.exists(CSV):
    st.warning("⚠️ data.csv not found in project root. Run `python data/generate_sample_data.py` first.")
    df = pd.DataFrame({"text": [], "label": []})
else:
    df = pd.read_csv(CSV)
    st.sidebar.success(f"✅ Loaded dataset with {len(df)} records")

# -------------------------------
# 📊 Dashboard Overview
# -------------------------------
st.sidebar.header("📈 Dataset Overview")

if not df.empty:
    # Label Distribution
    if "label" in df.columns:
        label_counts = df["label"].value_counts().reset_index()
        label_counts.columns = ["Label", "Count"]
        fig_label = px.bar(label_counts, x="Label", y="Count", title="Label Distribution", color="Label")
        st.sidebar.plotly_chart(fig_label, use_container_width=True)

    # Word Frequency (Basic)
    if "text" in df.columns:
        from collections import Counter
        import re
        all_words = " ".join(df["text"].astype(str).tolist()).lower()
        words = re.findall(r"\b\w+\b", all_words)
        freq = Counter(words).most_common(10)
        freq_df = pd.DataFrame(freq, columns=["Word", "Frequency"])
        fig_words = px.bar(freq_df, x="Word", y="Frequency", title="Top 10 Frequent Words", color="Frequency")
        st.sidebar.plotly_chart(fig_words, use_container_width=True)

# -------------------------------
# User Input
# -------------------------------
st.subheader("🔍 Enter Your Query")
user_query = st.text_input("Example: 'Show all Exploits on FTP connections'", value="Show Exploits on ftp")

# -------------------------------
# Pipeline Execution
# -------------------------------
if st.button("🚀 Run Pipeline"):
    with st.spinner("Running full SIEM AI pipeline..."):

        # 1️⃣ NLP Parser
        st.markdown("### 🧩 Step 1: NLP Parser Output")
        nlp_out = nlp_output_from_text(user_query)
        st.json(nlp_out)

        # 2️⃣ Query Generator
        st.markdown("### 🔧 Step 2: Generated Queries")
        elastic_query = generate_elastic_query(nlp_out)
        kql_query = generate_kql_string(nlp_out)
        col1, col2 = st.columns(2)
        with col1:
            st.code(elastic_query, language="json")
        with col2:
            st.code(kql_query, language="kql")

        # 3️⃣ SIEM Connector
        st.markdown("### 🧾 Step 3: SIEM Search Results")
        if df.empty:
            st.warning("No data to search.")
            results = pd.DataFrame()
        else:
            results = simulated_siem_search(nlp_out, df)
            if results.empty:
                st.error("No matching logs found.")
            else:
                st.success(f"Found {len(results)} matching logs.")
                st.dataframe(results.head(50), use_container_width=True)

                # 📊 Interactive Log Summary
                if "label" in results.columns:
                    fig = px.histogram(results, x="label", title="Matching Logs by Label", color="label")
                    st.plotly_chart(fig, use_container_width=True)

                # 📈 Time-based trends (if timestamp available)
                if "timestamp" in results.columns:
                    try:
                        results["timestamp"] = pd.to_datetime(results["timestamp"])
                        time_fig = px.line(results, x="timestamp", title="Timeline of Log Activity", markers=True)
                        st.plotly_chart(time_fig, use_container_width=True)
                    except Exception:
                        pass

        # 4️⃣ RAG Retriever (Fixed)
        st.markdown("### 📚 Step 4: RAG Retriever (related past threats)")
        if "text" in df.columns and not df["text"].empty:
            docs = [t.strip() for t in df["text"].astype(str).tolist() if isinstance(t, str) and t.strip()]
        else:
            docs = []

        if not docs:
            st.warning("⚠️ No valid text documents found for RAG retrieval.")
            rag_hits = []
        else:
            try:
                rag_hits = rag_retrieve(user_query, docs, top_k=5)
            except ValueError as e:
                st.error(f"RAG Retriever error: {e}")
                rag_hits = []

        if not rag_hits:
            st.warning("No related documents found.")
        else:
            for hit in rag_hits:
                st.markdown(f"**Score:** {hit['score']:.3f}")
                st.write(hit["text"])
                st.divider()

        # 5️⃣ Agentic AI Summarizer
        st.markdown("### 🤖 Step 5: Agentic AI Summary")
        log_texts = results["text"].astype(str).tolist() if not results.empty and "text" in results.columns else []
        rag_texts = [h["text"] for h in rag_hits] if rag_hits else []
        combined_docs = log_texts + rag_texts

        if not combined_docs:
            st.info("No data available for summarization.")
        else:
            summary = summarize_logs(combined_docs)
            st.info(summary)

        st.success("✅ Pipeline completed successfully.")
