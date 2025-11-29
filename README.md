# AI-Driven SIEM Assistant
### Natural Language → SIEM Query Translator using NLP, RAG & Agentic AI

## 📌 Project Description
Modern SIEM platforms like Elastic SIEM and Wazuh require analysts to write complex KQL or DSL queries to investigate security threats. This creates a barrier for beginners and slows down incident response for experienced analysts.

The **AI-Driven SIEM Assistant** solves this problem by introducing a conversational interface that allows users to query SIEM data using plain English. The system uses:

- **Natural Language Processing (NLP)** to understand user intent
- **Retrieval-Augmented Generation (RAG)** to bring contextual knowledge from schemas and previous logs
- **Agentic AI reasoning** to refine queries, maintain context, and automate multi-step analysis

The assistant transforms queries like:

> "Show me all failed login attempts from external IPs"

into optimized SIEM queries (KQL/DSL), executes them, and visualizes results through a modern dashboard.

This makes security analysis faster, more intuitive, and accessible to all skill levels.

---

## 🚀 Features

### 🔍 Natural Language → SIEM Query
Turn human language into valid KQL/DSL queries automatically.

### 🧠 NLP + RAG Contextual Intelligence
Retrieves schema knowledge and historical logs to improve accuracy.

### 🤖 Agentic AI Reasoning
Understands follow-up questions, refines searches, and performs multi-step investigations.

### 📊 Interactive Visual Dashboard
Streamlit-powered interface for charts, tables, summaries, and insights.

### 🧩 Modular & Scalable
Easy to integrate with Elastic SIEM, Wazuh, or custom log pipelines.

---

## 📁 Project Structure
```
AI-Driven-SIEM-Assistant/
├── src/
│   ├── nlp_parser.py
│   ├── query_generator.py
│   ├── rag_retriever.py
│   ├── response_formatter.py
│   └── dashboard_app.py
├── train_siem_nlp.py
├── requirements.txt
└── README.md
```

---

## 🖥️ System Architecture
```
User Query 
    ↓
NLP Parser (Intent + Entity Extraction)
    ↓
Query Generator (KQL/DSL Builder)
    ↓
RAG Retriever (Context, Schema, Prior Logs)
    ↓
Agentic AI (Reasoning & Refinement)
    ↓
Response Formatter
    ↓
Streamlit Dashboard (Visualization)
```

---

## 🧠 Model Training
The NLP classifier is trained on the **UNSW-NB15 dataset**, containing normal and malicious network traffic.

**Training details:**
- Model: DistilBERT
- Epochs: 3
- Train/Test Split: 80/20
- Achieved Accuracy: 97.8%
- Metrics: Precision, Recall, F1-score

Run training:
```
python train_siem_nlp.py
```

---

## 📥 Installation Guide

### 1. Clone this repository
```
git clone https://github.com/anupam2607/AI-Driven-SIEM-Assistant.git
cd AI-Driven-SIEM-Assistant
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Launch the UI Dashboard
```
streamlit run src/dashboard_app.py
```

---

## 📊 Model Performance
| Metric | Value |
|--------|-------|
| Train Accuracy | 97% |
| Test Accuracy | 97.8% |
| Avg. F1 Score | 0.96 |

---

## 🧪 Example User Queries
| User Query | Generated DSL Query | Purpose |
|------------|----------------------|---------|
| “Show all failed logins from external IPs.” | status:failed AND NOT src.ip:(10.* OR 192.168.*) | Detect suspicious login failures |
| “Find malware activity this week.” | attack_cat:"malware" AND timestamp:[now-7d TO now] | Time-based malware analysis |
| “Show VPN login failures.” | service:"vpn" AND status:"failed" | Investigate VPN authentication issues |

---

## 📚 Dataset
This project uses the **UNSW-NB15 dataset**, a modern benchmark dataset for intrusion detection research.

Dataset link:  
🔗 https://research.unsw.edu.au/projects/unsw-nb15-dataset

