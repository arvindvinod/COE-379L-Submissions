# 💸 Personal Finance Q&A with Retrieval-Augmented Generation (RAG)

## Overview

This project explores how Large Language Models (LLMs) can be used to make personal finance management more intelligent and conversational through **Retrieval-Augmented Generation (RAG)**.

Like many others, I’ve been manually maintaining a personal finance spreadsheet in Google Sheets. While this gives me full control over my data, it's time-consuming and easy to fall behind. I initially adopted this workflow because most personal finance apps rely on **Plaid**, which doesn’t work well with my mix of big banks and credit unions. On top of that, the user experience of existing apps was often cluttered, buggy, or lacked support for manual entry — especially for **cash** transactions and **multi-currency tracking**.

I've wanted to build my own app for a while — something clean, privacy-respecting, and tailored to how *I* think about money. But for now, this project focuses on the **LLM** side of the problem:  
> 🔍 *How can we make my spreadsheet usable for natural language Q&A?*

---

## 🧠 What This Project Does

This system enables questions like:

- “How much did I spend on groceries last month?”
- “Did I spend more on subscriptions this month than last?”
- “What were my top 3 spending categories this quarter?”
- “How much cash did I spend in India during summer break?”

Using a **RAG pipeline**, it retrieves relevant transactions from my records and passes them to an LLM to generate a natural language response. This removes the friction of writing filters, pivot tables, or formulas just to gain everyday insights.

---

## 📊 Data Source

The primary dataset is my personal finance spreadsheet, which I maintain manually. Each transaction includes:

- **Date**: When the transaction occurred  
- **Place**: Where I spent the money (merchant name or context)  
- **Amount in USD**: Primary amount (since I live in the U.S.)  
- **Amount in INR**: Converted amount using `GOOGLEFINANCE()`  
- **Category**: Expense type (e.g., groceries, rent, travel)  
- **Source**: Which account/card/cash (dropdown)

### 🌐 Currency Tracking

I track spending in multiple currencies — mainly:

- 🇺🇸 USD (U.S. expenses)
- 🇮🇳 INR (India-based accounts and trips)
- 🇹🇼 NTD (Study abroad in Taiwan)
- 🇹🇭 THB (Travel in Thailand)

For each transaction, I use formulas in Sheets to convert the primary currency into others using exchange rates from `GOOGLEFINANCE()`. This allows me to unify and compare spending across borders. Manual tracking is especially important for **cash**, which I spent heavily while abroad.

---

## 🔍 Example Conversion

To make the data usable by the LLM, transactions are turned into natural language entries like:

> _“On February 17th, I spent ₹820 ($9.60) at a coffee shop in Chennai using my Axis Bank Savings Account. It was categorized as ‘Eating/Drinking Out’.”_

These entries are embedded and stored in a vector database to support semantic retrieval.

---

## 🛠 Methods and Technologies

### 🔧 Core Stack
- **Embedding model**: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- **Vector store**: FAISS (local)
- **Language model**: Open-source LLM (e.g., Mistral 7B) or API (GPT/Claude)
- **Pipeline**: Query → Retrieve similar entries → Generate answer
- **Eval**: Manual accuracy checks + basic hallucination tracking

### 📦 Tools
- `Python` + `pandas` for data handling
- `SentenceTransformers` for embedding
- `FAISS` for vector storage
- `Jupyter Notebook` for experiments and demos
- *(Optional)* `Streamlit` or `Gradio` for basic UI

---

## 🚀 Deliverables

- 📓 Jupyter notebooks:
  - Data preprocessing + parsing from Google Sheets
  - Embedding + vector store creation
  - RAG-based Q&A pipeline
  - Prompt examples + response evaluation
- 📁 GitHub repo
- 📝 Final report (≤10 pages)
- 🎥 10-minute video walkthrough

---

## ✨ Stretch Goals

- Time-based trends (e.g., monthly spend by category)
- Summarization (e.g., "Give me a weekly spending report")
- A lightweight finance dashboard UI with natural language support

---

## 💬 Why This Matters

Spreadsheets are powerful, but not intuitive. By combining the structure of my personal data with the reasoning capabilities of LLMs, I hope to build a new kind of interface — one that makes managing money **smarter, simpler, and a lot more human**.

---

