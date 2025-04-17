Initial Proposal: Personal Finance Q&A with Retrieval-Augmented Generation
1. Introduction and Problem Statement

I’ve been manually maintaining a personal finance spreadsheet in Google Sheets for over a year now. While it helps me stay on top of my expenses, the process is tedious and easy to fall behind on. I started using Sheets because most personal finance apps rely on Plaid for automatic bank integration — and while Plaid works decently for large banks, it doesn’t support the mix of big banks and credit unions I use. On top of that, most apps I tried had frustrating or cluttered user experiences, making them more work than help. So I stuck with Google Sheets.

I've wanted to build my own app for a while — something clean, private, and tailored to how I think about money. But for now, I want to focus on the LLM side of the problem: How can we make this spreadsheet more intelligent?

This project aims to build a natural language Q&A system over my financial records using Retrieval-Augmented Generation (RAG). The system should let me ask questions like:

“How much did I spend on groceries last month?”
“Did I spend more on subscriptions this month than last?”
“What were my top 3 spending categories this quarter?”

These kinds of insights often take several filters, formulas, or pivot tables to get — and an LLM interface could dramatically reduce that friction.



2. Data Sources
The primary data source is my personal finance spreadsheet, which I maintain manually. Each entry includes the following fields:

Date: When the transaction occurred
Place: Where I spent the money (merchant name or context)
Amount in USD: The amount spent in USD (primary base)
Amount in INR: Converted amount using Google Finance
Category: Expense category (e.g., groceries, rent, travel)
Source: The source of the money — dropdown for bank accounts, credit cards, or cash

Because I maintain accounts in both the U.S. and India, I track transactions in multiple currencies. In 2024, I also tracked NTD (Taiwan Dollar) and THB (Thai Baht) since I studied abroad in Taiwan and spent time in Thailand. I used the Google Finance function within the spreadsheet to dynamically convert between currencies, depending on the transaction’s context.Cash spending is also an essential part of this system — and it's another major reason why I avoided finance apps with poor support for manual entries. Tracking cash (especially while abroad) was critical to getting a complete picture of my finances.

For the RAG system, each transaction will be converted to natural-language entries, such as:
“On February 17th, I spent ₹820 ($9.60) at a coffee shop in Chennai using my Axis Bank Savings Account. It was categorized as ‘Eating/Drinking Out’.”

These will be embedded and stored in a vector database to support semantic retrieval and LLM-powered reasoning. The system should be able to handle queries involving:
- Currency awareness and conversion, meaning ability to set primary currency so that if spending with other currencies, a converted figure in the primary currency also appers
- Temporal context (“last month,” “Q1 2024”)
- Category- or source-based filtering
- Aggregations (e.g., totals or comparisons)



3. Methods and Technologies
Core ML Stack
- Embedding model: all-MiniLM-L6-v2 (via SentenceTransformers)
- Vector database: FAISS
- Language model: Open-source model (like Mistral 7B) or API-based LLM (like Claude/GPT)
- Pipeline: RAG-style system with user query → retrieve relevant transactions → generate natural language answer
- Evaluation: Manual + automated evaluation of answer accuracy and hallucination rate

Tools
Python + Jupyter notebooks

Optional: Streamlit or Gradio for a simple UI demo



4. Products to Be Delivered
- Jupyter notebooks for:
    - Data parsing and preprocessing from spreadsheet
    - Vector store creation and document embedding
    - RAG-based Q&A pipeline
    - Example prompts and output evaluation
- Projecy 4 GitHub repository
- Final written report (≤10 pages)
- 10 minute video walkthrough



  5. Stretch Goals
- Trend summaries over time (e.g., monthly category-level spend)
- Build a basic personal finance dashboard powered by the LLM
