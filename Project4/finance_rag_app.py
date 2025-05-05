import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
from datetime import datetime, timedelta
import re
import requests
import json
from io import StringIO

# Constants
API_KEY = "AIzaSyCg2A6bWR3Xd16wyM7YnapV9NX3XJqCtIY"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
HEADERS = {"Content-Type": "application/json"}
ENCODINGS = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252', 'utf-16']
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
EMBEDDING_SIZE = 384

class FinanceRAG:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.data = pd.DataFrame()
        self.index = None
        self.embeddings = None

    def load_data(self, uploaded_file, encoding='ISO-8859-1'):
        try:
            content = uploaded_file.getvalue().decode(encoding)
            content = self._clean_content(content)
            
            try:
                date_samples = pd.read_csv(StringIO(content), nrows=5)['Date']
                day_first = any(
                    d.split('/')[0].isdigit() and int(d.split('/')[0]) > 12 
                    for d in date_samples if '/' in str(d)
                )
            except Exception as e:
                day_first = False

            self.data = pd.read_csv(
                StringIO(content),
                parse_dates=['Date'],
                dayfirst=day_first,
                infer_datetime_format=True,
                converters={
                    'Amount (INR)': lambda x: float(str(x).replace(',', '')),
                    'Amount (THB)': lambda x: float(str(x).replace(',', '')),
                    'Amount (TWD)': lambda x: float(str(x).replace(',', '')),
                    'Amount (USD)': lambda x: float(str(x).replace(',', ''))
                }
            )

            if not pd.api.types.is_datetime64_any_dtype(self.data['Date']):
                self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')
            
            if self.data['Date'].isnull().any():
                st.error("Failed to parse some dates. Please check date format.")
                return False

            if self.data['Date'].dt.year.min() < 2000:
                st.error("Invalid date parsing - check date format in CSV")
                return False

            # Generate text representations
            self.data['text'] = self.data.apply(self._transaction_to_text, axis=1)
            st.success(f"Loaded {len(self.data)} transactions")
            return True
        
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False

    def _clean_content(self, content):
        currency_symbols = r'[$\¥€₹฿₩£]'
        return re.sub(
            f'([^{currency_symbols}])([^\x00-\x7F])',
            lambda m: m.group(1) + ' ',
            content
        )

    def _transaction_to_text(self, row):
        date = pd.to_datetime(row['Date']).strftime('%B %d, %Y')
        amount_text = self._format_amount(row)
        return (
            f"On {date}, spent {amount_text} at {row['Name']} "
            f"({row['Category']} via {row['Source']})"
        )

    def _format_amount(self, row):
        currency = row['Currency Used']
        amounts = {
            'USD': row.get('Amount (USD)', 0),
            'TWD': row.get('Amount (TWD)', None),
            'INR': row.get('Amount (INR)', None),
            'THB': row.get('Amount (THB)', None)
        }
        
        if currency == 'USD':
            return f"${amounts['USD']:.2f}"
        else:
            native_amount = amounts.get(currency, None)
            usd_amount = amounts['USD']
            if native_amount is not None:
                return f"{native_amount:.2f} {currency} (${usd_amount:.2f} USD)"
            return f"${usd_amount:.2f} USD"

    def create_embeddings(self):
        texts = self.data['text'].tolist()
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        return True

    def build_index(self):
        if self.embeddings is None:
            st.error("Create embeddings first")
            return False

        self.index = faiss.IndexFlatL2(EMBEDDING_SIZE)
        self.index.add(self.embeddings.astype('float32'))
        return True

    def search(self, query, k=10):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), k)
        return [self.data.iloc[i] for i in indices[0] if i < len(self.data)]
    
    def smart_search(self, query, min_k=15, max_k=100):
        k = self._get_dynamic_k(query, min_k, max_k)
        
        date_range = self._extract_date_range(query)
        currencies = self._extract_currencies(query)
        categories = self._extract_categories(query)
        
        mask = pd.Series([True] * len(self.data), index=self.data.index)
        
        if date_range:
            st.write(f"🔍 Searching between {date_range['start'].strftime('%Y-%m-%d')} and {date_range['end'].strftime('%Y-%m-%d')}")
            data_start = self.data['Date'].min()
            data_end = self.data['Date'].max()
            
            if date_range['start'] > data_end or date_range['end'] < data_start:
                st.warning("⚠️ Your query dates are outside the dataset's range")
                return []
            
            date_mask = (self.data['Date'] >= date_range['start']) & (self.data['Date'] <= date_range['end'])
            mask = mask & date_mask
            
        if currencies:
            currency_mask = self.data['Currency Used'].isin(currencies)
            mask = mask & currency_mask
            
        if categories:
            category_mask = pd.Series([False] * len(self.data), index=self.data.index)
            for category in categories:
                category_mask = category_mask | self.data['Category'].str.contains(category, case=False)
            mask = mask & category_mask
        
        filtered_data = self.data[mask]
        
        if len(filtered_data) == 0:
            return []
        
        if 0 < len(filtered_data) < len(self.data) * 0.8:
            return self._search_filtered_subset(query, filtered_data, k)
        elif len(filtered_data) >= len(self.data) * 0.8:
            return self.search(query, k=k)
        else:
            return []
    
    def _search_filtered_subset(self, query, filtered_data, k):
        filtered_indices = filtered_data.index.tolist()
        filtered_embeddings = np.vstack([self.embeddings[i] for i in filtered_indices])
        
        temp_index = faiss.IndexFlatL2(EMBEDDING_SIZE)
        temp_index.add(filtered_embeddings.astype('float32'))
        
        query_embedding = self.model.encode([query])
        distances, indices = temp_index.search(query_embedding.astype('float32'), min(k, len(filtered_data)))
        
        return [filtered_data.iloc[i] for i in indices[0] if i < len(filtered_data)]
    
    def _get_dynamic_k(self, query, min_k=15, max_k=100, default_k=25):
        date_patterns = [r'\b\d{4}\b', r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*', 
                        r'\bq[1-4]\b', r'\bquarter\b', r'\bmonth\b', r'\bweek\b']
        category_terms = [r'\bcategor\w*\b', r'\bfood\b', r'\btravel\b', r'\bentertainment\b']
        currency_terms = [r'\b(usd|twd|inr|thb)\b', r'\bcurrenc\w*\b', r'\bdollar\b']
        aggregation_terms = [r'\btotal\b', r'\bsum\b', r'\baverage\b', r'\bmean\b', 
                           r'\bmost\b', r'\blargest\b', r'\bsmallest\b', r'\bspending\b', r'\bexpenses\b']
        
        total_matches = 0
        for pattern_list in [date_patterns, category_terms, currency_terms, aggregation_terms]:
            total_matches += sum(1 for pattern in pattern_list if re.search(pattern, query.lower()))
        
        k = default_k + (total_matches * 15)
        return max(min_k, min(k, max_k))
    
    def _extract_date_range(self, query):
        query_lower = query.lower()
        now = datetime.now()
        month_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }

        full_month_pattern = r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b'
        full_month_match = re.search(full_month_pattern, query_lower)
        if full_month_match:
            month_name = full_month_match.group(1)
            year = self._extract_year(query_lower) or self._infer_year_from_data()
            month_num = month_map[month_name.lower()]
            return self._create_month_range(year, month_num)

        patterns = [
            (r'\blast month\b', lambda: self._last_month_range(now)),
            (r'\bthis month\b', lambda: {'start': now.replace(day=1), 'end': now}),
            (r'\blast year\b', lambda: {
                'start': datetime(now.year - 1, 1, 1),
                'end': datetime(now.year - 1, 12, 31)
            }),
            (r'\bthis year\b', lambda: {
                'start': datetime(now.year, 1, 1),
                'end': now
            }),
            (r'(?:q([1-4])|(?:first|second|third|fourth) quarter)', self._handle_quarter),
            (r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*', self._handle_month_abbreviation)
        ]

        for pattern, handler in patterns:
            match = re.search(pattern, query_lower)
            if match:
                result = handler(match) if callable(handler) else handler
                if result:
                    if 'start' in result and 'end' in result:
                        return result
        return None

    def _extract_year(self, query_lower):
        year_match = re.search(r'\b(20\d{2})\b', query_lower)
        if year_match:
            return int(year_match.group(1))
        
        if not self.data.empty:
            return self.data['Date'].dt.year.mode()[0]
        
        return datetime.now().year

    def _infer_year_from_data(self):
        if self.data.empty:
            return datetime.now().year
        return self.data['Date'].dt.year.mode()[0]

    def _create_month_range(self, year, month_num):
        if month_num == 2:
            last_day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
        elif month_num in [4, 6, 9, 11]:
            last_day = 30
        else:
            last_day = 31
            
        return {
            'start': datetime(year, month_num, 1),
            'end': datetime(year, month_num, last_day)
        }

    def _last_month_range(self, now):
        last_month = (now.replace(day=1) - timedelta(days=1))
        return {
            'start': last_month.replace(day=1),
            'end': last_month
        }

    def _handle_quarter(self, match):
        quarter = int(match.group(1)) if match.group(1) else \
                 1 if 'first' in match.group(0) else \
                 2 if 'second' in match.group(0) else \
                 3 if 'third' in match.group(0) else 4
        
        year = self._extract_year(match.string) or self._infer_year_from_data()
        start_month = (quarter - 1) * 3 + 1
        end_month = quarter * 3
        
        return self._create_quarter_range(year, start_month, end_month)

    def _create_quarter_range(self, year, start_month, end_month):
        start_date = datetime(year, start_month, 1)
        end_day = 31
        if end_month == 2:
            end_day = 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
        elif end_month in [4, 6, 9, 11]:
            end_day = 30
        end_date = datetime(year, end_month, end_day)
        return {'start': start_date, 'end': end_date}

    def _handle_month_abbreviation(self, match):
        month_abbr = match.group(1)[:3].lower()
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
            'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
            'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month_num = month_map.get(month_abbr)
        if not month_num:
            return None
            
        year = self._extract_year(match.string) or self._infer_year_from_data()
        return self._create_month_range(year, month_num)
    
    def _extract_currencies(self, query):
        query_lower = query.lower()
        currencies = []
        
        if re.search(r'\busd\b', query_lower):
            currencies.append('USD')
        if re.search(r'\btwd\b', query_lower):
            currencies.append('TWD')
        if re.search(r'\binr\b', query_lower):
            currencies.append('INR')
        if re.search(r'\bthb\b', query_lower):
            currencies.append('THB')
            
        if re.search(r'\bdollar', query_lower):
            currencies.append('USD')
        if re.search(r'\btaiwan', query_lower):
            currencies.append('TWD')
        if re.search(r'\bindian|rupee', query_lower):
            currencies.append('INR')
        if re.search(r'\bthai|baht', query_lower):
            currencies.append('THB')
            
        return currencies if currencies else None
    
    def _extract_categories(self, query):
        query_lower = query.lower()
        
        category_patterns = [
            r'\bfood\b', r'\bgroceries\b', r'\brestaurant\b', r'\bdining\b',
            r'\btravel\b', r'\btransportation\b', r'\bentertainment\b', 
            r'\bshopping\b', r'\bclothing\b', r'\bhealth\b', r'\butilities\b',
            r'\bhousing\b', r'\beducation\b', r'\bsubscription\b'
        ]
        
        categories = []
        for pattern in category_patterns:
            if re.search(pattern, query_lower):
                match = re.search(pattern, query_lower)
                if match:
                    categories.append(match.group(0))
                    
        return categories if categories else None
    
    def filter_by_date_range(self, start_date, end_date):
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            mask = (self.data['Date'] >= start) & (self.data['Date'] <= end)
            return self.data[mask]
        except Exception as e:
            st.error(f"Error filtering by date: {str(e)}")
            return pd.DataFrame()
    
    def get_largest_expenses(self, n=5, currency='USD'):
        if currency == 'USD':
            col = 'Amount (USD)'
        else:
            col = f'Amount ({currency})'
            
        if col not in self.data.columns:
            st.warning(f"Currency {currency} not found in data")
            return pd.DataFrame()
            
        return self.data.sort_values(by=col, ascending=False).head(n)

    def save_system(self, path='rag_system'):
        os.makedirs(path, exist_ok=True)
        self.data.to_csv(f'{path}/data.csv', index=False)
        np.save(f'{path}/embeddings.npy', self.embeddings)
        faiss.write_index(self.index, f'{path}/index.faiss')

    def load_system(self, path='rag_system'):
        try:
            self.data = pd.read_csv(f'{path}/data.csv')
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.embeddings = np.load(f'{path}/embeddings.npy')
            self.index = faiss.read_index(f'{path}/index.faiss')
            return True
        except Exception as e:
            st.error(f"Error loading RAG system: {str(e)}")
            return False

def query_gemini(context, question, dataframe_summary=None):
    df_context = ""
    if dataframe_summary is not None:
        df_context = f"""
        Dataset Summary:
        - Total transactions: {dataframe_summary['count']}
        - Date range: {dataframe_summary['date_range']}
        - Available currencies: {dataframe_summary['currencies']}
        - Categories: {dataframe_summary['categories']}
        """

    prompt = f"""Analyze these multi-currency transactions and answer the question:
    
    {df_context}
    
    Transactions (Format: Date | Merchant | Amounts | Category | Source):
    {context}

    Question: {question}

    Guidelines:
    1. Always show original currency and USD equivalent
    2. Convert amounts when comparing different currencies
    3. Highlight currency-specific patterns
    4. Mention payment sources when relevant
    5. Use approximate exchange rates if needed
    6. Make sure to analyze ALL transactions in the provided context
    7. If the data doesn't contain information for a specific time period mentioned in the question, clearly state this
    """

    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topP": 0.95,
            "maxOutputTokens": 2000
        }
    }

    try:
        response = requests.post(
            f"{API_URL}?key={API_KEY}",
            headers=HEADERS,
            json=payload
        )
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_in_chunks(context_entries, question, df_summary, chunk_size=20):
    if len(context_entries) <= chunk_size:
        context_str = "\n".join(context_entries)
        return query_gemini(context_str, question, df_summary)
    
    chunks = [context_entries[i:i+chunk_size] for i in range(0, len(context_entries), chunk_size)]
    results = []
    
    first_chunk_context = "\n".join(chunks[0])
    first_chunk_prompt = f"{question} (Analyzing first batch of transactions)"
    first_result = query_gemini(first_chunk_context, first_chunk_prompt, df_summary)
    results.append(first_result)
    
    if len(chunks) > 3:
        middle_idx = len(chunks) // 2
        middle_chunk_context = "\n".join(chunks[middle_idx])
        middle_chunk_prompt = f"{question} (Analyzing middle batch of transactions)"
        middle_result = query_gemini(middle_chunk_context, middle_chunk_prompt, df_summary)
        results.append(middle_result)
        
        last_chunk_context = "\n".join(chunks[-1])
        last_chunk_prompt = f"{question} (Analyzing final batch of transactions)"
        last_result = query_gemini(last_chunk_context, last_chunk_prompt, df_summary)
        results.append(last_result)
    else:
        for i, chunk in enumerate(chunks[1:], 1):
            chunk_context = "\n".join(chunk)
            chunk_prompt = f"{question} (Analyzing batch {i+1}/{len(chunks)})"
            chunk_result = query_gemini(chunk_context, chunk_prompt, df_summary)
            results.append(chunk_result)
    
    synthesis_prompt = f"""Based on the following batch analyses, provide a comprehensive answer to: {question}
    
    First Batch Analysis:
    {results[0]}
    
    {'Middle Batch Analysis:' if len(results) > 2 else 'Additional Batch Analysis:'}
    {results[1] if len(results) > 1 else ''}
    
    {'Final Batch Analysis:' if len(results) > 2 else ''}
    {results[2] if len(results) > 2 else ''}
    
    Synthesize a complete answer considering all batches of transactions.
    """
    
    return query_gemini("", synthesis_prompt, df_summary)

def main():
    st.set_page_config(page_title="Personal Finance Q&A with RAG", layout="wide")
    st.title("Personal Finance Q&A with Retrieval-Augmented Generation")

    if 'rag' not in st.session_state:
        st.session_state.rag = FinanceRAG()
        if os.path.exists('rag_system'):
            st.session_state.rag.load_system()

    with st.sidebar:
        st.header("Data Management")
        uploaded_file = st.file_uploader("Upload CSV", type='csv')
        selected_encoding = st.selectbox("File Encoding", ENCODINGS, index=0)
        
        if uploaded_file:
            if st.button("Process Data"):
                with st.spinner("Processing..."):
                    success = False
                    if st.session_state.rag.load_data(uploaded_file, selected_encoding):
                        success = True
                    else:
                        for encoding in ENCODINGS:
                            if encoding == selected_encoding:
                                continue
                            uploaded_file.seek(0)
                            if st.session_state.rag.load_data(uploaded_file, encoding):
                                success = True
                                st.info(f"Used encoding: {encoding}")
                                break
                    
                    if success:
                        st.session_state.rag.create_embeddings()
                        st.session_state.rag.build_index()
                        st.session_state.rag.save_system()
                        st.rerun()
                    else:
                        st.error("Failed to process file")

        if st.button("Reload Existing Data"):
            st.session_state.rag.load_system()
            st.rerun()

    tab1, tab2, tab3 = st.tabs(["Ask Question", "View Data", "Simple Analytics"])

    with tab1:
        query = st.text_input("Enter your financial question:", 
                            placeholder="e.g. What were my largest TWD expenses last month?")
        
        use_smart_retrieval = st.checkbox("Use smart retrieval", value=True, 
                                        help="Automatically determine optimal number of transactions to analyze")
        
        k_results = 25  # Default value
        if not use_smart_retrieval:
            k_results = st.slider("Number of transactions to analyze:", 5, 1200, 25)
        
        if query:
            with st.spinner("Analyzing transactions..."):
                try:
                    df_summary = None
                    if not st.session_state.rag.data.empty:
                        df_summary = {
                            'count': len(st.session_state.rag.data),
                            'date_range': f"{st.session_state.rag.data['Date'].min().strftime('%Y-%m-%d')} to {st.session_state.rag.data['Date'].max().strftime('%Y-%m-%d')}",
                            'currencies': ', '.join(st.session_state.rag.data['Currency Used'].unique()),
                            'categories': ', '.join(st.session_state.rag.data['Category'].unique().tolist()[:5]) + 
                                         (', ...' if len(st.session_state.rag.data['Category'].unique()) > 5 else '')
                        }
                    
                    results = []
                    if use_smart_retrieval:
                        results = st.session_state.rag.smart_search(query, min_k=15, max_k=150)
                        st.info(f"Smart retrieval found {len(results)} relevant transactions")
                    else:
                        results = st.session_state.rag.search(query, k=k_results)
                    
                    if not results:
                        st.warning("No matching transactions found")
                        return
                    
                    context = []
                    for _, row in enumerate(results):
                        date = row['Date'].strftime('%Y-%m-%d')
                        context.append(
                            f"{date} | {row['Name']} | {row['text'].split('spent ')[1]} | "
                            f"{row['Category']} | {row['Source']}"
                        )
                    
                    if len(results) > 30:
                        with st.spinner("Processing large dataset in chunks..."):
                            answer = analyze_in_chunks(context, query, df_summary, chunk_size=20)
                            st.markdown(f"**Answer (based on {len(results)} transactions):**\n\n{answer}")
                    else:
                        context_str = "\n".join(context)
                        with st.spinner("Generating answer..."):
                            answer = query_gemini(context_str, query, df_summary)
                            st.markdown(f"**Answer:**\n\n{answer}")

                    with st.expander("View analyzed transactions"):
                        st.write(f"Total transactions analyzed: {len(results)}")
                        st.dataframe(pd.DataFrame(results)[['Date', 'Name', 'Category', 'Currency Used', 'Amount (USD)']])

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

    with tab2:
        if not st.session_state.rag.data.empty:
            st.subheader("Processed Transactions")
            st.dataframe(st.session_state.rag.data)
        else:
            st.warning("No data loaded")
            
    with tab3:
        if not st.session_state.rag.data.empty:
            st.subheader("Quick Analytics")
            
            col1, col2 = st.columns(2)
            with col1:
                min_date = st.session_state.rag.data['Date'].min().date()
                max_date = st.session_state.rag.data['Date'].max().date()
                start_date = st.date_input("Start Date", min_date)
            
            with col2:
                end_date = st.date_input("End Date", max_date)
            
            currencies = ['USD'] + [c for c in st.session_state.rag.data['Currency Used'].unique() if c != 'USD']
            selected_currency = st.selectbox("Select Currency", currencies)
            
            filtered_data = st.session_state.rag.filter_by_date_range(start_date, end_date)
            
            if not filtered_data.empty:
                st.subheader("Largest Expenses")
                largest_expenses = st.session_state.rag.get_largest_expenses(n=5, currency=selected_currency)
                
                if not largest_expenses.empty:
                    for idx, row in largest_expenses.iterrows():
                        date = row['Date'].strftime('%Y-%m-%d')
                        amount_col = f'Amount ({selected_currency})'
                        if selected_currency == 'USD':
                            amount_str = f"${row[amount_col]:.2f}"
                        else:
                            amount_str = f"{row[amount_col]:.2f} {selected_currency} (${row['Amount (USD)']:.2f} USD)"
                        
                        st.write(f"**{date}:** {amount_str} at **{row['Name']}** ({row['Category']} via {row['Source']})")
                else:
                    st.warning(f"No transactions found for {selected_currency}")
            else:
                st.warning("No transactions found for selected date range")

if __name__ == "__main__":
    main()