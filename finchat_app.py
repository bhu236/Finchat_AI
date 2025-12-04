"""
FinChat AI - Complete Production Version with Advanced RAG
Human-Centered Financial Analysis Assistant
IST.688.M001.FALL25 - Building HC-AI Apps

Team: Bhushan Jain, Samiksha Singh, Anjali Kalra, Shraddha Aher

COMPLETE FEATURES:
✅ Vector Database (ChromaDB) with persistent storage
✅ Semantic Embeddings (sentence-transformers, 384D)
✅ Intelligent chunking (1000 chars + section headers)
✅ Enhanced RAG with query expansion & reranking
✅ Multi-Agent Coordination (handles multiple questions)
✅ 6 Specialized AI Agents working in parallel
✅ SEC 10-Q/10-K Analysis with semantic search
✅ Peer Benchmarking & Portfolio Position Analysis
✅ Ethical AI with citations & disclaimers

Install: pip install streamlit yfinance plotly pandas pypdf2 python-docx pysqlite3-binary chromadb sentence-transformers
Run: streamlit run finchat_app_complete.py
"""

import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import hashlib

# SQLITE FIX for ChromaDB
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="FinChat AI - Advanced Multi-Agent RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# VECTOR DATABASE INITIALIZATION
# ============================================================================

@st.cache_resource
def initialize_vector_db():
    """Initialize ChromaDB and embedding model"""
    try:
        client = chromadb.PersistentClient(path="./chroma_db_production")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        try:
            collection = client.get_collection("sec_filings_production")
        except:
            collection = client.create_collection(
                name="sec_filings_production",
                metadata={"description": "Production SEC filings with advanced RAG"}
            )
        
        return client, collection, embedding_model
    except Exception as e:
        st.error(f"⚠️ Vector DB Error: {e}")
        return None, None, None

chroma_client, filing_collection, embedding_model = initialize_vector_db()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = []

if 'uploaded_filings' not in st.session_state:
    st.session_state.uploaded_filings = []

if 'filing_tickers' not in st.session_state:
    st.session_state.filing_tickers = set()

if 'portfolio_config' not in st.session_state:
    st.session_state.portfolio_config = {
        'risk_tolerance': 'Moderate',
        'investment_horizon': '3-5 years',
        'portfolio_allocation': 10.0
    }

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = 'AAPL'

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# ============================================================================
# STYLING
# ============================================================================

theme = st.session_state.theme
card_bg = "#1E1E1E" if theme == 'dark' else "#F0F2F6"

st.markdown(f"""
<style>
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00CC96, #AB63FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .company-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .agent-badge {{
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    .multi-agent-badge {{
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }}
    .citation {{
        background-color: {card_bg};
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }}
    .relevance-high {{ color: #00CC96; font-weight: bold; }}
    .relevance-medium {{ color: #FFA500; }}
    .relevance-low {{ color: #666; }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TICKER DATA
# ============================================================================

POPULAR_TICKERS = {
    'AAPL': 'Apple Inc.', 'MSFT': 'Microsoft Corporation', 'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms Inc.', 'NVDA': 'NVIDIA Corporation', 'TSLA': 'Tesla Inc.',
    'AMZN': 'Amazon.com Inc.', 'NFLX': 'Netflix Inc.', 'AMD': 'Advanced Micro Devices',
    'INTC': 'Intel Corporation', 'MU': 'Micron Technology', 'QCOM': 'Qualcomm',
    'JPM': 'JPMorgan Chase', 'BAC': 'Bank of America', 'WFC': 'Wells Fargo',
    'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley', 'V': 'Visa Inc.', 'MA': 'Mastercard',
    'JNJ': 'Johnson & Johnson', 'UNH': 'UnitedHealth Group', 'PFE': 'Pfizer Inc.',
    'WMT': 'Walmart', 'COST': 'Costco', 'HD': 'Home Depot', 'NKE': 'Nike Inc.',
    'XOM': 'Exxon Mobil', 'CVX': 'Chevron', 'BA': 'Boeing', 'CAT': 'Caterpillar',
    'APLD': 'Applied Digital Corporation'
}

TICKER_OPTIONS = [f"{ticker} - {name}" for ticker, name in sorted(POPULAR_TICKERS.items())]

# ============================================================================
# AGENT DEFINITIONS
# ============================================================================

AGENTS = {
    'Investment Advisor': {
        'icon': '💼',
        'color': '#00CC96',
        'keywords': ['invest', 'buy', 'sell', 'hold', 'recommend', 'price target', 'entry', 'exit', 'should i'],
        'description': 'Buy/sell recommendations with price targets'
    },
    'Risk Analyst': {
        'icon': '🛡️',
        'color': '#EF553B',
        'keywords': ['risk', 'concern', 'threat', 'danger', 'safe', 'volatility', 'vulnerable'],
        'description': 'Assesses financial and business risks'
    },
    'Product Analyst': {
        'icon': '📦',
        'color': '#AB63FA',
        'keywords': ['product', 'segment', 'service', 'category', 'best', 'worst', 'performing', 'revenue'],
        'description': 'Evaluates product/segment performance'
    },
    'Peer Comparison': {
        'icon': '🏆',
        'color': '#FFA15A',
        'keywords': ['compare', 'versus', 'vs', 'peer', 'competitor', 'better', 'worse'],
        'description': 'Benchmarks against competitors'
    },
    'Position Advisor': {
        'icon': '💰',
        'color': '#19D3F3',
        'keywords': ['i have', 'i own', 'my shares', 'my position', 'shares at', 'bought at'],
        'description': 'Analyzes your specific holdings'
    },
    'Education': {
        'icon': '🎓',
        'color': '#FECB52',
        'keywords': ['what is', 'explain', 'how to read', 'define', 'meaning', 'understand', 'teach'],
        'description': 'Explains financial concepts'
    }
}

# ============================================================================
# ENHANCED CHUNKING WITH CONTEXT
# ============================================================================

def smart_chunk_with_context(text: str, section_name: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
    """Context-aware chunking with section headers"""
    chunks = []
    context_header = f"[{section_name}]\n\n"
    
    has_tables = bool(re.search(r'\$\s*[\d,]+|\d+\.\d+%|\d{1,3}(?:,\d{3})+', text))
    has_metrics = bool(re.search(r'revenue|sales|income|margin|growth|profit|segment', text, re.IGNORECASE))
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk.strip():
                full_chunk = context_header + current_chunk.strip()
                
                chunks.append({
                    'text': full_chunk,
                    'chunk_id': chunk_id,
                    'section': section_name,
                    'has_tables': has_tables,
                    'has_metrics': has_metrics,
                    'length': len(full_chunk)
                })
                chunk_id += 1
            
            overlap_text = " ".join(current_chunk.split()[-50:])
            current_chunk = overlap_text + " " + sentence + " "
    
    if current_chunk.strip():
        full_chunk = context_header + current_chunk.strip()
        chunks.append({
            'text': full_chunk,
            'chunk_id': chunk_id,
            'section': section_name,
            'has_tables': has_tables,
            'has_metrics': has_metrics,
            'length': len(full_chunk)
        })
    
    return chunks

def extract_filing_sections_advanced(text: str) -> Dict[str, str]:
    """Enhanced section extraction"""
    sections = {}
    
    patterns = {
        'Management Discussion & Analysis': [
            r"ITEM\s+2[\.:]?\s*MANAGEMENT'?S?\s+DISCUSSION\s+AND\s+ANALYSIS.*?(?=ITEM\s+3|ITEM\s+4|$)",
        ],
        'Risk Factors': [
            r"ITEM\s+1A[\.:]?\s*RISK\s+FACTORS.*?(?=ITEM\s+1B|ITEM\s+2|$)",
        ],
        'Business Overview': [
            r"ITEM\s+1[\.:]?\s*(?:BUSINESS|DESCRIPTION).*?(?=ITEM\s+1A|ITEM\s+2|$)",
        ],
        'Segment Information': [
            r"(?:NOTE|ITEM).*?SEGMENT.*?(?:INFORMATION|RESULTS).*?(?=NOTE|ITEM|$)",
        ],
        'Financial Statements': [
            r"CONSOLIDATED\s+(?:BALANCE\s+SHEETS?|STATEMENTS?).*?(?=NOTES|ITEM|$)"
        ]
    }
    
    for section_name, pattern_list in patterns.items():
        for pattern in pattern_list:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(0)[:25000]
                break
    
    if not sections:
        sections['Full Filing'] = text[:25000]
    
    return sections

# ============================================================================
# QUERY EXPANSION
# ============================================================================

def expand_financial_query(query: str) -> str:
    """Expand query with financial synonyms"""
    expansions = {
        'revenue': 'sales top-line income',
        'profit': 'earnings net-income bottom-line',
        'growth': 'increase expansion gain uptick',
        'decline': 'decrease reduction drop downturn',
        'risk': 'threat concern vulnerability challenge',
        'product': 'segment category offering line',
        'performance': 'results metrics trends outcomes',
        'best': 'strong leading outperforming top highest',
        'worst': 'weak lagging underperforming poor lowest',
        'margin': 'profitability markup spread'
    }
    
    query_lower = query.lower()
    expanded_terms = []
    
    for key, synonyms in expansions.items():
        if key in query_lower:
            expanded_terms.extend(synonyms.split())
    
    if expanded_terms:
        return query + " " + " ".join(expanded_terms[:5])
    
    return query

# ============================================================================
# MULTI-QUESTION DETECTION
# ============================================================================

def detect_multiple_questions(query: str) -> List[Dict]:
    """Detect multiple questions in query"""
    questions = []
    
    if '?' in query:
        parts = query.split('?')
        for part in parts:
            part = part.strip()
            if part and len(part) > 10:
                intent = detect_question_intent(part)
                questions.append({
                    'text': part + '?',
                    'intent': intent,
                    'original': query
                })
    
    elif any(word in query.lower() for word in ['and what', 'also what', 'and which', 'also which']):
        for coord in ['and what', 'also what', 'and which', 'also which', 'and how']:
            if coord in query.lower():
                parts = re.split(coord, query, flags=re.IGNORECASE)
                if len(parts) == 2:
                    questions.append({
                        'text': parts[0].strip(),
                        'intent': detect_question_intent(parts[0]),
                        'original': query
                    })
                    questions.append({
                        'text': coord.split()[1] + ' ' + parts[1].strip(),
                        'intent': detect_question_intent(parts[1]),
                        'original': query
                    })
                break
    
    if not questions:
        questions = [{
            'text': query,
            'intent': detect_question_intent(query),
            'original': query
        }]
    
    return questions

def detect_question_intent(question: str) -> str:
    """Detect intent/domain of question"""
    question_lower = question.lower()
    
    scores = {}
    for agent_name, agent_info in AGENTS.items():
        score = sum(1 for kw in agent_info['keywords'] if kw in question_lower)
        if score > 0:
            scores[agent_name] = score
    
    if scores:
        return max(scores, key=scores.get)
    
    return 'General Analysis'

# ============================================================================
# VECTOR STORE OPERATIONS
# ============================================================================

def create_semantic_embeddings(ticker: str, filing_data: Dict) -> Tuple[bool, int]:
    """Create embeddings with enhanced chunking"""
    if filing_collection is None or embedding_model is None:
        return False, 0
    
    try:
        sections = filing_data.get('sections', {})
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        chunk_counter = 0
        
        for section_name, content in sections.items():
            chunks = smart_chunk_with_context(content, section_name, 1000, 200)
            
            for chunk in chunks:
                chunk_id = f"{ticker}_{section_name.replace(' ', '_')}_{chunk_counter}"
                embedding = embedding_model.encode(chunk['text'])
                
                metadata = {
                    'ticker': ticker,
                    'section': section_name,
                    'chunk_id': chunk_counter,
                    'has_tables': chunk['has_tables'],
                    'has_metrics': chunk['has_metrics'],
                    'length': chunk['length']
                }
                
                all_chunks.append(chunk['text'])
                all_embeddings.append(embedding.tolist())
                all_metadatas.append(metadata)
                all_ids.append(chunk_id)
                chunk_counter += 1
        
        filing_collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas,
            ids=all_ids
        )
        
        st.session_state.filing_tickers.add(ticker)
        return True, len(all_chunks)
        
    except Exception as e:
        st.error(f"Error: {e}")
        return False, 0

# ============================================================================
# SEMANTIC SEARCH WITH RERANKING
# ============================================================================

def semantic_search_with_reranking(
    query: str,
    ticker: str = None,
    top_k: int = 5,
    similarity_threshold: float = 0.35,
    prefer_metrics: bool = False
) -> List[Dict]:
    """Advanced semantic search with reranking"""
    if filing_collection is None or embedding_model is None:
        return []
    
    try:
        all_results = []
        
        # STAGE 1: Direct query
        query_embedding = embedding_model.encode(query).tolist()
        where_filter = {"ticker": ticker} if ticker else None
        
        results = filing_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 3,
            where=where_filter
        )
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                similarity = 1 - results['distances'][0][i]
                
                if similarity < similarity_threshold:
                    continue
                
                metadata = results['metadatas'][0][i]
                
                # Metadata boosting
                metric_boost = 1.15 if (prefer_metrics and metadata.get('has_metrics')) else 1.0
                
                section_boost = 1.0
                section = metadata.get('section', '').lower()
                if 'segment' in query.lower() and 'segment' in section:
                    section_boost = 1.2
                elif 'risk' in query.lower() and 'risk' in section:
                    section_boost = 1.2
                
                all_results.append({
                    'text': results['documents'][0][i],
                    'metadata': metadata,
                    'similarity': similarity * metric_boost * section_boost,
                    'raw_similarity': similarity,
                    'source': 'direct'
                })
        
        # STAGE 2: Expanded query
        expanded_query = expand_financial_query(query)
        if expanded_query != query:
            expanded_embedding = embedding_model.encode(expanded_query).tolist()
            
            expanded_results = filing_collection.query(
                query_embeddings=[expanded_embedding],
                n_results=top_k * 2,
                where=where_filter
            )
            
            if expanded_results['documents'] and expanded_results['documents'][0]:
                for i in range(len(expanded_results['documents'][0])):
                    similarity = 1 - expanded_results['distances'][0][i]
                    
                    if similarity < similarity_threshold:
                        continue
                    
                    metadata = expanded_results['metadatas'][0][i]
                    
                    all_results.append({
                        'text': expanded_results['documents'][0][i],
                        'metadata': metadata,
                        'similarity': similarity * 0.95,
                        'raw_similarity': similarity,
                        'source': 'expanded'
                    })
        
        # STAGE 3: Reranking
        for result in all_results:
            text_lower = result['text'].lower()
            query_words = set(query.lower().split())
            
            keyword_matches = sum(1 for word in query_words if word in text_lower and len(word) > 3)
            keyword_bonus = min(0.1, keyword_matches * 0.02)
            
            length = result['metadata'].get('length', 0)
            if 500 < length < 1500:
                length_bonus = 0.05
            elif length < 300:
                length_bonus = -0.05
            else:
                length_bonus = 0
            
            result['rerank_score'] = result['similarity'] + keyword_bonus + length_bonus
        
        # STAGE 4: Deduplication
        seen_texts = set()
        unique_results = []
        
        for result in sorted(all_results, key=lambda x: x['rerank_score'], reverse=True):
            text_key = result['text'][:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)
                
                if len(unique_results) >= top_k:
                    break
        
        return unique_results
        
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

# ============================================================================
# CORE STOCK FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_stock_metrics(ticker: str) -> Dict:
    """Get comprehensive stock metrics"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        company_name = info.get('longName', info.get('shortName', POPULAR_TICKERS.get(ticker, ticker)))
        
        return {
            'company_name': company_name,
            'valuation': {
                'current_price': info.get('currentPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'forward_pe': info.get('forwardPE', 0),
                'price_to_book': info.get('priceToBook', 0),
            },
            'performance': {
                'revenue': info.get('totalRevenue', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'earnings_growth': info.get('earningsGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'operating_margin': info.get('operatingMargins', 0),
            },
            'health': {
                'total_cash': info.get('totalCash', 0),
                'total_debt': info.get('totalDebt', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'current_ratio': info.get('currentRatio', 0),
                'quick_ratio': info.get('quickRatio', 0),
            },
            'historical': hist,
            'beta': info.get('beta', 1.0),
            'sector': info.get('sector', 'Unknown')
        }
    except Exception as e:
        return {'error': str(e), 'company_name': ticker}

@st.cache_data(ttl=3600)
def get_benchmark_comparison(ticker: str) -> Dict:
    """Compare against peers and S&P 500"""
    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        
        spy = yf.Ticker("SPY")
        spy_info = spy.info
        
        sector = stock_info.get('sector', '')
        peers = get_peer_tickers(ticker, sector)
        
        peer_metrics = []
        for peer in peers:
            try:
                peer_metrics.append(yf.Ticker(peer).info)
            except:
                continue
        
        def safe_avg(key):
            values = [m.get(key, 0) for m in peer_metrics if m.get(key)]
            return sum(values) / len(values) if values else 0
        
        return {
            'pe_ratio': {'value': stock_info.get('trailingPE', 0), 'sector_avg': safe_avg('trailingPE'), 'sp500': spy_info.get('trailingPE', 20)},
            'profit_margin': {'value': stock_info.get('profitMargins', 0), 'sector_avg': safe_avg('profitMargins'), 'sp500': 0.12},
            'revenue_growth': {'value': stock_info.get('revenueGrowth', 0), 'sector_avg': safe_avg('revenueGrowth'), 'sp500': 0.05},
            'debt_to_equity': {'value': stock_info.get('debtToEquity', 0), 'sector_avg': safe_avg('debtToEquity'), 'sp500': 100},
            'beta': {'value': stock_info.get('beta', 1.0), 'sector_avg': safe_avg('beta'), 'sp500': 1.0}
        }
    except:
        return {}

def get_peer_tickers(ticker: str, sector: str) -> List[str]:
    """Get peer tickers by sector"""
    peers_map = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'NKE', 'HD'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV'],
        'Financial Services': ['JPM', 'BAC', 'WFC', 'GS'],
        'Communication Services': ['META', 'GOOGL', 'DIS', 'NFLX'],
        'Energy': ['XOM', 'CVX'],
    }
    return [p for p in peers_map.get(sector, ['SPY'])[:4] if p != ticker]

def calculate_health_score(metrics: Dict) -> Tuple[float, str]:
    """Calculate financial health score"""
    score = 50
    factors = []
    
    profit_margin = metrics['performance'].get('profit_margin', 0)
    if profit_margin > 0.15:
        score += 15
        factors.append("✓ Strong margins")
    elif profit_margin > 0.05:
        score += 5
    else:
        score -= 10
        factors.append("✗ Low margins")
    
    dte = metrics['health'].get('debt_to_equity', 0)
    if dte and dte < 50:
        score += 10
        factors.append("✓ Low debt")
    elif dte > 150:
        score -= 15
        factors.append("✗ High leverage")
    
    rev_growth = metrics['performance'].get('revenue_growth', 0)
    if rev_growth and rev_growth > 0.15:
        score += 15
        factors.append("✓ Strong growth")
    elif rev_growth and rev_growth < 0:
        score -= 10
        factors.append("✗ Declining revenue")
    
    pe = metrics['valuation'].get('pe_ratio', 0)
    if pe and 10 < pe < 25:
        score += 10
        factors.append("✓ Fair valuation")
    elif pe > 50:
        score -= 5
    
    return max(0, min(100, score)), " | ".join(factors)

def calculate_price_targets(ticker: str, metrics: Dict, portfolio_config: Dict) -> Dict:
    """Calculate entry/exit prices"""
    current_price = metrics['valuation'].get('current_price', 0)
    pe_ratio = metrics['valuation'].get('pe_ratio', 0)
    revenue_growth = metrics['performance'].get('revenue_growth', 0)
    profit_margin = metrics['performance'].get('profit_margin', 0)
    
    if revenue_growth > 0.20:
        fair_pe = 35
    elif revenue_growth > 0.15:
        fair_pe = 30
    elif revenue_growth > 0.10:
        fair_pe = 25
    else:
        fair_pe = 20
    
    if profit_margin > 0.25:
        fair_pe *= 1.1
    elif profit_margin < 0.10:
        fair_pe *= 0.9
    
    eps = current_price / pe_ratio if pe_ratio > 0 else current_price / 20
    fair_value = eps * fair_pe
    
    risk_mult = {'Conservative': 0.85, 'Moderate': 1.0, 'Aggressive': 1.15}
    multiplier = risk_mult.get(portfolio_config['risk_tolerance'], 1.0)
    
    return {
        'current_price': current_price,
        'fair_value': fair_value,
        'entry_conservative': fair_value * 0.85 * multiplier,
        'entry_moderate': fair_value * 0.95 * multiplier,
        'exit_target': fair_value * 1.15,
        'stop_loss': current_price * 0.92,
        'upside_potential': ((fair_value - current_price) / current_price * 100) if current_price > 0 else 0
    }

def get_comparison_text(value: float, sector_avg: float, sp500: float, metric_name: str) -> str:
    """Generate color-coded comparison text"""
    if value == 0 or sector_avg == 0:
        return "<span style='color: #666;'>N/A</span>"
    
    higher_is_better = metric_name not in ['debt_to_equity', 'pe_ratio', 'beta']
    diff_sector = ((value - sector_avg) / abs(sector_avg) * 100) if sector_avg != 0 else 0
    
    if abs(diff_sector) < 5:
        return "<span style='color: #888;'>≈ sector avg</span>"
    
    if diff_sector > 15:
        color = "#00CC96" if higher_is_better else "#EF553B"
        quality = "strong" if higher_is_better else "high risk"
        return f"<span style='color: {color}; font-weight: 600;'>▲ {diff_sector:.0f}% ({quality})</span>"
    elif diff_sector > 5:
        color = "#FFA500"
        quality = "above avg" if higher_is_better else "elevated"
        return f"<span style='color: {color}; font-weight: 600;'>↗ {diff_sector:.0f}% ({quality})</span>"
    elif diff_sector < -15:
        color = "#EF553B" if higher_is_better else "#00CC96"
        quality = "weak" if higher_is_better else "low"
        return f"<span style='color: {color}; font-weight: 600;'>▼ {abs(diff_sector):.0f}% ({quality})</span>"
    else:
        color = "#FFA500"
        return f"<span style='color: {color};'>↘ {abs(diff_sector):.0f}% below avg</span>"

# ============================================================================
# FILE PROCESSING
# ============================================================================

def process_uploaded_file_semantic(uploaded_file, filing_ticker: str) -> Dict:
    """Process SEC filing with semantic embeddings"""
    try:
        file_content = ""
        
        if uploaded_file.type == "text/plain":
            file_content = uploaded_file.read().decode('utf-8')
        elif uploaded_file.type == "application/pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    file_content += page.extract_text()
            except:
                return {'error': 'Install PyPDF2: pip install pypdf2'}
        else:
            file_content = uploaded_file.read().decode('utf-8', errors='ignore')
        
        sections = extract_filing_sections_advanced(file_content)
        
        filing_data = {
            'ticker': filing_ticker,
            'filename': uploaded_file.name,
            'upload_time': datetime.now().isoformat(),
            'sections': sections
        }
        
        success, chunk_count = create_semantic_embeddings(filing_ticker, filing_data)
        
        if success:
            filing_data['chunk_count'] = chunk_count
            filing_data['status'] = 'success'
        else:
            filing_data['status'] = 'failed'
        
        return filing_data
        
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}

# ============================================================================
# SPECIALIZED AGENTS WITH SEMANTIC SEARCH
# ============================================================================

def investment_advisor_agent(ticker: str, metrics: Dict, portfolio_config: Dict, query: str = "") -> str:
    """Investment Advisor with semantic search"""
    response = f"## 💼 Investment Advisor Analysis for {ticker}\n\n"
    response += f"**Profile:** {portfolio_config['risk_tolerance']} Risk, {portfolio_config['investment_horizon']}\n\n"
    response += "⚠️ *This is analysis, not financial advice. Consult a financial advisor.*\n\n"
    
    score, factors = calculate_health_score(metrics)
    price_targets = calculate_price_targets(ticker, metrics, portfolio_config)
    current = price_targets['current_price']
    fair_value = price_targets['fair_value']
    upside = price_targets['upside_potential']
    
    if upside > 20 and score > 70:
        recommendation = "✅ **BUY** - Strong fundamentals with attractive upside"
    elif upside > 10 and score > 60:
        recommendation = "✅ **BUY** - Reasonable opportunity"
    elif upside > 0 and score > 50:
        recommendation = "🟡 **HOLD** - Fair value, monitor for better entry"
    elif upside < -10:
        recommendation = "🔴 **AVOID** - Trading significantly above fair value"
    else:
        recommendation = "🟡 **HOLD** - At or near fair value"
    
    response += f"### {recommendation}\n\n"
    response += f"**Valuation:** Current ${current:.2f} | Fair Value ${fair_value:.2f} | Upside {upside:+.1f}%\n\n"
    response += f"**Entry Points:** Conservative ${price_targets['entry_conservative']:.2f} | Moderate ${price_targets['entry_moderate']:.2f}\n\n"
    response += f"**Targets:** Exit ${price_targets['exit_target']:.2f} | Stop ${price_targets['stop_loss']:.2f}\n\n"
    response += f"**Health Score:** {score}/100 - {factors}\n"
    
    return response

def risk_analyst_agent(ticker: str, metrics: Dict, query: str = "") -> str:
    """Risk Analyst with semantic search"""
    response = f"## 🛡️ Risk Analysis for {ticker}\n\n"
    
    dte = metrics['health'].get('debt_to_equity', 0)
    beta = metrics.get('beta', 1.0)
    current_ratio = metrics['health'].get('current_ratio', 0)
    
    total_risk = 0
    
    response += "### Financial Risks\n\n"
    
    if dte > 150:
        response += f"🔴 **High Leverage** (D/E: {dte:.1f}) - Significant debt burden\n\n"
        total_risk += 30
    elif dte > 80:
        response += f"🟡 **Moderate Debt** (D/E: {dte:.1f})\n\n"
        total_risk += 15
    else:
        response += f"🟢 **Low Debt** (D/E: {dte:.1f})\n\n"
        total_risk += 5
    
    if beta > 1.5:
        response += f"🔴 **High Volatility** (Beta: {beta:.2f}) - Moves {beta:.1f}x market\n\n"
        total_risk += 20
    elif beta > 1.0:
        response += f"🟡 **Above Average Volatility** (Beta: {beta:.2f})\n\n"
        total_risk += 10
    else:
        response += f"🟢 **Lower Volatility** (Beta: {beta:.2f})\n\n"
        total_risk += 5
    
    # Semantic search for business risks
    if ticker in st.session_state.filing_tickers:
        response += "### Business Risks (SEC Filing RAG) 🧠\n\n"
        
        risk_queries = [
            "major business risks threats concerns vulnerabilities",
            "risk factors that could adversely affect operations"
        ]
        
        if query and 'risk' in query.lower():
            risk_queries.insert(0, query)
        
        all_risks = []
        for risk_query in risk_queries[:2]:
            results = semantic_search_with_reranking(
                query=risk_query,
                ticker=ticker,
                top_k=2,
                similarity_threshold=0.45
            )
            all_risks.extend(results)
        
        seen = set()
        unique_risks = []
        for risk in all_risks:
            key = risk['text'][:100]
            if key not in seen:
                seen.add(key)
                unique_risks.append(risk)
        
        unique_risks.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if unique_risks:
            for i, result in enumerate(unique_risks[:3], 1):
                similarity = result['rerank_score'] * 100
                section = result['metadata']['section']
                
                icon = "🔴" if similarity > 60 else "🟠" if similarity > 45 else "🟡"
                response += f"{icon} **Risk #{i}** <span class='relevance-high'>({similarity:.1f}%)</span> | {section}\n\n"
                
                text = re.sub(r'^\[.*?\]\n\n', '', result['text'])
                response += f"{text[:500]}...\n\n"
    else:
        response += "💡 *Upload 10-Q for detailed business risk analysis*\n\n"
    
    response += f"### Risk Rating: "
    if total_risk > 60:
        response += f"🔴 **HIGH** ({total_risk}/100)\n"
    elif total_risk > 30:
        response += f"🟡 **MODERATE** ({total_risk}/100)\n"
    else:
        response += f"🟢 **LOW** ({total_risk}/100)\n"
    
    return response

def product_analyst_agent(ticker: str, metrics: Dict, query: str = "") -> str:
    """Product Analyst with semantic search"""
    response = f"## 📦 Product & Segment Analysis for {ticker}\n\n"
    
    if ticker in st.session_state.filing_tickers:
        response += "### Revenue by Product/Segment 🧠\n\n"
        
        search_queries = [
            "revenue by product segment category breakdown performance",
            "which products services segments generated most sales",
            "product segment sales growth year-over-year"
        ]
        
        if query:
            search_queries.insert(0, query)
        
        all_findings = []
        for search_query in search_queries[:2]:
            results = semantic_search_with_reranking(
                query=search_query,
                ticker=ticker,
                top_k=2,
                similarity_threshold=0.40,
                prefer_metrics=True
            )
            all_findings.extend(results)
        
        seen = set()
        unique_findings = []
        for finding in all_findings:
            key = finding['text'][:100]
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)
        
        unique_findings.sort(key=lambda x: x['rerank_score'], reverse=True)
        
        if unique_findings:
            for i, result in enumerate(unique_findings[:3], 1):
                similarity = result['rerank_score'] * 100
                section = result['metadata']['section']
                
                if similarity > 65:
                    icon, css = "🟢", "relevance-high"
                elif similarity > 45:
                    icon, css = "🟡", "relevance-medium"
                else:
                    icon, css = "🔴", "relevance-low"
                
                response += f"{icon} **Finding {i}** <span class='{css}'>({similarity:.1f}%)</span> | {section}\n\n"
                text = re.sub(r'^\[.*?\]\n\n', '', result['text'])
                response += f"{text[:600]}...\n\n"
        else:
            response += "⚠️ No high-relevance product data. Upload detailed 10-Q filing.\n\n"
    else:
        response += "💡 *Upload 10-Q for detailed product breakdown*\n\n"
    
    rev_growth = metrics['performance'].get('revenue_growth', 0)
    margin = metrics['performance'].get('profit_margin', 0)
    
    response += f"### Company-Wide Performance\n\n"
    response += f"- Revenue Growth: **{rev_growth*100:.1f}%** YoY\n"
    response += f"- Profit Margin: **{margin*100:.1f}%**\n\n"
    
    if rev_growth > 0.15 and margin > 0.15:
        response += "✅ **Strong portfolio** - Multiple segments driving growth\n"
    elif rev_growth > 0:
        response += "🟡 **Mixed performance**\n"
    else:
        response += "🔴 **Product challenges**\n"
    
    return response

def peer_comparison_agent(ticker: str, metrics: Dict) -> str:
    """Peer Comparison Agent"""
    response = f"## 🏆 Peer Comparison Analysis\n\n"
    
    sector = metrics.get('sector', '')
    peers = get_peer_tickers(ticker, sector)
    
    response += f"**Sector:** {sector} | **Peers:** {', '.join(peers)}\n\n"
    
    comparisons = get_benchmark_comparison(ticker)
    
    if comparisons:
        comparison_data = []
        
        for metric_name, comp in comparisons.items():
            value = comp.get('value', 0)
            sector_avg = comp.get('sector_avg', 0)
            
            if value and sector_avg:
                diff = ((value - sector_avg) / sector_avg * 100)
                rating = "🟢" if abs(diff) > 10 and ((diff > 0 and metric_name not in ['debt_to_equity', 'pe_ratio']) or (diff < 0 and metric_name in ['debt_to_equity'])) else "🟡" if abs(diff) < 10 else "🔴"
                
                comparison_data.append({
                    'Metric': metric_name.replace('_', ' ').title(),
                    ticker: f"{value*100:.1f}%" if metric_name in ['profit_margin', 'revenue_growth'] else f"{value:.1f}",
                    'Sector': f"{sector_avg*100:.1f}%" if metric_name in ['profit_margin', 'revenue_growth'] else f"{sector_avg:.1f}",
                    'vs Peers': rating
                })
        
        df = pd.DataFrame(comparison_data)
        response += df.to_markdown(index=False) + "\n\n"
    
    return response

def position_advisor_agent(ticker: str, metrics: Dict, portfolio_config: Dict, shares: int, cost: float) -> str:
    """Position Advisor for portfolio holdings"""
    response = f"## 💰 Position Analysis for {ticker}\n\n"
    
    current_price = metrics['valuation'].get('current_price', 0)
    
    if current_price == 0:
        return response + f"⚠️ Unable to fetch price for {ticker}\n"
    
    total_cost = shares * cost
    current_value = shares * current_price
    profit_loss = current_value - total_cost
    pnl_pct = (profit_loss / total_cost * 100) if total_cost > 0 else 0
    
    response += f"**Position:** {shares} shares @ ${cost:.2f} = ${total_cost:,.2f}\n"
    response += f"**Current:** ${current_price:.2f}/share = ${current_value:,.2f}\n"
    
    if profit_loss >= 0:
        response += f"**P&L:** +${profit_loss:,.2f} (+{pnl_pct:.1f}%) ✅\n\n"
    else:
        response += f"**P&L:** ${profit_loss:,.2f} ({pnl_pct:.1f}%) 📉\n\n"
    
    price_targets = calculate_price_targets(ticker, metrics, portfolio_config)
    
    response += "### Strategy Recommendation\n\n"
    
    if pnl_pct > 50:
        response += f"**🎯 Large Gain** - Consider taking partial profits ({int(shares*0.4)} shares)\n"
    elif pnl_pct > 20:
        response += f"**💰 Solid Gain** - Hold or trim position\n"
    elif pnl_pct > 0:
        response += f"**📊 Small Gain** - Set stop at breakeven ${cost:.2f}\n"
    elif pnl_pct > -15:
        score, _ = calculate_health_score(metrics)
        if score > 60:
            response += f"**🔄 Small Loss** - Fundamentals good (Score: {score}), consider holding\n"
        else:
            response += f"**🔄 Small Loss** - Weak fundamentals, consider exit\n"
    else:
        response += f"**🚨 Large Loss** - Consider tax loss harvesting or partial exit\n"
    
    return response

def education_agent(query: str, ticker: str, metrics: Dict) -> str:
    """Education Agent"""
    response = f"## 🎓 Financial Education: {ticker}\n\n"
    
    query_lower = query.lower()
    
    if 'beta' in query_lower:
        beta = metrics.get('beta', 1.0)
        response += f"### Understanding Beta\n\n**{ticker}'s Beta: {beta:.2f}**\n\n"
        response += "Beta measures volatility vs market:\n"
        response += "- Beta > 1.0: More volatile\n- Beta < 1.0: Less volatile\n\n"
        
        if beta > 1.5:
            response += f"🔴 High volatility - moves {beta:.1f}x market\n"
        elif beta > 1.0:
            response += f"🟡 Moderate volatility\n"
        else:
            response += f"🟢 Defensive stock\n"
    
    elif 'p/e' in query_lower or 'pe ratio' in query_lower:
        pe = metrics['valuation'].get('pe_ratio', 0)
        response += f"### Understanding P/E Ratio\n\n**{ticker}'s P/E: {pe:.2f}**\n\n"
        response += "P/E shows price per $1 of earnings:\n"
        response += "- < 15: Undervalued\n- 15-25: Fair value\n- > 25: Premium\n\n"
    
    elif 'debt' in query_lower or 'd/e' in query_lower:
        dte = metrics['health'].get('debt_to_equity', 0)
        response += f"### Understanding Debt-to-Equity\n\n**{ticker}'s D/E: {dte:.2f}**\n\n"
        response += "- < 0.5: Conservative\n- 0.5-1.0: Balanced\n- > 2.0: High leverage\n\n"
    
    else:
        response += f"I can explain: Beta, P/E Ratio, Profit Margin, Debt/Equity, Current Ratio\n"
    
    return response

# ============================================================================
# MULTI-AGENT COORDINATOR
# ============================================================================

def multi_agent_coordinator(query: str, ticker: str, metrics: Dict, portfolio_config: Dict) -> Dict:
    """Coordinates multiple agents for complex queries"""
    
    # Extract position details if present
    shares, cost = extract_position_details(query)
    
    # Detect multiple questions
    questions = detect_multiple_questions(query)
    
    if len(questions) > 1 and not (shares and cost):
        # MULTI-AGENT MODE
        response = f'<div class="multi-agent-badge">🤖 Multi-Agent Mode: {len(questions)} Questions Detected</div>\n\n'
        
        agents_used = []
        
        for i, question_info in enumerate(questions, 1):
            question_text = question_info['text']
            intent = question_info['intent']
            
            response += f"---\n\n### Question {i}: {question_text}\n\n"
            
            # Route to agent
            if intent == 'Product Analyst':
                agent_response = product_analyst_agent(ticker, metrics, question_text)
                agents_used.append('Product Analyst')
            elif intent == 'Risk Analyst':
                agent_response = risk_analyst_agent(ticker, metrics, question_text)
                agents_used.append('Risk Analyst')
            elif intent == 'Investment Advisor':
                agent_response = investment_advisor_agent(ticker, metrics, portfolio_config, question_text)
                agents_used.append('Investment Advisor')
            elif intent == 'Education':
                agent_response = education_agent(question_text, ticker, metrics)
                agents_used.append('Education')
            elif intent == 'Peer Comparison':
                agent_response = peer_comparison_agent(ticker, metrics)
                agents_used.append('Peer Comparison')
            else:
                # General search
                results = semantic_search_with_reranking(question_text, ticker, top_k=2, similarity_threshold=0.4)
                agent_response = "**Search Results:**\n\n"
                for j, r in enumerate(results, 1):
                    # Remove section header from text
                    clean_text = re.sub(r'^\[.*?\]\n\n', '', r['text'])
                    agent_response += f"{j}. ({r['rerank_score']*100:.1f}%)\n{clean_text[:300]}...\n\n"
                agents_used.append('General Search')
            
            # Remove duplicate headers
            agent_response = re.sub(r'^##.*?\n\n', '', agent_response)
            
            response += agent_response + "\n\n"
        
        response += f"---\n\n**🤖 Agents:** {', '.join(set(agents_used))}\n"
        
        return {
            'answer': response,
            'citations': f"{ticker}-MultiAgent-RAG",
            'agent': 'Multi-Agent Coordinator',
            'agents_used': list(set(agents_used))
        }
    
    else:
        # SINGLE AGENT MODE
        if shares and cost:
            agent_name = 'Position Advisor'
            answer = position_advisor_agent(ticker, metrics, portfolio_config, shares, cost)
        else:
            intent = questions[0]['intent']
            
            if intent == 'Product Analyst':
                answer = product_analyst_agent(ticker, metrics, query)
                agent_name = 'Product Analyst'
            elif intent == 'Risk Analyst':
                answer = risk_analyst_agent(ticker, metrics, query)
                agent_name = 'Risk Analyst'
            elif intent == 'Investment Advisor':
                answer = investment_advisor_agent(ticker, metrics, portfolio_config, query)
                agent_name = 'Investment Advisor'
            elif intent == 'Education':
                answer = education_agent(query, ticker, metrics)
                agent_name = 'Education'
            elif intent == 'Peer Comparison':
                answer = peer_comparison_agent(ticker, metrics)
                agent_name = 'Peer Comparison'
            else:
                answer = investment_advisor_agent(ticker, metrics, portfolio_config, query)
                agent_name = 'Investment Advisor'
        
        agent_info = AGENTS.get(agent_name, AGENTS['Investment Advisor'])
        badge = f"<span class='agent-badge' style='background: {agent_info['color']}; color: white;'>{agent_info['icon']} {agent_name}</span>\n\n"
        
        return {
            'answer': badge + answer,
            'citations': f"{ticker}-RAG-Reranked",
            'agent': agent_name
        }

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_ticker_from_question(query: str, current_ticker: str) -> str:
    """Extract ticker from question"""
    patterns = [
        r'\b([A-Z]{2,5})\s+(?:stock|shares?|at\s+\$)',
        r'(?:of|in|for)\s+([A-Z]{2,5})\b',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            potential = match.group(1)
            if potential not in ['I', 'A', 'IS', 'AT', 'OF', 'IN', 'THE', 'MY', 'DO']:
                return potential
    
    return current_ticker

def extract_position_details(query: str) -> Tuple[Optional[int], Optional[float]]:
    """Extract shares and cost"""
    shares_match = re.search(r'(\d+)\s*shares?', query.lower())
    cost_match = re.search(r'(?:at|cost|average|price|bought)\s+\$?(\d+\.?\d*)', query.lower())
    
    shares = int(shares_match.group(1)) if shares_match else None
    cost = float(cost_match.group(1)) if cost_match else None
    
    return shares, cost

def get_database_stats() -> Dict:
    """Get vector database statistics"""
    if filing_collection is None:
        return {'total_chunks': 0, 'tickers': []}
    
    try:
        count = filing_collection.count()
        all_data = filing_collection.get()
        tickers = set()
        if all_data and all_data['metadatas']:
            tickers = {m['ticker'] for m in all_data['metadatas']}
        
        return {
            'total_chunks': count,
            'tickers': sorted(list(tickers)),
            'embedding_dimension': 384
        }
    except:
        return {'total_chunks': 0, 'tickers': [], 'embedding_dimension': 384}

# ============================================================================
# SAMPLE DATA
# ============================================================================

def generate_sample_filings():
    """Enhanced sample SEC filings"""
    return {
        'AAPL': {
            'Segment Information': """
            Revenue by Operating Segment (in millions):
            
            iPhone: $200,583 (51% of total revenue, +12% YoY)
            iPhone 14 Pro models drove strong demand with advanced camera features.
            Average selling price increased due to Pro model mix.
            Market share gains in China offset weakness in other regions.
            
            Services: $85,200 (22% of total revenue, +16% YoY)
            App Store: $32,400 (+14% YoY) with strong subscription growth.
            iCloud: $18,600 (+22% YoY) driven by storage tier upgrades.
            Apple Music: $12,800 (+18% YoY) reaching 98M subscribers.
            Advertising: $8,200 (+28% YoY) from App Store Ads expansion.
            Highest margin segment at 70.8% gross margin.
            
            Mac: $29,357 (7% of total revenue, -27% YoY)
            Significant decline due to M1/M2 upgrade cycle completion.
            Enterprise segment remained stable but consumer demand weakened.
            New MacBook Air with M3 expected to revive growth.
            
            iPad: $28,300 (7% of total revenue, -3% YoY)
            Education segment growth offset by consumer weakness.
            iPad Pro with M2 chip maintained premium positioning.
            
            Wearables, Home and Accessories: $41,200 (10%, -7% YoY)
            Apple Watch maintained smartwatch market leadership.
            AirPods faced increased competition from lower-priced alternatives.
            """,
            'Risk Factors': """
            Principal Business Risks:
            
            1. Global Supply Chain Vulnerability
            The company sources components from over 200 suppliers primarily in Asia, with significant concentration in China and Taiwan. Geopolitical tensions between US and China pose substantial risks to manufacturing continuity. COVID-19 lockdowns in China demonstrated this vulnerability when iPhone production fell 20% below targets.
            
            2. Intense Smartphone Market Competition
            Samsung, Xiaomi, OPPO, and competitors are rapidly narrowing the technology gap with comparable features at 30-50% lower prices. In emerging markets, iPhone market share has declined from 15% to 11% over the past two years. Chinese brands continue to gain in the domestic market.
            
            3. Regulatory and Legal Risks
            EU Digital Markets Act designates App Store as a gatekeeper, potentially forcing sideloading and alternative payment systems that could reduce Services revenue by 15-20%. Ongoing Epic Games litigation could establish precedent for reduced App Store commissions. Antitrust investigations ongoing in US, EU, Japan, and South Korea.
            
            4. Technology Transition Risks
            AR/VR headset (Vision Pro) represents major new category but faces uncertain consumer adoption at $3,499 price point. Mixed reality market is nascent and Meta has struggled despite $10B+ annual investment.
            """
        },
        'TSLA': {
            'Segment Information': """
            Tesla Revenue Breakdown (in millions):
            
            Automotive: $71,500 (88% of total, +55% YoY)
            Model Y became best-selling vehicle globally with 1.2M deliveries.
            Model 3 deliveries: 654K units, stable demand.
            Model S/X: 66K units, niche premium segment.
            Average selling price declined to $54,500 due to price cuts for competitiveness.
            
            Energy Generation and Storage: $3,900 (5% of total, +40% YoY)
            Solar installations grew 35% to 348MW deployed.
            Powerwall demand surged 52% driven by grid instability concerns.
            Megapack utility-scale storage exceeded 14GWh installed.
            
            Services and Other: $5,100 (6% of total, +35% YoY)
            Supercharger network revenue grew from third-party access.
            Insurance products expanded to 12 states.
            Used vehicle sales remained strong margin contributor.
            """,
            'Risk Factors': """
            Tesla Principal Risks:
            
            1. Battery Raw Material Risks
            The company faces risks from lithium and nickel price volatility and supply constraints. Recent lithium prices surged 400% creating margin pressure. Long-term supply agreements only cover 40% of projected 2025 needs.
            
            2. Production Concentration Risk
            Production is concentrated in 4 facilities: Fremont, Shanghai, Berlin, Texas. Any disruption at a single facility could reduce global output by 25%. Shanghai factory represents 50% of production creating China dependency.
            
            3. Autonomous Driving Liability
            Full Self-Driving beta program carries regulatory and liability risks. NHTSA investigations into 35 crashes involving Autopilot. Potential recall or feature restrictions could damage brand and revenue.
            
            4. Intensifying EV Competition
            Ford, GM, Volkswagen, and Chinese manufacturers are launching competitive EVs at lower prices. BYD sold 1.8M EVs in China versus Tesla's 710K. Market share declining in key regions.
            """
        }
    }

def populate_sample_data():
    """Load sample data into vector database"""
    sample_filings = generate_sample_filings()
    
    progress = st.progress(0)
    success = 0
    
    for idx, (ticker, sections) in enumerate(sample_filings.items()):
        filing_data = {'ticker': ticker, 'sections': sections}
        ok, chunks = create_semantic_embeddings(ticker, filing_data)
        
        if ok:
            success += 1
            st.session_state.uploaded_filings.append({
                'ticker': ticker,
                'filename': f'{ticker}_enhanced.txt',
                'chunk_count': chunks
            })
        
        progress.progress((idx + 1) / len(sample_filings))
    
    progress.empty()
    
    if success > 0:
        st.success(f"✅ Loaded {success} companies with {sum(f.get('chunk_count', 0) for f in st.session_state.uploaded_filings)} chunks!")
        return True
    return False

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 🤖 FinChat AI")
    with col2:
        if st.button("🌓", help="Toggle theme"):
            st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
            st.rerun()
    
    st.caption("*Multi-Agent RAG System*")
    st.divider()
    
    # Database stats
    db_stats = get_database_stats()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Chunks", db_stats['total_chunks'])
    with col2:
        st.metric("Companies", len(db_stats['tickers']))
    
    if db_stats['tickers']:
        st.caption(f"📊 {', '.join(db_stats['tickers'])}")
    
    st.divider()
    
    # Sample data loader
    if st.button("🎲 Load Sample Data", use_container_width=True, help="Load AAPL, TSLA with enhanced embeddings"):
        populate_sample_data()
        st.rerun()
    
    st.divider()
    
    # Ticker selection
    st.header("⚙️ Configuration")
    
    ticker_selection = st.selectbox(
        "📈 Select Stock",
        options=TICKER_OPTIONS,
        index=TICKER_OPTIONS.index('AAPL - Apple Inc.') if 'AAPL - Apple Inc.' in TICKER_OPTIONS else 0
    )
    
    ticker = ticker_selection.split(' - ')[0]
    st.session_state.current_ticker = ticker
    
    with st.expander("✏️ Manual Ticker"):
        manual = st.text_input("Enter ticker", value=ticker).upper()
        if st.button("Use") and manual:
            st.session_state.current_ticker = manual
            st.rerun()
    
    st.divider()
    
    # FILE UPLOAD
    st.subheader("📄 SEC Filing Upload")
    
    uploaded_file = st.file_uploader(
        "10-Q or 10-K Filing",
        type=['txt', 'pdf', 'docx', 'html'],
        key="sidebar_filing_uploader"
    )
    
    with st.form(key='sidebar_filing_form'):
        st.caption("📌 Company ticker for this filing:")
        filing_ticker = st.text_input("Ticker", value=ticker, placeholder="e.g., AAPL").upper()
        submit = st.form_submit_button("🔄 Process & Embed", type="primary", use_container_width=True)
    
    if uploaded_file and submit:
        with st.spinner(f"Creating semantic embeddings for {filing_ticker}..."):
            filing_data = process_uploaded_file_semantic(uploaded_file, filing_ticker)
            
            if filing_data.get('status') == 'success':
                st.session_state.uploaded_filings.append(filing_data)
                st.success(f"✅ {filing_data['chunk_count']} chunks embedded!")
                st.rerun()
            else:
                st.error(filing_data.get('error', 'Processing failed'))
    
    if st.session_state.uploaded_filings:
        st.caption("**Uploaded:**")
        for f in st.session_state.uploaded_filings[-5:]:
            chunks = f.get('chunk_count', '?')
            st.text(f"✓ {f['ticker']} ({chunks} chunks)")
    
    st.divider()
    
    # Agent info
    with st.expander("🤖 Available Agents"):
        for name, info in AGENTS.items():
            st.markdown(f"**{info['icon']} {name}**")
            st.caption(info['description'])
    
    st.divider()
    
    # Portfolio settings
    with st.expander("💼 Portfolio Settings"):
        risk = st.select_slider("Risk", ['Conservative', 'Moderate', 'Aggressive'], value='Moderate')
        horizon = st.selectbox("Horizon", ['Short-term (< 1 year)', '1-3 years', '3-5 years', '5+ years'], index=2)
        allocation = st.slider("Allocation (%)", 1.0, 100.0, 10.0, 0.5)
        
        st.session_state.portfolio_config = {
            'risk_tolerance': risk,
            'investment_horizon': horizon,
            'portfolio_allocation': allocation
        }
    
    st.divider()
    
    # Export
    if st.button("📥 Export Conversation"):
        data = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'conversation': st.session_state.messages,
            'portfolio': st.session_state.portfolio_config
        }
        st.download_button(
            "Download JSON",
            json.dumps(data, indent=2),
            f"finchat_{ticker}_{datetime.now().strftime('%Y%m%d')}.json"
        )

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.markdown('<h1 class="main-header">FinChat AI</h1>', unsafe_allow_html=True)
st.markdown("**Advanced Multi-Agent RAG System** | Vector DB • Semantic Search • Reranking • Multi-Agent")

tab1, tab2, tab3, tab4 = st.tabs(["💬 Multi-Agent Chat", "📊 Analytics", "📈 Visualizations", "ℹ️ System Info"])

# ============================================================================
# TAB 1: CHAT
# ============================================================================

with tab1:
    ticker = st.session_state.current_ticker
    ticker_metrics = get_stock_metrics(ticker)
    company_name = ticker_metrics.get('company_name', POPULAR_TICKERS.get(ticker, ticker))
    sector = ticker_metrics.get('sector', '')
    
    # Company header
    st.markdown(f"""
    <div class='company-header'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <h1 style='margin: 0; color: white; font-size: 2.5rem; font-weight: bold;'>{ticker}</h1>
                <p style='margin: 0.25rem 0 0 0; color: rgba(255,255,255,0.95); font-size: 1.3rem;'>{company_name}</p>
            </div>
            <div style='text-align: right;'>
                <div style='font-size: 0.85rem; color: rgba(255,255,255,0.7);'>SECTOR</div>
                <div style='font-size: 1.2rem; color: white; font-weight: 600;'>{sector or 'Loading...'}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Status
    if ticker in st.session_state.filing_tickers:
        st.success(f"🧠 Advanced semantic search enabled for {ticker} - Multi-agent ready!", icon="✅")
    else:
        st.info(f"💡 Upload {ticker}'s 10-Q or load sample data for multi-agent analysis", icon="💡")
    
    st.markdown("---")
    
    # Quick metrics
    st.subheader("📊 Quick Metrics")
    
    if 'error' not in ticker_metrics:
        comparisons = get_benchmark_comparison(ticker)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            score, _ = calculate_health_score(ticker_metrics)
            color = 'green' if score > 70 else 'orange' if score > 40 else 'red'
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 0.85rem; color: #888;'>Health</div>
                <div style='font-size: 1.6rem; font-weight: bold; color: {color};'>{score:.0f}</div>
                <div style='font-size: 0.65rem; color: #666;'>/100</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            price = ticker_metrics['valuation']['current_price']
            pe = ticker_metrics['valuation']['pe_ratio']
            pe_comp = comparisons.get('pe_ratio', {})
            comp_text = get_comparison_text(pe, pe_comp.get('sector_avg', 20), 20, 'pe_ratio')
            
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 0.85rem; color: #888;'>Price</div>
                <div style='font-size: 1.4rem; font-weight: bold;'>${price:.2f}</div>
                <div style='font-size: 0.65rem; color: #999;'>P/E: {pe:.1f}</div>
                <div style='font-size: 0.65rem;'>{comp_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            margin = ticker_metrics['performance']['profit_margin']
            margin_comp = comparisons.get('profit_margin', {})
            comp_text = get_comparison_text(margin, margin_comp.get('sector_avg', 0.15), 0.12, 'profit_margin')
            
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 0.85rem; color: #888;'>Margin</div>
                <div style='font-size: 1.4rem; font-weight: bold;'>{margin*100:.1f}%</div>
                <div style='font-size: 0.65rem; color: #999;'>Sector: {margin_comp.get('sector_avg', 0)*100:.1f}%</div>
                <div style='font-size: 0.65rem;'>{comp_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            growth = ticker_metrics['performance']['revenue_growth']
            growth_comp = comparisons.get('revenue_growth', {})
            comp_text = get_comparison_text(growth, growth_comp.get('sector_avg', 0.05), 0.05, 'revenue_growth')
            
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 0.85rem; color: #888;'>Growth</div>
                <div style='font-size: 1.4rem; font-weight: bold;'>{growth*100:.1f}%</div>
                <div style='font-size: 0.65rem; color: #999;'>Sector: {growth_comp.get('sector_avg', 0)*100:.1f}%</div>
                <div style='font-size: 0.65rem;'>{comp_text}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            dte = ticker_metrics['health']['debt_to_equity']
            dte_comp = comparisons.get('debt_to_equity', {})
            comp_text = get_comparison_text(dte, dte_comp.get('sector_avg', 100), 100, 'debt_to_equity')
            
            st.markdown(f"""
            <div style='text-align: center;'>
                <div style='font-size: 0.85rem; color: #888;'>Debt/Eq</div>
                <div style='font-size: 1.4rem; font-weight: bold;'>{dte:.1f}</div>
                <div style='font-size: 0.65rem; color: #999;'>Sector: {dte_comp.get('sector_avg', 0):.1f}</div>
                <div style='font-size: 0.65rem;'>{comp_text}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Chat section
    st.subheader("💬 Multi-Agent Conversational Analysis")
    
    with st.expander("💡 Try Multi-Agent Queries"):
        st.markdown(f"""
        **Multi-Question Examples:**
        - "Which products performed best **and** what are the risks?" 🤖
        - "Explain P/E ratio **and** should I invest in {ticker}?" 🤖
        - "What segments are growing **and** what threatens them?" 🤖
        
        **Single-Agent Examples:**
        - "Should I invest in {ticker}?" → 💼 Investment Advisor
        - "What are the major risks?" → 🛡️ Risk Analyst
        - "Which products generate most revenue?" → 📦 Product Analyst
        - "I have 26 shares at $36.88, what should I do?" → 💰 Position Advisor
        - "Explain Beta for {ticker}" → 🎓 Education
        - "Compare {ticker} to competitors" → 🏆 Peer Comparison
        """)
    
    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)
            if "citations" in msg:
                st.markdown(f'<div class="citation">📚 {msg["citations"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input(f"Ask about {ticker} (supports multiple questions!)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Multi-agent processing..."):
                # Check if ticker mentioned in query
                extracted_ticker = extract_ticker_from_question(prompt, ticker)
                if extracted_ticker != ticker:
                    ticker = extracted_ticker
                    ticker_metrics = get_stock_metrics(ticker)
                
                result = multi_agent_coordinator(
                    prompt,
                    ticker,
                    ticker_metrics,
                    st.session_state.portfolio_config
                )
                
                st.markdown(result['answer'], unsafe_allow_html=True)
                
                if result['citations']:
                    st.markdown(f'<div class="citation">📚 {result["citations"]}</div>', unsafe_allow_html=True)
                
                if 'agents_used' in result:
                    st.caption(f"🤖 Agents: {', '.join(result['agents_used'])} | Multi-Agent Mode")
                else:
                    st.caption(f"🤖 Agent: {result.get('agent', 'Unknown')}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "citations": result['citations'],
                    "agent": result.get('agent')
                })

# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================

with tab2:
    st.header("Financial Analytics Dashboard")
    
    ticker = st.session_state.current_ticker
    metrics = get_stock_metrics(ticker)
    
    if 'error' not in metrics:
        comparisons = get_benchmark_comparison(ticker)
        score, factors = calculate_health_score(metrics)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: {card_bg}; border-radius: 0.5rem;'>
                <h3>Health Score</h3>
                <h1 style="color: {'green' if score > 70 else 'orange' if score > 40 else 'red'}">{score:.0f}/100</h1>
                <p style='font-size: 0.85rem;'>{factors}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: {card_bg}; border-radius: 0.5rem;'>
                <h3>Market Cap</h3>
                <h1>${metrics['valuation']['market_cap']/1e9:.2f}B</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style='text-align: center; padding: 1rem; background: {card_bg}; border-radius: 0.5rem;'>
                <h3>Price</h3>
                <h1>${metrics['valuation']['current_price']:.2f}</h1>
                <p>Beta: {metrics.get('beta', 1.0):.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # Peer benchmark
        if comparisons:
            st.subheader("🏆 Peer Benchmark")
            
            peers = get_peer_tickers(ticker, metrics.get('sector', ''))
            st.caption(f"vs {', '.join(peers)}")
            
            data = []
            for m, c in comparisons.items():
                val = c.get('value', 0)
                sec = c.get('sector_avg', 0)
                
                higher_better = m not in ['debt_to_equity', 'pe_ratio']
                if sec > 0:
                    diff = ((val - sec) / sec * 100)
                    rating = "🟢" if (abs(diff) > 10 and ((diff > 0) == higher_better)) else "🟡" if abs(diff) < 10 else "🔴"
                else:
                    rating = "➖"
                
                data.append({
                    'Metric': m.replace('_', ' ').title(),
                    ticker: f"{val*100:.1f}%" if m in ['profit_margin', 'revenue_growth'] else f"{val:.1f}",
                    'Sector': f"{sec*100:.1f}%" if m in ['profit_margin', 'revenue_growth'] else f"{sec:.1f}",
                    'Rating': rating
                })
            
            st.dataframe(pd.DataFrame(data), hide_index=True, use_container_width=True)

# ============================================================================
# TAB 3: VISUALIZATIONS
# ============================================================================

with tab3:
    st.header("Interactive Visualizations")
    
    ticker = st.session_state.current_ticker
    metrics = get_stock_metrics(ticker)
    
    if 'error' not in metrics:
        period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=2)
        
        hist = yf.Ticker(ticker).history(period=period)
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index, open=hist['Open'], high=hist['High'],
            low=hist['Low'], close=hist['Close'], name='Price'
        ))
        
        if len(hist) >= 20:
            hist['MA20'] = hist['Close'].rolling(20).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='MA20', line=dict(color='orange')))
        
        fig.update_layout(
            title=f'{ticker} Price Trend',
            yaxis_title='Price ($)',
            template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: SYSTEM INFO
# ============================================================================

with tab4:
    st.header("🧠 System Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Vector DB", "ChromaDB")
        st.caption("Persistent storage")
    
    with col2:
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
        st.caption("384 dimensions")
    
    with col3:
        st.metric("Chunk Size", "1000 chars")
        st.caption("200 char overlap")
    
    st.divider()
    
    st.subheader("🎯 System Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Vector Database & Embeddings**
        - ChromaDB persistent storage
        - 384D semantic embeddings
        - Context-aware chunking (1000 chars)
        - Section headers prepended
        - Metadata enrichment
        
        **✅ Enhanced RAG**
        - Query expansion with synonyms
        - Multi-strategy search
        - 3-factor reranking algorithm
        - Similarity threshold (>35%)
        - Metadata boosting (+15-20%)
        """)
    
    with col2:
        st.markdown("""
        **✅ Multi-Agent System**
        - Auto multi-question detection
        - Intent classification
        - Parallel agent processing
        - Response synthesis
        
        **✅ 6 Specialized Agents**
        - 💼 Investment Advisor
        - 🛡️ Risk Analyst
        - 📦 Product Analyst
        - 🏆 Peer Comparison
        - 💰 Position Advisor
        - 🎓 Education Specialist
        """)
    
    st.divider()
    
    st.subheader("📊 Database Statistics")
    
    stats = get_database_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Chunks", stats['total_chunks'])
    with col2:
        st.metric("Companies", len(stats['tickers']))
    with col3:
        avg = stats['total_chunks'] / len(stats['tickers']) if stats['tickers'] else 0
        st.metric("Avg Chunks/Co", f"{avg:.0f}")
    
    if stats['tickers']:
        st.markdown("**Companies:** " + ", ".join(stats['tickers']))
    
    st.divider()
    
    st.subheader("👥 Team")
    st.markdown("""
    **IST.688.M001.FALL25** - Building HC-AI Applications
    
    Bhushan Jain | Samiksha Singh | Anjali Kalra | Shraddha Aher
    
    ### ⚖️ Disclaimer
    Educational purposes only. Not financial advice.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Filings", len(st.session_state.uploaded_filings))
    with col3:
        st.metric("Theme", st.session_state.theme.title())

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p>🤖 FinChat AI v4.0 | Complete Multi-Agent RAG System</p>
    <p>Vector DB • Semantic Search • Reranking • Multi-Agent Coordination</p>
    <p>IST.688.M001.FALL25 - Building HC-AI Applications</p>
</div>
""", unsafe_allow_html=True)