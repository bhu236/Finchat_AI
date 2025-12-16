"""
FinChat AI - Streamlit Web Application
Deploy with: streamlit run finchat_app.py

First install dependencies:
    pip install streamlit yfinance plotly pandas pypdf2 python-docx
"""
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List, Tuple
import re
import io

# ============================================================================
# CONFIGURATION - MUST BE FIRST
# ============================================================================

st.set_page_config(
    page_title="FinChat AI - Financial Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INITIALIZE SESSION STATE - BEFORE ANYTHING ELSE
# ============================================================================

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'conversation_memory' not in st.session_state:
    st.session_state.conversation_memory = []

if 'uploaded_filings' not in st.session_state:
    st.session_state.uploaded_filings = []

if 'filing_context' not in st.session_state:
    st.session_state.filing_context = {}

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

# NOW we can safely use theme
theme = st.session_state.theme
bg_color = "#0E1117" if theme == 'dark' else "#FFFFFF"
text_color = "#FAFAFA" if theme == 'dark' else "#262730"
card_bg = "#1E1E1E" if theme == 'dark' else "#F0F2F6"
border_color = "#00CC96"

# Custom CSS with theme support
st.markdown(f"""
<style>
    .main-header {{
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00CC96, #AB63FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    .metric-card {{
        background-color: {card_bg};
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid {border_color};
    }}
    .citation {{
        background-color: {card_bg};
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }}
    .small-metric {{
        font-size: 0.9rem;
    }}
    .metric-comparison {{
        font-size: 0.75rem;
        color: #888;
        font-style: italic;
    }}
    .stAlert {{
        padding: 0.5rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }}
    .company-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TICKER AUTOCOMPLETE DATA
# ============================================================================

POPULAR_TICKERS = {
    # Technology
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'META': 'Meta Platforms Inc. (Facebook)',
    'NVDA': 'NVIDIA Corporation',
    'TSLA': 'Tesla Inc.',
    'AMZN': 'Amazon.com Inc.',
    'NFLX': 'Netflix Inc.',
    'AMD': 'Advanced Micro Devices',
    'INTC': 'Intel Corporation',
    'CRM': 'Salesforce Inc.',
    'ORCL': 'Oracle Corporation',
    'ADBE': 'Adobe Inc.',
    'CSCO': 'Cisco Systems',
    'IBM': 'IBM Corporation',
    
    # Semiconductors
    'MU': 'Micron Technology',  # Correct ticker for Micron
    'QCOM': 'Qualcomm',
    'TXN': 'Texas Instruments',
    'AVGO': 'Broadcom Inc.',
    
    # Finance
    'JPM': 'JPMorgan Chase',
    'BAC': 'Bank of America',
    'WFC': 'Wells Fargo',
    'GS': 'Goldman Sachs',
    'MS': 'Morgan Stanley',
    'V': 'Visa Inc.',
    'MA': 'Mastercard',
    
    # Healthcare
    'JNJ': 'Johnson & Johnson',
    'UNH': 'UnitedHealth Group',
    'PFE': 'Pfizer Inc.',
    'ABBV': 'AbbVie Inc.',
    'TMO': 'Thermo Fisher',
    'LLY': 'Eli Lilly',
    
    # Consumer
    'WMT': 'Walmart',
    'COST': 'Costco',
    'HD': 'Home Depot',
    'NKE': 'Nike Inc.',
    'SBUX': 'Starbucks',
    'MCD': 'McDonald\'s',
    'KO': 'Coca-Cola',
    'PEP': 'PepsiCo',
    
    # Energy
    'XOM': 'Exxon Mobil',
    'CVX': 'Chevron',
    
    # Industrial
    'BA': 'Boeing',
    'CAT': 'Caterpillar',
    'GE': 'General Electric',
}

# Create searchable list
TICKER_OPTIONS = [f"{ticker} - {name}" for ticker, name in sorted(POPULAR_TICKERS.items())]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_stock_metrics(ticker: str) -> Dict:
    """Get comprehensive stock metrics (cached for 1 hour)"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")
        
        return {
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
        return {'error': str(e)}

@st.cache_data(ttl=3600)
def get_benchmark_comparison(ticker: str) -> Dict:
    """Compare stock metrics against S&P 500 and sector peers"""
    try:
        # Get target stock
        stock = yf.Ticker(ticker)
        stock_info = stock.info
        
        # Get S&P 500 (SPY as proxy)
        spy = yf.Ticker("SPY")
        spy_info = spy.info
        
        # Get sector average (simplified - using a few major companies)
        sector = stock_info.get('sector', '')
        peer_tickers = get_peer_tickers(ticker, sector)
        
        peer_metrics = []
        for peer in peer_tickers:
            try:
                peer_stock = yf.Ticker(peer)
                peer_metrics.append(peer_stock.info)
            except:
                continue
        
        # Calculate averages
        def safe_avg(key):
            values = [m.get(key, 0) for m in peer_metrics if m.get(key)]
            return sum(values) / len(values) if values else 0
        
        comparisons = {
            'pe_ratio': {
                'value': stock_info.get('trailingPE', 0),
                'sector_avg': safe_avg('trailingPE'),
                'sp500': spy_info.get('trailingPE', 0)
            },
            'profit_margin': {
                'value': stock_info.get('profitMargins', 0),
                'sector_avg': safe_avg('profitMargins'),
                'sp500': spy_info.get('profitMargins', 0)
            },
            'revenue_growth': {
                'value': stock_info.get('revenueGrowth', 0),
                'sector_avg': safe_avg('revenueGrowth'),
                'sp500': 0.05  # Typical S&P growth
            },
            'debt_to_equity': {
                'value': stock_info.get('debtToEquity', 0),
                'sector_avg': safe_avg('debtToEquity'),
                'sp500': 100  # Typical S&P D/E
            },
            'beta': {
                'value': stock_info.get('beta', 1.0),
                'sector_avg': safe_avg('beta'),
                'sp500': 1.0
            }
        }
        
        return comparisons
        
    except Exception as e:
        return {'error': str(e)}

def get_peer_tickers(ticker: str, sector: str) -> List[str]:
    """Get peer company tickers based on sector"""
    peers_map = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
        'Consumer Cyclical': ['AMZN', 'TSLA', 'NKE', 'SBUX', 'HD'],
        'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
        'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'Communication Services': ['META', 'GOOGL', 'DIS', 'NFLX', 'T'],
        'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
        'Industrials': ['BA', 'CAT', 'GE', 'HON', 'UPS']
    }
    
    peers = peers_map.get(sector, ['SPY'])
    return [p for p in peers if p != ticker][:4]

def get_comparison_text(value: float, sector_avg: float, sp500: float, metric_name: str, is_percentage: bool = False) -> str:
    """Generate color-coded comparison text showing how metric compares to benchmarks"""
    if value == 0 or sector_avg == 0:
        return "<span style='color: #666;'>data unavailable</span>"
    
    # Determine if higher is better for this metric
    higher_is_better = metric_name not in ['debt_to_equity', 'pe_ratio', 'beta']
    
    # Calculate differences
    diff_sector = ((value - sector_avg) / abs(sector_avg) * 100) if sector_avg != 0 else 0
    
    # Determine color and text
    if abs(diff_sector) < 5:
        color = "#888"
        symbol = "≈"
        text = "sector average"
    elif diff_sector > 15:
        color = "#00CC96" if higher_is_better else "#EF553B"  # Green if good, red if bad
        symbol = "▲"
        quality = "strong" if higher_is_better else "high risk"
        text = f"{diff_sector:.0f}% vs sector ({quality})"
    elif diff_sector > 5:
        color = "#FFA500" if higher_is_better else "#FFA500"  # Orange
        symbol = "↗"
        quality = "above avg" if higher_is_better else "elevated"
        text = f"{diff_sector:.0f}% vs sector ({quality})"
    elif diff_sector < -15:
        color = "#EF553B" if higher_is_better else "#00CC96"  # Red if bad, green if good
        symbol = "▼"
        quality = "weak" if higher_is_better else "low"
        text = f"{abs(diff_sector):.0f}% vs sector ({quality})"
    else:
        color = "#FFA500"
        symbol = "↘"
        quality = "below avg" if higher_is_better else "favorable"
        text = f"{abs(diff_sector):.0f}% vs sector ({quality})"
    
    return f"<span style='color: {color}; font-weight: 600;'>{symbol} {text}</span>"

def get_comparison_text(value: float, sector_avg: float, sp500: float, metric_name: str, is_percentage: bool = False) -> str:
    """Generate comparison text for a metric"""
    if value == 0 or sector_avg == 0:
        return "Insufficient data"
    
    # Determine if higher is better
    higher_is_better = metric_name not in ['debt_to_equity', 'pe_ratio', 'beta']
    
    # Compare to sector
    diff_sector = ((value - sector_avg) / sector_avg * 100) if sector_avg != 0 else 0
    diff_sp500 = ((value - sp500) / sp500 * 100) if sp500 != 0 else 0
    
    # Build comparison text
    comparisons = []
    
    if abs(diff_sector) > 5:
        direction = "above" if diff_sector > 0 else "below"
        better_worse = "better" if (diff_sector > 0) == higher_is_better else "worse"
        comparisons.append(f"{abs(diff_sector):.0f}% {direction} sector ({better_worse})")
    else:
        comparisons.append("in line with sector")
    
    if abs(diff_sp500) > 5:
        direction = "above" if diff_sp500 > 0 else "below"
        comparisons.append(f"{abs(diff_sp500):.0f}% {direction} S&P500")
    
    return " | ".join(comparisons) if comparisons else "market average"

def calculate_price_targets(ticker: str, metrics: Dict, portfolio_config: Dict) -> Dict:
    """Calculate entry/exit price points based on analysis"""
    
    current_price = metrics['valuation'].get('current_price', 0)
    pe_ratio = metrics['valuation'].get('pe_ratio', 0)
    revenue_growth = metrics['performance'].get('revenue_growth', 0)
    profit_margin = metrics['performance'].get('profit_margin', 0)
    
    # Calculate fair value P/E based on growth
    if revenue_growth > 0.20:
        fair_pe = 35
    elif revenue_growth > 0.15:
        fair_pe = 30
    elif revenue_growth > 0.10:
        fair_pe = 25
    elif revenue_growth > 0.05:
        fair_pe = 20
    else:
        fair_pe = 15
    
    # Adjust for margin quality
    if profit_margin > 0.25:
        fair_pe *= 1.1
    elif profit_margin < 0.10:
        fair_pe *= 0.9
    
    # Calculate target prices
    eps = current_price / pe_ratio if pe_ratio > 0 else 0
    fair_value = eps * fair_pe
    
    # Risk-adjusted targets based on risk tolerance
    risk_multipliers = {
        'Conservative': 0.85,
        'Moderate': 1.0,
        'Aggressive': 1.15
    }
    
    multiplier = risk_multipliers.get(portfolio_config['risk_tolerance'], 1.0)
    
    # Entry points
    entry_conservative = fair_value * 0.85 * multiplier
    entry_moderate = fair_value * 0.95 * multiplier
    
    # Exit points
    exit_target = fair_value * 1.15
    exit_stop_loss = current_price * 0.92  # 8% stop loss
    
    # Horizon-based holding period
    horizon_map = {
        'Short-term (< 1 year)': 'Consider taking profits at 10-15% gain',
        '1-3 years': 'Hold through volatility, target 15-25% total return',
        '3-5 years': 'Long-term hold, compound growth focus',
        '5+ years': 'Buy and hold, ride through cycles'
    }
    
    return {
        'current_price': current_price,
        'fair_value': fair_value,
        'entry_conservative': entry_conservative,
        'entry_moderate': entry_moderate,
        'exit_target': exit_target,
        'stop_loss': exit_stop_loss,
        'upside_potential': ((fair_value - current_price) / current_price * 100),
        'holding_strategy': horizon_map.get(portfolio_config['investment_horizon'], 'Long-term hold'),
        'risk_adjusted': multiplier != 1.0
    }

def calculate_health_score(metrics: Dict) -> tuple:
    """Calculate financial health score"""
    score = 50
    factors = []
    
    profit_margin = metrics['performance'].get('profit_margin', 0)
    if profit_margin > 0.15:
        score += 15
        factors.append("✓ Strong profit margins")
    elif profit_margin > 0.05:
        score += 5
        factors.append("~ Moderate profit margins")
    else:
        score -= 10
        factors.append("✗ Low profit margins")
    
    dte = metrics['health'].get('debt_to_equity', 0)
    if dte and dte < 50:
        score += 10
        factors.append("✓ Conservative debt levels")
    elif dte > 150:
        score -= 15
        factors.append("✗ High leverage risk")
    
    rev_growth = metrics['performance'].get('revenue_growth', 0)
    if rev_growth and rev_growth > 0.15:
        score += 15
        factors.append("✓ Strong revenue growth")
    elif rev_growth and rev_growth < 0:
        score -= 10
        factors.append("✗ Revenue contraction")
    
    pe = metrics['valuation'].get('pe_ratio', 0)
    if pe and 10 < pe < 25:
        score += 10
        factors.append("✓ Reasonable valuation")
    elif pe > 50:
        score -= 5
        factors.append("~ High valuation multiple")
    
    score = max(0, min(100, score))
    return score, " | ".join(factors)

def process_uploaded_file(uploaded_file, filing_ticker: str) -> Dict:
    """Process uploaded SEC filing file"""
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
            except ImportError:
                return {'error': 'PyPDF2 not installed. Run: pip install pypdf2'}
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                import docx
                doc = docx.Document(uploaded_file)
                file_content = "\n".join([para.text for para in doc.paragraphs])
            except ImportError:
                return {'error': 'python-docx not installed. Run: pip install python-docx'}
        else:
            file_content = uploaded_file.read().decode('utf-8', errors='ignore')
        
        sections = extract_filing_sections(file_content)
        
        filing_data = {
            'ticker': filing_ticker,
            'filename': uploaded_file.name,
            'upload_time': datetime.now().isoformat(),
            'sections': sections,
            'full_text': file_content[:50000]
        }
        
        return filing_data
        
    except Exception as e:
        return {'error': f"Error processing file: {str(e)}"}

def extract_filing_sections(text: str) -> Dict[str, str]:
    """Extract key sections from SEC filing text"""
    sections = {}
    
    mda_patterns = [
        r"MANAGEMENT'?S\s+DISCUSSION\s+AND\s+ANALYSIS.*?(?=QUANTITATIVE\s+AND\s+QUALITATIVE|ITEM\s+3|$)",
        r"MD&A.*?(?=ITEM\s+3|$)",
        r"ITEM\s+2\..*?(?=ITEM\s+3|$)"
    ]
    
    for pattern in mda_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Management Discussion & Analysis'] = match.group(0)[:15000]
            break
    
    risk_patterns = [
        r"RISK\s+FACTORS.*?(?=ITEM\s+2|UNREGISTERED|$)",
        r"ITEM\s+1A\..*?(?=ITEM\s+2|$)"
    ]
    
    for pattern in risk_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Risk Factors'] = match.group(0)[:15000]
            break
    
    financial_patterns = [
        r"CONSOLIDATED\s+(?:BALANCE\s+SHEETS?|STATEMENTS?\s+OF\s+(?:OPERATIONS|INCOME)).*?(?=NOTES\s+TO|ITEM\s+2|$)",
        r"FINANCIAL\s+STATEMENTS.*?(?=NOTES\s+TO|$)"
    ]
    
    for pattern in financial_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            sections['Financial Statements'] = match.group(0)[:15000]
            break
    
    if not sections:
        sections['Full Filing (Excerpt)'] = text[:15000]
    
    return sections

def search_filing_content(query: str, filing_context: Dict[str, str]) -> Dict:
    """Search through filing sections for query-relevant content"""
    
    query_lower = query.lower()
    results = {
        'relevant_sections': [],
        'snippets': [],
        'section_names': []
    }
    
    risk_keywords = ['risk', 'threat', 'concern', 'challenge', 'uncertainty', 'danger']
    revenue_keywords = ['revenue', 'sales', 'income', 'earnings', 'profit', 'performance']
    product_keywords = ['product', 'service', 'segment', 'offering', 'line', 'category']
    outlook_keywords = ['outlook', 'future', 'expect', 'forecast', 'guidance', 'plan']
    debt_keywords = ['debt', 'liability', 'obligation', 'loan', 'borrowing']
    cash_keywords = ['cash', 'liquidity', 'working capital']
    
    is_risk_query = any(kw in query_lower for kw in risk_keywords)
    is_revenue_query = any(kw in query_lower for kw in revenue_keywords)
    is_product_query = any(kw in query_lower for kw in product_keywords)
    is_outlook_query = any(kw in query_lower for kw in outlook_keywords)
    is_debt_query = any(kw in query_lower for kw in debt_keywords)
    is_cash_query = any(kw in query_lower for kw in cash_keywords)
    
    for section_name, content in filing_context.items():
        relevance_score = 0
        
        if is_risk_query and 'risk' in section_name.lower():
            relevance_score = 10
        elif (is_revenue_query or is_product_query) and 'management' in section_name.lower():
            relevance_score = 9
        elif (is_debt_query or is_cash_query) and ('financial' in section_name.lower() or 'balance' in section_name.lower()):
            relevance_score = 8
        
        content_lower = content.lower()
        for word in query_lower.split():
            if len(word) > 3 and word in content_lower:
                relevance_score += 1
        
        if relevance_score > 0:
            results['relevant_sections'].append((section_name, content, relevance_score))
    
    results['relevant_sections'].sort(key=lambda x: x[2], reverse=True)
    
    for section_name, content, score in results['relevant_sections'][:2]:
        results['section_names'].append(section_name)
        
        paragraphs = content.split('\n\n')
        relevant_paras = []
        
        for para in paragraphs:
            para_lower = para.lower()
            match_count = sum(1 for word in query_lower.split() if len(word) > 3 and word in para_lower)
            
            if match_count >= 2 or any(kw in para_lower for kw in risk_keywords + revenue_keywords + product_keywords):
                relevant_paras.append(para)
        
        for para in relevant_paras[:2]:
            if len(para) > 100:
                results['snippets'].append({
                    'section': section_name,
                    'text': para[:800]
                })
    
    return results

def generate_response(query: str, ticker: str, metrics: Dict, filing_context: str = "", portfolio_config: Dict = None) -> Dict:
    """Generate AI response tailored to the specific query"""
    
    if portfolio_config is None:
        portfolio_config = st.session_state.portfolio_config
    
    val = metrics.get('valuation', {})
    perf = metrics.get('performance', {})
    health = metrics.get('health', {})
    
    citations = []
    uncertainty = []
    
    query_lower = query.lower()
    
    # EXPANDED QUESTION DETECTION
    is_educational_question = any(word in query_lower for word in ['what is', 'explain', 'how to read', 'what does', 'define', 'meaning of', 'help me understand'])
    is_position_question = any(word in query_lower for word in ['i have', 'i own', 'my shares', 'my position', 'bought at', 'average cost'])
    is_risk_question = any(word in query_lower for word in ['risk', 'concern', 'threat', 'danger', 'worry'])
    is_product_question = any(word in query_lower for word in ['product', 'service', 'segment', 'performing', 'best', 'worst'])
    is_revenue_question = any(word in query_lower for word in ['revenue', 'sales', 'income', 'earnings'])
    is_outlook_question = any(word in query_lower for word in ['outlook', 'future', 'expect', 'forecast', 'guidance'])
    is_valuation_question = any(word in query_lower for word in ['valuation', 'price', 'expensive', 'cheap', 'worth', 'overvalued', 'undervalued'])
    is_debt_question = any(word in query_lower for word in ['debt', 'leverage', 'liability', 'borrowing'])
    is_investment_question = any(word in query_lower for word in ['invest', 'buy', 'sell', 'hold', 'should i', 'recommend'])
    is_price_target_question = any(word in query_lower for word in ['target', 'entry', 'exit', 'price point'])
    
    filing_insights = ""
    if filing_context:
        search_results = search_filing_content(query, filing_context)
        
        if search_results['snippets']:
            filing_insights = "\n\n**From SEC 10-Q Filing:**\n"
            for snippet in search_results['snippets'][:2]:
                filing_insights += f"\n*[{snippet['section']}]*: {snippet['text'][:400]}...\n"
                citations.append(f"{ticker}-10Q-{snippet['section']}")
        else:
            uncertainty.append("No directly relevant filing content found for this query")
    
    answer = ""
    
    # PRIORITY ORDER: Check specific questions first
    if is_educational_question:
        answer = generate_educational_response(query, ticker, metrics)
    elif is_position_question:
        answer = generate_position_response(query, ticker, metrics, portfolio_config)
    elif is_price_target_question or is_investment_question:
        answer = generate_investment_response(ticker, metrics, filing_insights, portfolio_config)
    elif is_risk_question:
        answer = generate_risk_response(ticker, metrics, filing_insights, filing_context)
    elif is_product_question:
        answer = generate_product_response(ticker, metrics, filing_insights, filing_context)
    elif is_revenue_question:
        answer = generate_revenue_response(ticker, metrics, filing_insights)
    elif is_outlook_question:
        answer = generate_outlook_response(ticker, metrics, filing_insights, filing_context)
    elif is_valuation_question:
        answer = generate_valuation_response(ticker, metrics, filing_insights)
    elif is_debt_question:
        answer = generate_debt_response(ticker, metrics, filing_insights)
    else:
        answer = generate_general_response(ticker, metrics, filing_insights)
    
    citations.append(f"{ticker}-Real-time-yfinance")
    
    uncertainty_terms = ['may', 'could', 'should', 'warrant', 'consider', 'potential', 'possible']
    for term in uncertainty_terms:
        if term in answer.lower():
            uncertainty.append(f"Conditional language: '{term}'")
    
    if not filing_context:
        uncertainty.append("Analysis based on metrics only - upload 10-Q for deeper insights")
    
    return {
        'answer': answer,
        'citations': ' | '.join(set(citations)),
        'uncertainty': list(set(uncertainty))[:3]
    }

def generate_educational_response(query: str, ticker: str, metrics: Dict) -> str:
    """Generate response for educational/informational questions"""
    
    query_lower = query.lower()
    response = ""
    
    # Detect which financial term is being asked about
    if 'beta' in query_lower:
        beta = metrics.get('beta', 1.0)
        response = f"## Understanding Beta for {ticker}\n\n"
        response += f"**{ticker}'s Beta: {beta:.2f}**\n\n"
        response += "**What is Beta?**\n"
        response += "Beta measures a stock's volatility compared to the overall market (S&P 500).\n\n"
        response += "**How to Read Beta:**\n"
        response += "- **Beta = 1.0**: Stock moves exactly with the market\n"
        response += "- **Beta > 1.0**: Stock is MORE volatile than the market\n"
        response += "- **Beta < 1.0**: Stock is LESS volatile than the market\n"
        response += "- **Negative Beta**: Stock moves opposite to the market (rare)\n\n"
        
        response += f"**{ticker}'s Beta Interpretation:**\n"
        if beta > 1.5:
            response += f"🔴 **High Volatility** ({beta:.2f}): {ticker} tends to move {beta:.1f}x as much as the market.\n"
            response += "- When market goes up 10%, this stock might go up ~{:.0f}%\n".format(beta * 10)
            response += "- When market goes down 10%, this stock might go down ~{:.0f}%\n".format(beta * 10)
            response += "- **Risk Profile**: Higher risk, higher potential reward\n"
        elif beta > 1.0:
            response += f"🟡 **Moderate-High Volatility** ({beta:.2f}): {ticker} is somewhat more volatile than the market.\n"
            response += f"- More price swings than average stock\n"
            response += "- Suitable for growth-oriented investors\n"
        elif beta > 0.5:
            response += f"🟢 **Moderate Volatility** ({beta:.2f}): {ticker} moves roughly in line with the market.\n"
            response += "- Balanced risk/reward profile\n"
            response += "- Suitable for most investors\n"
        else:
            response += f"🟢 **Low Volatility** ({beta:.2f}): {ticker} is less volatile than the market.\n"
            response += "- More stable, defensive stock\n"
            response += "- Suitable for conservative investors\n"
        
        response += f"\n**Practical Application:**\n"
        response += f"If you're building a portfolio and want to manage volatility, stocks with lower beta can help reduce overall portfolio risk, while higher beta stocks can amplify potential returns (and losses)."
    
    elif 'p/e' in query_lower or 'pe ratio' in query_lower or 'price to earnings' in query_lower:
        pe = metrics['valuation'].get('pe_ratio', 0)
        response = f"## Understanding P/E Ratio for {ticker}\n\n"
        response += f"**{ticker}'s P/E Ratio: {pe:.2f}**\n\n"
        response += "**What is P/E Ratio?**\n"
        response += "Price-to-Earnings (P/E) ratio shows how much investors are willing to pay for each dollar of earnings.\n\n"
        response += "**Formula:** P/E = Stock Price ÷ Earnings Per Share\n\n"
        response += f"For {ticker}: ${metrics['valuation'].get('current_price', 0):.2f} price ÷ ${metrics['valuation'].get('current_price', 0)/pe:.2f} EPS = {pe:.2f}\n\n"
        
        response += "**How to Read P/E:**\n"
        response += "- **< 15**: Generally considered undervalued or low growth expectations\n"
        response += "- **15-25**: Fair value for stable, profitable companies\n"
        response += "- **25-40**: Premium valuation, high growth expected\n"
        response += "- **> 40**: Very expensive, requires exceptional growth\n\n"
        
        response += f"**{ticker}'s P/E Analysis:**\n"
        if pe < 15:
            response += f"🟢 **Low P/E** ({pe:.2f}): May indicate value opportunity or market concerns about growth.\n"
        elif pe < 25:
            response += f"🟢 **Reasonable P/E** ({pe:.2f}): Fair valuation for a quality company.\n"
        elif pe < 40:
            response += f"🟡 **Elevated P/E** ({pe:.2f}): Market expects strong growth. Verify fundamentals support this.\n"
        else:
            response += f"🔴 **Very High P/E** ({pe:.2f}): Significant growth already priced in. Higher risk if growth disappoints.\n"
        
        response += f"\n**Context:**\n"
        response += f"- Revenue Growth: {perf.get('revenue_growth', 0)*100:.1f}% → {'Supports premium P/E' if perf.get('revenue_growth', 0) > 0.15 else 'Growth may not justify high P/E'}\n"
        response += f"- Profit Margin: {perf.get('profit_margin', 0)*100:.1f}% → {'Quality company' if perf.get('profit_margin', 0) > 0.15 else 'Margin pressure'}\n"
    
    elif 'debt to equity' in query_lower or 'd/e' in query_lower:
        dte = metrics['health'].get('debt_to_equity', 0)
        response = f"## Understanding Debt-to-Equity for {ticker}\n\n"
        response += f"**{ticker}'s D/E Ratio: {dte:.2f}**\n\n"
        response += "**What is Debt-to-Equity?**\n"
        response += "Measures how much debt a company uses to finance its assets relative to shareholder equity.\n\n"
        response += "**Formula:** D/E = Total Debt ÷ Shareholders' Equity\n\n"
        
        response += "**How to Read D/E:**\n"
        response += "- **< 0.5**: Very conservative, low financial risk\n"
        response += "- **0.5-1.0**: Moderate debt, balanced approach\n"
        response += "- **1.0-2.0**: Higher leverage, monitor closely\n"
        response += "- **> 2.0**: High debt burden, significant risk\n\n"
        
        response += f"**{ticker}'s Analysis:**\n"
        if dte < 50:
            response += f"🟢 Low leverage - financially conservative\n"
        elif dte < 100:
            response += f"🟡 Moderate leverage - manageable debt levels\n"
        else:
            response += f"🔴 High leverage - monitor debt servicing ability\n"
        
        response += f"\n- Total Debt: ${metrics['health'].get('total_debt', 0)/1e9:.2f}B\n"
        response += f"- Total Cash: ${metrics['health'].get('total_cash', 0)/1e9:.2f}B\n"
        response += f"- Net Debt: ${(metrics['health'].get('total_debt', 0) - metrics['health'].get('total_cash', 0))/1e9:.2f}B\n"
    
    else:
        # Check if there's a specific financial term in the query
        if 'profit margin' in query_lower or 'margin' in query_lower:
            margin = perf.get('profit_margin', 0)
            response = f"## Understanding Profit Margin for {ticker}\n\n"
            response += f"**{ticker}'s Profit Margin: {margin*100:.1f}%**\n\n"
            response += "**What is Profit Margin?**\n"
            response += "Shows what percentage of revenue becomes profit after all expenses.\n\n"
            response += "**Formula:** (Net Income ÷ Revenue) × 100\n\n"
            response += "**Benchmarks:**\n"
            response += "- **< 5%**: Thin margins, low profitability\n"
            response += "- **5-10%**: Decent profitability\n"
            response += "- **10-20%**: Good profitability\n"
            response += "- **> 20%**: Excellent profitability, pricing power\n\n"
            response += f"**{ticker}'s Analysis:** "
            if margin > 0.20:
                response += f"🟢 Excellent margins indicate strong competitive position and pricing power.\n"
            elif margin > 0.10:
                response += f"🟢 Good profitability.\n"
            elif margin > 0.05:
                response += f"🟡 Decent margins but room for improvement.\n"
            else:
                response += f"🔴 Thin margins - cost pressures or competitive challenges.\n"
        else:
            answer = generate_general_response(ticker, metrics, filing_insights)
            return {
                'answer': answer,
                'citations': ' | '.join(set(citations)),
                'uncertainty': list(set(uncertainty))[:3]
            }
    
    citations.append(f"{ticker}-Real-time-yfinance")
    return {
        'answer': response,
        'citations': ' | '.join(set(citations)),
        'uncertainty': []
    }

def generate_position_response(query: str, ticker: str, metrics: Dict, portfolio_config: Dict) -> str:
    """Generate response for specific portfolio position questions"""
    
    query_lower = query.lower()
    
    # Extract position details from query using regex
    shares_match = re.search(r'(\d+)\s*shares?', query_lower)
    cost_match = re.search(r'(?:at|cost|price|average)\s*(?:of\s*)?\$?(\d+\.?\d*)', query_lower)
    
    shares = int(shares_match.group(1)) if shares_match else None
    cost_basis = float(cost_match.group(1)) if cost_match else None
    
    response = f"## Portfolio Position Analysis for {ticker}\n\n"
    
    if shares and cost_basis:
        current_price = metrics['valuation'].get('current_price', 0)
        
        # Calculate position metrics
        total_cost = shares * cost_basis
        current_value = shares * current_price
        profit_loss = current_value - total_cost
        profit_loss_pct = (profit_loss / total_cost * 100) if total_cost > 0 else 0
        
        response += "**Your Position:**\n"
        response += f"- Shares Owned: {shares}\n"
        response += f"- Cost Basis: ${cost_basis:.2f} per share\n"
        response += f"- Total Investment: ${total_cost:,.2f}\n\n"
        
        response += "**Current Status:**\n"
        response += f"- Current Price: ${current_price:.2f}\n"
        response += f"- Current Value: ${current_value:,.2f}\n"
        response += f"- **Profit/Loss: ${profit_loss:,.2f} ({profit_loss_pct:+.1f}%)**\n\n"
        
        if profit_loss > 0:
            response += f"✅ **You're profitable!** Your position is up ${profit_loss:,.2f} ({profit_loss_pct:.1f}%).\n\n"
        else:
            response += f"📉 **Currently at a loss** of ${abs(profit_loss):,.2f} ({profit_loss_pct:.1f}%).\n\n"
        
        # Calculate price targets for profit
        price_targets = calculate_price_targets(ticker, metrics, portfolio_config)
        
        response += "**To Maximize Profitability:**\n\n"
        
        if profit_loss_pct > 15:
            response += f"**Option 1 - Take Profits:**\n"
            response += f"- Current gain: {profit_loss_pct:.1f}% is solid\n"
            response += f"- Consider taking partial profits (sell 30-50%)\n"
            response += f"- Let remaining position run to target: ${price_targets['exit_target']:.2f}\n"
            response += f"- This locks in ${profit_loss * 0.4:,.2f} while keeping upside potential\n\n"
            
            response += f"**Option 2 - Hold for Higher Target:**\n"
            response += f"- Target exit: ${price_targets['exit_target']:.2f} (+{((price_targets['exit_target']/current_price - 1)*100):.1f}% from here)\n"
            response += f"- Potential gain at target: ${shares * (price_targets['exit_target'] - cost_basis):,.2f}\n"
            response += f"- Set stop loss at ${price_targets['stop_loss']:.2f} to protect gains\n\n"
        
        elif profit_loss_pct > 0:
            response += f"**Strategy - Small Gain Position:**\n"
            response += f"- You're up {profit_loss_pct:.1f}% - modest gain\n"
            response += f"- Breakeven: ${cost_basis:.2f}\n"
            response += f"- Next resistance: ${price_targets['exit_target']:.2f} (potential +${shares * (price_targets['exit_target'] - cost_basis):,.2f})\n"
            response += f"- Consider holding if fundamentals remain strong\n"
            response += f"- Set stop at ${cost_basis * 0.95:.2f} to protect capital\n\n"
        
        elif profit_loss_pct > -10:
            response += f"**Strategy - Small Loss Position:**\n"
            response += f"- Currently down {abs(profit_loss_pct):.1f}%\n"
            response += f"- Breakeven price: ${cost_basis:.2f} (needs {((cost_basis/current_price - 1)*100):.1f}% recovery)\n"
            response += f"- Decision point:\n"
            response += f"  - If fundamentals strong → Hold and avg down if drops to ${price_targets['entry_conservative']:.2f}\n"
            response += f"  - If fundamentals weak → Consider cutting loss at ${price_targets['stop_loss']:.2f}\n\n"
        
        else:
            response += f"**Strategy - Significant Loss Position:**\n"
            response += f"- Down {abs(profit_loss_pct):.1f}% (${abs(profit_loss):,.2f})\n"
            response += f"- Breakeven: ${cost_basis:.2f} (needs {((cost_basis/current_price - 1)*100):.1f}% recovery)\n\n"
            
            response += f"**Options:**\n"
            response += f"1. **Tax Loss Harvesting**: Realize loss for tax benefits, reassess investment thesis\n"
            response += f"2. **Average Down**: Buy more at ${current_price:.2f} to lower cost basis\n"
            response += f"   - Would need {shares * 2:.0f} more shares to avg down to ${(cost_basis + current_price)/2:.2f}\n"
            response += f"3. **Cut Loss**: Exit now to preserve capital for better opportunities\n"
            response += f"4. **Hold and Wait**: If conviction strong, set alert for ${cost_basis:.2f} breakeven\n\n"
        
        # Add fundamental assessment
        score, _ = calculate_health_score(metrics)
        response += f"**Fundamental Check (Health Score: {score}/100):**\n"
        
        if score > 70:
            response += f"✅ Strong fundamentals support holding or adding to position\n"
        elif score > 50:
            response += f"🟡 Mixed fundamentals - monitor closely before adding\n"
        else:
            response += f"🔴 Weak fundamentals - consider if this investment still aligns with your thesis\n"
        
        response += f"\n**Risk-Adjusted Action Plan ({portfolio_config['risk_tolerance']} profile):**\n"
        
        if profit_loss_pct > 10:
            response += f"- Take profits on 30-50% of position\n"
            response += f"- Move stop loss to ${cost_basis * 1.05:.2f} (protect 5% gain)\n"
            response += f"- Let remainder run to ${price_targets['exit_target']:.2f}\n"
        elif profit_loss_pct > 0:
            response += f"- Hold current position\n"
            response += f"- Set stop at breakeven: ${cost_basis:.2f}\n"
            response += f"- Target exit: ${price_targets['exit_target']:.2f}\n"
        else:
            response += f"- Assess conviction in investment thesis\n"
            response += f"- If conviction remains: Set max loss limit at ${price_targets['stop_loss']:.2f}\n"
            response += f"- If conviction weak: Consider exiting to redeploy capital\n"
    
    else:
        response += "**I couldn't extract your position details from the question.**\n\n"
        response += "Please specify:\n"
        response += "- Number of shares (e.g., '26 shares')\n"
        response += "- Your cost basis (e.g., 'bought at $36.88' or 'average cost $36.88')\n\n"
        response += "Example: 'I have 26 shares of AAPL at $36.88 average cost. What should I do?'"
    
    return response

def generate_risk_response(ticker: str, metrics: Dict, filing_insights: str, filing_context: Dict) -> str:
    response = f"## Risk Analysis for {ticker}\n\n"
    
    if filing_insights:
        response += filing_insights + "\n\n"
    
    response += "**Financial Risk Indicators:**\n\n"
    
    dte = metrics['health'].get('debt_to_equity', 0)
    if dte > 150:
        response += f"🔴 **High Leverage Risk**: Debt-to-Equity ratio of {dte:.2f} is elevated.\n\n"
    elif dte > 80:
        response += f"🟡 **Moderate Debt Levels**: Debt-to-Equity at {dte:.2f}.\n\n"
    else:
        response += f"🟢 **Low Debt Risk**: Conservative D/E of {dte:.2f}.\n\n"
    
    current_ratio = metrics['health'].get('current_ratio', 0)
    if current_ratio < 1.0:
        response += f"🔴 **Liquidity Concern**: Current ratio {current_ratio:.2f} below 1.0.\n\n"
    elif current_ratio < 1.5:
        response += f"🟡 **Monitor Liquidity**: Current ratio {current_ratio:.2f}.\n\n"
    else:
        response += f"🟢 **Strong Liquidity**: Current ratio {current_ratio:.2f}.\n\n"
    
    pe = metrics['valuation'].get('pe_ratio', 0)
    if pe > 40:
        response += f"🟡 **Valuation Risk**: P/E of {pe:.2f} suggests high expectations.\n\n"
    
    beta = metrics.get('beta', 1.0)
    if beta > 1.5:
        response += f"🟡 **High Volatility**: Beta of {beta:.2f} indicates higher market sensitivity.\n\n"
    
    if not filing_insights:
        response += "⚠️ *Upload 10-Q filing for detailed management risk disclosures.*\n\n"
    
    response += "**Recommendation:** Assess risks against your risk tolerance."
    
    return response

def generate_product_response(ticker: str, metrics: Dict, filing_insights: str, filing_context: Dict) -> str:
    """Generate detailed response for product/segment performance questions"""
    response = f"## Product & Segment Analysis for {ticker}\n\n"
    
    # DEEPLY SEARCH FILING FOR PRODUCT DATA
    product_data = {}
    
    if filing_context:
        response += "**From SEC 10-Q Filing:**\n\n"
        
        # Search all sections for product/segment mentions
        all_text = ""
        for section_name, content in filing_context.items():
            all_text += content + "\n"
        
        # Extract product performance patterns
        # Look for patterns like: "iPhone revenue increased 12% to $45B"
        product_patterns = [
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:revenue|sales|net sales)\s+(?:increased|decreased|grew|declined)\s+(\d+(?:\.\d+)?%)\s+(?:to\s+)?\$?([\d.]+)\s*([BMK])?',
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+segment\s+revenue\s+(?:was\s+)?\$?([\d.]+)\s*([BMK])?,?\s+(?:an?\s+)?(?:increase|decrease)\s+of\s+(\d+(?:\.\d+)?%)',
            r'([A-Z][a-zA-Z]+)\s+revenue\s+of\s+\$?([\d.]+)\s*([BMK])?',
        ]
        
        found_products = []
        
        for pattern in product_patterns:
            matches = re.finditer(pattern, all_text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                product_name = groups[0] if groups[0] else "Unknown"
                
                # Skip common words that aren't products
                if product_name.lower() in ['total', 'net', 'consolidated', 'item', 'the', 'our', 'company']:
                    continue
                
                found_products.append(match.group(0))
        
        if found_products:
            response += "*Product/Segment Performance Mentions:*\n\n"
            # Remove duplicates and show unique mentions
            unique_products = list(set(found_products))[:10]
            for prod in unique_products:
                response += f"- {prod}\n"
            response += "\n"
        else:
            # Try to find any product mentions even without numbers
            product_keywords = ['iPhone', 'iPad', 'Mac', 'Services', 'Wearables', 'Watch', 'AirPods', 
                              'Cloud', 'Azure', 'Office', 'Windows', 'Surface',
                              'Model S', 'Model 3', 'Model X', 'Model Y', 'Energy',
                              'AWS', 'Prime', 'Advertising']
            
            mentions = []
            for keyword in product_keywords:
                # Look for the keyword in context near revenue/sales/performance
                pattern = rf'({keyword}[^.]*(?:revenue|sales|performance|grew|increased|decreased)[^.]*\.)'
                matches = re.finditer(pattern, all_text, re.IGNORECASE)
                for match in matches:
                    sentence = match.group(1)
                    if len(sentence) > 50 and len(sentence) < 500:
                        mentions.append(sentence)
            
            if mentions:
                response += "*Product Performance Insights:*\n\n"
                for mention in list(set(mentions))[:5]:
                    response += f"• {mention}\n\n"
        
        # Also show the filing insights that were already extracted
        if filing_insights:
            response += filing_insights + "\n"
    else:
        response += "⚠️ **No 10-Q filing uploaded.** To get detailed product segment breakdowns:\n"
        response += "1. Upload the latest 10-Q filing above\n"
        response += "2. The filing's MD&A section contains product category revenue data\n"
        response += "3. I'll extract and analyze specific product performance\n\n"
    
    # Overall performance indicators from metrics
    revenue_growth = metrics['performance'].get('revenue_growth', 0)
    profit_margin = metrics['performance'].get('profit_margin', 0)
    
    response += f"**Overall Company Performance:**\n\n"
    response += f"- **Total Revenue Growth**: {revenue_growth*100:.1f}% YoY "
    
    if revenue_growth > 0.15:
        response += "🟢 Strong growth suggests multiple product lines performing well\n"
    elif revenue_growth > 0.05:
        response += "🟡 Moderate growth - likely mixed performance across segments\n"
    else:
        response += "🔴 Weak/negative growth indicates challenges in core products\n"
    
    response += f"- **Profit Margin**: {profit_margin*100:.1f}% "
    
    if profit_margin > 0.20:
        response += "🟢 Premium margins indicate strong product differentiation and pricing power\n"
    elif profit_margin > 0.10:
        response += "🟡 Healthy margins but facing some competitive pressure\n"
    else:
        response += "🔴 Thin margins suggest commoditization or cost challenges\n"
    
    response += f"\n**Interpretation**: "
    
    if revenue_growth > 0.10 and profit_margin > 0.15:
        response += "Strong growth with maintained high margins typically indicates successful product portfolio with multiple growth drivers. Best-performing products likely have pricing power and expanding market share."
    elif revenue_growth > 0:
        response += "Positive revenue with reasonable margins suggests a stable product mix, though identifying specific top performers requires the detailed segment data from the 10-Q filing."
    else:
        response += "Challenging overall performance suggests some product lines are underperforming. The 10-Q filing's segment data would reveal which specific products/services need attention."
    
    response += "\n\n💡 **Pro Tip**: The 10-Q filing's MD&A section typically breaks down revenue by:\n"
    response += "- Product categories (e.g., iPhone, iPad, Mac, Services)\n"
    response += "- Geographic segments (Americas, Europe, China, etc.)\n"
    response += "- Service vs Product revenue\n\n"
    response += "Upload the filing to see this detailed breakdown with specific growth rates for each segment."
    
    return response

def generate_revenue_response(ticker: str, metrics: Dict, filing_insights: str) -> str:
    response = f"## Revenue Analysis for {ticker}\n\n"
    
    if filing_insights:
        response += filing_insights + "\n\n"
    
    revenue = metrics['performance'].get('revenue', 0)
    revenue_growth = metrics['performance'].get('revenue_growth', 0)
    
    response += f"**Current Revenue:**\n"
    response += f"- Total: ${revenue/1e9:.2f}B (TTM)\n"
    response += f"- Growth: {revenue_growth*100:.1f}% YoY\n\n"
    
    if revenue_growth > 0.15:
        response += "🟢 Strong double-digit growth\n\n"
    elif revenue_growth > 0:
        response += "🟡 Moderate positive growth\n\n"
    else:
        response += "🔴 Revenue decline - investigate causes\n\n"
    
    return response

def generate_outlook_response(ticker: str, metrics: Dict, filing_insights: str, filing_context: Dict) -> str:
    response = f"## Future Outlook for {ticker}\n\n"
    
    if filing_insights:
        response += "**Management Discussion:**\n" + filing_insights + "\n\n"
    
    revenue_growth = metrics['performance'].get('revenue_growth', 0)
    earnings_growth = metrics['performance'].get('earnings_growth', 0)
    
    response += "**Forward Indicators:**\n"
    
    if revenue_growth > 0 and earnings_growth > 0:
        response += f"📈 Positive momentum: Revenue ({revenue_growth*100:.1f}%) and earnings growing\n\n"
    else:
        response += f"⚠️ Headwinds present in growth metrics\n\n"
    
    return response

def generate_valuation_response(ticker: str, metrics: Dict, filing_insights: str) -> str:
    response = f"## Valuation Analysis for {ticker}\n\n"
    
    price = metrics['valuation'].get('current_price', 0)
    pe = metrics['valuation'].get('pe_ratio', 0)
    
    response += f"**Current Valuation:**\n"
    response += f"- Price: ${price:.2f}\n"
    response += f"- P/E: {pe:.2f}\n\n"
    
    if pe < 15:
        response += f"P/E of {pe:.2f} is low - potential value opportunity\n\n"
    elif pe < 25:
        response += f"P/E of {pe:.2f} is reasonable\n\n"
    elif pe < 40:
        response += f"P/E of {pe:.2f} is elevated - growth expectations priced in\n\n"
    else:
        response += f"⚠️ P/E of {pe:.2f} is very high - significant premium\n\n"
    
    return response

def generate_debt_response(ticker: str, metrics: Dict, filing_insights: str) -> str:
    response = f"## Debt Analysis for {ticker}\n\n"
    
    total_debt = metrics['health'].get('total_debt', 0)
    total_cash = metrics['health'].get('total_cash', 0)
    dte = metrics['health'].get('debt_to_equity', 0)
    
    response += f"**Debt Position:**\n"
    response += f"- Total Debt: ${total_debt/1e9:.2f}B\n"
    response += f"- Cash: ${total_cash/1e9:.2f}B\n"
    response += f"- Net Debt: ${(total_debt-total_cash)/1e9:.2f}B\n"
    response += f"- D/E Ratio: {dte:.2f}\n\n"
    
    net_debt = total_debt - total_cash
    if net_debt < 0:
        response += f"🟢 Net cash position - strong flexibility\n\n"
    elif dte < 100:
        response += f"🟡 Moderate leverage\n\n"
    else:
        response += f"🔴 High leverage - key risk factor\n\n"
    
    return response

def generate_investment_response(ticker: str, metrics: Dict, filing_insights: str, portfolio_config: Dict) -> str:
    response = f"## Investment Analysis for {ticker}\n\n"
    
    response += "⚠️ **Disclaimer**: This is analysis, not financial advice.\n\n"
    
    # Calculate price targets
    price_targets = calculate_price_targets(ticker, metrics, portfolio_config)
    
    score, factors = calculate_health_score(metrics)
    
    response += f"**Overall Assessment (Health Score: {score}/100)**\n\n"
    
    # Portfolio context
    response += f"**Your Profile:**\n"
    response += f"- Risk Tolerance: {portfolio_config['risk_tolerance']}\n"
    response += f"- Investment Horizon: {portfolio_config['investment_horizon']}\n"
    response += f"- Target Allocation: {portfolio_config['portfolio_allocation']:.1f}%\n\n"
    
    # Price points
    response += f"**Price Targets (Based on {portfolio_config['risk_tolerance']} profile):**\n\n"
    response += f"- **Current Price**: ${price_targets['current_price']:.2f}\n"
    response += f"- **Fair Value Estimate**: ${price_targets['fair_value']:.2f}\n"
    response += f"- **Conservative Entry**: ${price_targets['entry_conservative']:.2f} (wait for pullback)\n"
    response += f"- **Moderate Entry**: ${price_targets['entry_moderate']:.2f} (reasonable entry)\n"
    response += f"- **Target Exit**: ${price_targets['exit_target']:.2f} (+{((price_targets['exit_target']/price_targets['current_price']-1)*100):.1f}% from current)\n"
    response += f"- **Stop Loss**: ${price_targets['stop_loss']:.2f} (-8% protection)\n\n"
    
    upside = price_targets['upside_potential']
    if upside > 20:
        response += f"💡 **Upside Potential**: {upside:.1f}% to fair value - attractive entry point\n\n"
    elif upside > 0:
        response += f"💡 **Upside Potential**: {upside:.1f}% to fair value - moderate opportunity\n\n"
    else:
        response += f"⚠️ **Trading Above Fair Value**: {abs(upside):.1f}% premium - wait for pullback\n\n"
    
    response += f"**Holding Strategy ({portfolio_config['investment_horizon']}):**\n"
    response += f"{price_targets['holding_strategy']}\n\n"
    
    # Strengths & Concerns
    response += "**Strengths:**\n"
    strengths = [f for f in factors.split(' | ') if '✓' in f]
    for s in strengths:
        response += f"- {s}\n"
    
    response += "\n**Concerns:**\n"
    concerns = [f for f in factors.split(' | ') if '✗' in f]
    for c in concerns:
        response += f"- {c}\n"
    
    response += f"\n**Action Plan:**\n"
    if upside > 15 and score > 60:
        response += f"- Consider building position at ${price_targets['entry_moderate']:.2f} or below\n"
        response += f"- Target allocation: {portfolio_config['portfolio_allocation']:.1f}% of portfolio\n"
        response += f"- Set stop loss at ${price_targets['stop_loss']:.2f}\n"
        response += f"- Take partial profits at ${price_targets['exit_target']:.2f}\n"
    elif score > 50:
        response += f"- Wait for entry near ${price_targets['entry_conservative']:.2f}\n"
        response += f"- Consider smaller allocation given mixed signals\n"
    else:
        response += f"- Multiple concerns present - consider alternatives\n"
        response += f"- If invested, reassess position\n"
    
    if filing_insights:
        response += "\n" + filing_insights
    
    return response

def generate_general_response(ticker: str, metrics: Dict, filing_insights: str) -> str:
    response = f"## Financial Overview for {ticker}\n\n"
    
    if filing_insights:
        response += filing_insights + "\n\n"
    
    score, factors = calculate_health_score(metrics)
    
    response += f"**Health Score: {score}/100**\n\n{factors}\n\n"
    response += "**Key Metrics:**\n"
    response += f"- Price: ${metrics['valuation'].get('current_price', 0):.2f} | P/E: {metrics['valuation'].get('pe_ratio', 0):.2f}\n"
    response += f"- Revenue Growth: {metrics['performance'].get('revenue_growth', 0)*100:.1f}% | Margin: {metrics['performance'].get('profit_margin', 0)*100:.1f}%\n"
    
    return response

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    # Theme toggle at the very top
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 📊 FinChat AI")
        st.markdown("*Human-Centered Financial Analysis*")
    with col2:
        # Theme toggle button
        current_theme = st.session_state.theme
        if st.button("🌓", help="Toggle theme", key="theme_toggle"):
            st.session_state.theme = 'light' if current_theme == 'dark' else 'dark'
            st.rerun()
    
    st.divider()
    
    st.header("⚙️ Configuration")
    
    # Ticker selection with autocomplete
    ticker_selection = st.selectbox(
        "📈 Select Stock",
        options=TICKER_OPTIONS,
        index=TICKER_OPTIONS.index('AAPL - Apple Inc.') if 'AAPL - Apple Inc.' in TICKER_OPTIONS else 0,
        help="Search by ticker or company name"
    )
    
    # Extract ticker from selection
    ticker = ticker_selection.split(' - ')[0] if ' - ' in ticker_selection else ticker_selection
    st.session_state.current_ticker = ticker
    
    # Manual ticker input option
    with st.expander("✏️ Manual Ticker Entry"):
        manual_ticker = st.text_input("Enter ticker manually", value=ticker, help="Enter any ticker not in the list").upper()
        if st.button("Use Manual Ticker") and manual_ticker:
            ticker = manual_ticker
            st.session_state.current_ticker = manual_ticker
            st.success(f"✓ Using {manual_ticker}")
    
    st.divider()
    
    # ===== MOVED: SEC FILING UPLOAD SECTION =====
    st.subheader("📄 SEC Filing Upload")
    
    # File uploader - ONLY ONE IN ENTIRE APP
    uploaded_file = st.file_uploader(
        "Upload 10-Q or 10-K Filing",
        type=['txt', 'pdf', 'docx', 'html', 'htm'],
        help="Upload SEC filing for deep analysis",
        key="sidebar_filing_uploader"  # UNIQUE KEY
    )
    
    # Form for filing ticker (which company is this filing for?)
    with st.form(key='sidebar_filing_form', clear_on_submit=False):  # UNIQUE KEY
        st.caption("📌 Which company is this filing for?")
        filing_ticker = st.text_input(
            "Company Ticker",
            value=ticker,
            help="Example: If uploading Apple's 10-Q, enter AAPL",
            placeholder="e.g., AAPL, MSFT, TSLA",
            key="sidebar_filing_ticker"  # UNIQUE KEY
        ).upper()
        
        submit_button = st.form_submit_button(
            "🔄 Process Filing",
            type="primary",
            use_container_width=True,
            help="Click to extract and analyze filing sections"
        )
    
    # Processing logic
    if uploaded_file is not None and submit_button:
        with st.spinner(f"📊 Processing {uploaded_file.name} for {filing_ticker}..."):
            filing_data = process_uploaded_file(uploaded_file, filing_ticker)
            
            if 'error' not in filing_data:
                st.session_state.uploaded_filings.append(filing_data)
                st.session_state.filing_context[filing_ticker] = filing_data['sections']
                
                st.success(f"✅ Processed for **{filing_ticker}**")
                st.info(f"📋 Extracted {len(filing_data['sections'])} sections")
                
                # Show what was extracted
                section_names = list(filing_data['sections'].keys())
                st.caption(f"Sections: {', '.join(section_names)}")
            else:
                st.error(filing_data['error'])
    
    # Show uploaded filings
    if st.session_state.uploaded_filings:
        st.divider()
        st.caption("**📁 Uploaded Filings:**")
        for idx, filing in enumerate(st.session_state.uploaded_filings):
            st.text(f"✓ {filing['ticker']}: {filing['filename'][:25]}...")
        
        if st.button("🗑️ Clear All", key="clear_filings_btn"):
            st.session_state.uploaded_filings = []
            st.session_state.filing_context = {}
            st.rerun()
    
    st.divider()
    # ===== END FILING UPLOAD SECTION =====
    
    # Portfolio Configuration
    with st.expander("💼 Portfolio Settings", expanded=False):
        st.markdown("**Customize analysis for your profile**")
        
        risk_tolerance = st.select_slider(
            "Risk Tolerance",
            options=['Conservative', 'Moderate', 'Aggressive'],
            value='Moderate',
            help="Affects entry/exit price recommendations"
        )
        
        investment_horizon = st.selectbox(
            "Investment Horizon",
            ['Short-term (< 1 year)', '1-3 years', '3-5 years', '5+ years'],
            index=2,
            help="Your planned holding period"
        )
        
        portfolio_allocation = st.slider(
            "Target Allocation (%)",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=0.5,
            help="What % of portfolio for this position"
        )
        
        # Update session state
        st.session_state.portfolio_config = {
            'risk_tolerance': risk_tolerance,
            'investment_horizon': investment_horizon,
            'portfolio_allocation': portfolio_allocation
        }
        
        st.caption(f"🎯 {risk_tolerance} | {investment_horizon} | {portfolio_allocation:.1f}%")
    
    st.divider()
    
    # Analysis options
    st.subheader("Analysis Options")
    include_realtime = st.checkbox("Real-time Metrics", value=True)
    use_filing_context = st.checkbox("Use Filing Context", value=True,
                                     help="Include uploaded filing data")
    
    st.divider()
    
    # Export
    if st.button("📥 Export Session"):
        export_data = {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'portfolio_config': st.session_state.portfolio_config,
            'conversation': st.session_state.messages,
            'filings': len(st.session_state.uploaded_filings)
        }
        
        st.download_button(
            label="Download JSON",
            data=json.dumps(export_data, indent=2),
            file_name=f"finchat_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

# ============================================================================
# MAIN INTERFACE
# ============================================================================

st.markdown('<h1 class="main-header">FinChat AI</h1>', unsafe_allow_html=True)
st.markdown("**Human-Centered Financial Analysis Assistant** | SEC 10-Q Analysis + Real-Time Insights")

tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📊 Analytics", "📈 Visualizations", "ℹ️ About"])

# ============================================================================
# TAB 1: CHAT INTERFACE
# ============================================================================

with tab1:
    # FILE UPLOAD SECTION
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload SEC 10-Q Filing")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload 10-Q Filing (TXT, PDF, DOCX, HTML)",
            type=['txt', 'pdf', 'docx', 'html', 'htm'],
            help="Upload SEC 10-Q for deeper analysis",
            key="filing_uploader"
        )
    
    with col2:
        # Create a form for the filing ticker to handle Enter key
        with st.form(key='filing_form'):
            filing_ticker = st.text_input("Filing Ticker", value=ticker, key="filing_ticker_input").upper()
            submit_button = st.form_submit_button("🔄 Process", type="primary", use_container_width=True)
    
    if uploaded_file is not None and submit_button:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            filing_data = process_uploaded_file(uploaded_file, filing_ticker)
            
            if 'error' not in filing_data:
                st.session_state.uploaded_filings.append(filing_data)
                st.session_state.filing_context[filing_ticker] = filing_data['sections']
                
                st.success(f"✓ Successfully processed {uploaded_file.name} for {filing_ticker}")
                
                with st.expander("📄 Extracted Sections"):
                    for section_name, content in filing_data['sections'].items():
                        st.markdown(f"**{section_name}**")
                        st.text_area(
                            f"Content preview",
                            value=content[:500] + "..." if len(content) > 500 else content,
                            height=150,
                            key=f"preview_{section_name}",
                            disabled=True
                        )
            else:
                st.error(filing_data['error'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # QUICK METRICS SECTION (Compact version)
    st.subheader("📊 Quick Metrics Overview")
    
    if ticker:
        try:
            metrics = get_stock_metrics(ticker)
            comparisons = get_benchmark_comparison(ticker)
            
            if 'error' not in metrics:
                # Compact 5-column layout with comparisons
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    score, _ = calculate_health_score(metrics)
                    color = 'green' if score > 70 else 'orange' if score > 40 else 'red'
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 0.9rem; color: #888;'>Health Score</div>
                        <div style='font-size: 1.8rem; font-weight: bold; color: {color};'>{score:.0f}</div>
                        <div style='font-size: 0.7rem; color: #666;'>out of 100</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    price = metrics['valuation'].get('current_price', 0)
                    pe = metrics['valuation'].get('pe_ratio', 0)
                    pe_comp = comparisons.get('pe_ratio', {})
                    sector_pe = pe_comp.get('sector_avg', 0)
                    comparison = get_comparison_text(pe, sector_pe, pe_comp.get('sp500', 0), 'pe_ratio')
                    
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 0.9rem; color: #888;'>Price</div>
                        <div style='font-size: 1.5rem; font-weight: bold;'>${price:.2f}</div>
                        <div style='font-size: 0.7rem; color: #666;'>P/E: {pe:.1f}</div>
                        <div style='font-size: 0.65rem; color: #888; font-style: italic;'>{comparison}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    margin = metrics['performance'].get('profit_margin', 0)
                    margin_comp = comparisons.get('profit_margin', {})
                    sector_margin = margin_comp.get('sector_avg', 0)
                    comparison = get_comparison_text(margin, sector_margin, margin_comp.get('sp500', 0), 'profit_margin', True)
                    
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 0.9rem; color: #888;'>Profit Margin</div>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{margin*100:.1f}%</div>
                        <div style='font-size: 0.7rem; color: #666;'>Sector: {sector_margin*100:.1f}%</div>
                        <div style='font-size: 0.65rem; color: #888; font-style: italic;'>{comparison}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    growth = metrics['performance'].get('revenue_growth', 0)
                    growth_comp = comparisons.get('revenue_growth', {})
                    sector_growth = growth_comp.get('sector_avg', 0)
                    comparison = get_comparison_text(growth, sector_growth, growth_comp.get('sp500', 0), 'revenue_growth', True)
                    
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 0.9rem; color: #888;'>Revenue Growth</div>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{growth*100:.1f}%</div>
                        <div style='font-size: 0.7rem; color: #666;'>Sector: {sector_growth*100:.1f}%</div>
                        <div style='font-size: 0.65rem; color: #888; font-style: italic;'>{comparison}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col5:
                    dte = metrics['health'].get('debt_to_equity', 0)
                    dte_comp = comparisons.get('debt_to_equity', {})
                    sector_dte = dte_comp.get('sector_avg', 0)
                    comparison = get_comparison_text(dte, sector_dte, dte_comp.get('sp500', 0), 'debt_to_equity')
                    
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <div style='font-size: 0.9rem; color: #888;'>Debt/Equity</div>
                        <div style='font-size: 1.5rem; font-weight: bold;'>{dte:.1f}</div>
                        <div style='font-size: 0.7rem; color: #666;'>Sector: {sector_dte:.1f}</div>
                        <div style='font-size: 0.65rem; color: #888; font-style: italic;'>{comparison}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error loading metrics: {e}")
    
    st.divider()
    
    # CHAT INTERFACE
    st.subheader("💬 Ask Questions")
    
    with st.expander("💡 Example Queries"):
        st.markdown(f"""
        **Investment Decisions:**
        - "Should I invest in {ticker}?"
        - "What are good entry and exit prices?"
        - "Give me price targets for my risk profile"
        
        **Risk Analysis:**
        - "What are the main risks for {ticker}?"
        - "How risky is this investment?"
        
        **Product Performance:**
        - "Which products are performing best?"
        - "How are different segments doing?"
        
        **Financial Health:**
        - "Analyze {ticker}'s debt situation"
        - "How is the cash position?"
        - "Is revenue growing?"
        """)
    
    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "citations" in message and message["citations"]:
                st.markdown(f'<div class="citation">📚 {message["citations"]}</div>', 
                           unsafe_allow_html=True)
            
            if "uncertainty" in message and message["uncertainty"]:
                with st.expander("⚠️ Uncertainty Flags", expanded=False):
                    for flag in message["uncertainty"]:
                        st.caption(f"• {flag}")
    
    # Chat input
    if prompt := st.chat_input("Ask about financial performance, risks, investment decisions..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing..."):
                metrics = get_stock_metrics(ticker) if include_realtime else {}
                
                filing_ctx = ""
                if use_filing_context and ticker in st.session_state.filing_context:
                    filing_ctx = st.session_state.filing_context[ticker]
                
                result = generate_response(
                    prompt, 
                    ticker, 
                    metrics, 
                    filing_ctx,
                    st.session_state.portfolio_config
                )
                
                st.markdown(result['answer'])
                
                if result['citations']:
                    st.markdown(f'<div class="citation">📚 {result["citations"]}</div>', 
                               unsafe_allow_html=True)
                
                if result['uncertainty']:
                    with st.expander("⚠️ Flags", expanded=False):
                        for flag in result['uncertainty']:
                            st.caption(f"• {flag}")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result['answer'],
                    "citations": result['citations'],
                    "uncertainty": result['uncertainty']
                })

# ============================================================================
# TAB 2: ANALYTICS WITH PEER BENCHMARKING
# ============================================================================

with tab2:
    st.header("Financial Analytics Dashboard")
    
    # Use synced ticker
    ticker = st.session_state.current_ticker
    st.caption(f"Analyzing: **{ticker}** - {POPULAR_TICKERS.get(ticker, 'Custom Symbol')}")
    
    if ticker:
        metrics = get_stock_metrics(ticker)
        
        if 'error' in metrics:
            st.error(f"⚠️ Unable to fetch data for '{ticker}'")
            st.markdown("""
            **Common ticker mistakes:**
            - ❌ MCRN → ✅ MU (Micron Technology)
            - ❌ GOOG → ✅ GOOGL (Alphabet Class A)
            - ❌ FB → ✅ META (Meta Platforms)
            
            💡 Use the sidebar dropdown to select from verified tickers
            """)
        else:
            comparisons = get_benchmark_comparison(ticker)
        
        if 'error' not in metrics:
            score, explanation = calculate_health_score(metrics)
            
            # Top metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Health Score</h3>
                    <h1 style="color: {'green' if score > 70 else 'yellow' if score > 40 else 'red'}">{score:.0f}/100</h1>
                    <p>{explanation}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                mc = metrics["valuation"].get("market_cap", 0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Market Cap</h3>
                    <h1>${mc/1e9:.2f}B</h1>
                    <p>Sector: {metrics.get('sector', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                price = metrics['valuation'].get('current_price', 0)
                beta = metrics.get('beta', 1.0)
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Price</h3>
                    <h1>${price:.2f}</h1>
                    <p>Beta: {beta:.2f} (volatility vs market)</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # PEER BENCHMARKING SECTION
            st.subheader("🏆 Peer Benchmark Comparison")
            
            if 'error' not in comparisons:
                # Get peer tickers
                sector = metrics.get('sector', '')
                peers = get_peer_tickers(ticker, sector)
                
                st.caption(f"Comparing {ticker} against sector peers: {', '.join(peers)}")
                
                # Comparison table
                benchmark_data = []
                
                for metric_name, comp_data in comparisons.items():
                    if metric_name == 'error':
                        continue
                    
                    value = comp_data.get('value', 0)
                    sector_avg = comp_data.get('sector_avg', 0)
                    sp500_avg = comp_data.get('sp500', 0)
                    
                    # Format values
                    if metric_name in ['profit_margin', 'revenue_growth']:
                        val_str = f"{value*100:.1f}%"
                        sector_str = f"{sector_avg*100:.1f}%"
                        sp500_str = f"{sp500_avg*100:.1f}%"
                    else:
                        val_str = f"{value:.2f}"
                        sector_str = f"{sector_avg:.2f}"
                        sp500_str = f"{sp500_avg:.2f}"
                    
                    # Determine rating
                    higher_is_better = metric_name not in ['debt_to_equity', 'pe_ratio', 'beta']
                    
                    if sector_avg > 0:
                        diff_pct = ((value - sector_avg) / sector_avg * 100)
                        
                        if abs(diff_pct) < 10:
                            rating = "🟡 Average"
                        elif (diff_pct > 10 and higher_is_better) or (diff_pct < -10 and not higher_is_better):
                            rating = "🟢 Above Average"
                        else:
                            rating = "🔴 Below Average"
                    else:
                        rating = "➖ N/A"
                    
                    benchmark_data.append({
                        'Metric': metric_name.replace('_', ' ').title(),
                        ticker: val_str,
                        'Sector Avg': sector_str,
                        'S&P 500': sp500_str,
                        'Rating': rating
                    })
                
                benchmark_df = pd.DataFrame(benchmark_data)
                st.dataframe(
                    benchmark_df,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Rating": st.column_config.TextColumn(width="small")
                    }
                )
                
                st.caption("🟢 Above Average | 🟡 Average | 🔴 Below Average compared to sector peers")
            
            st.divider()
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("💰 Valuation")
                valuation_df = pd.DataFrame([
                    {"Metric": k.replace('_', ' ').title(), 
                     "Value": f"{v:.2f}" if isinstance(v, (int, float)) and v != 0 else "N/A"}
                    for k, v in metrics['valuation'].items()
                ])
                st.dataframe(valuation_df, hide_index=True, use_container_width=True)
            
            with col2:
                st.subheader("📈 Performance")
                perf_df = pd.DataFrame([
                    {"Metric": k.replace('_', ' ').title(), 
                     "Value": f"{v*100:.2f}%" if k in ['revenue_growth', 'profit_margin', 'operating_margin', 'earnings_growth'] and v else f"${v/1e9:.2f}B" if k == 'revenue' and v else "N/A"}
                    for k, v in metrics['performance'].items()
                ])
                st.dataframe(perf_df, hide_index=True, use_container_width=True)
            
            with col3:
                st.subheader("🏥 Health")
                health_df = pd.DataFrame([
                    {"Metric": k.replace('_', ' ').title(), 
                     "Value": f"${v/1e9:.2f}B" if k in ['total_cash', 'total_debt'] and v else f"{v:.2f}" if isinstance(v, (int, float)) and v != 0 else "N/A"}
                    for k, v in metrics['health'].items()
                ])
                st.dataframe(health_df, hide_index=True, use_container_width=True)
            
            st.divider()
            
            # Price Targets Section
            st.subheader("🎯 Price Targets & Entry/Exit Points")
            
            price_targets = calculate_price_targets(ticker, metrics, st.session_state.portfolio_config)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price target visualization with ACTUAL DATA
                current = price_targets['current_price']
                conservative_entry = price_targets['entry_conservative']
                moderate_entry = price_targets['entry_moderate']
                target = price_targets['exit_target']
                stop = price_targets['stop_loss']
                fair_value = price_targets['fair_value']
                
                # Create figure with proper data
                fig_targets = go.Figure()
                
                # Add price zones as shapes
                y_min = min(stop, conservative_entry) * 0.95
                y_max = max(target, current) * 1.05
                
                # Conservative entry zone (green)
                fig_targets.add_shape(
                    type="rect",
                    x0=0, x1=1,
                    y0=conservative_entry * 0.98, y1=conservative_entry * 1.02,
                    fillcolor="rgba(0, 255, 0, 0.2)",
                    line=dict(width=0)
                )
                
                # Moderate entry zone (yellow)
                fig_targets.add_shape(
                    type="rect",
                    x0=0, x1=1,
                    y0=moderate_entry * 0.98, y1=moderate_entry * 1.02,
                    fillcolor="rgba(255, 255, 0, 0.2)",
                    line=dict(width=0)
                )
                
                # Add horizontal lines for each price level
                fig_targets.add_hline(
                    y=current, 
                    line_dash="solid", 
                    line_color="white",
                    line_width=3,
                    annotation_text=f"Current: ${current:.2f}",
                    annotation_position="right"
                )
                
                fig_targets.add_hline(
                    y=conservative_entry,
                    line_dash="dot",
                    line_color="green",
                    line_width=2,
                    annotation_text=f"Conservative Entry: ${conservative_entry:.2f}",
                    annotation_position="left"
                )
                
                fig_targets.add_hline(
                    y=moderate_entry,
                    line_dash="dot",
                    line_color="yellow",
                    line_width=2,
                    annotation_text=f"Moderate Entry: ${moderate_entry:.2f}",
                    annotation_position="left"
                )
                
                fig_targets.add_hline(
                    y=target,
                    line_dash="dash",
                    line_color="green",
                    line_width=2,
                    annotation_text=f"Target Exit: ${target:.2f}",
                    annotation_position="right"
                )
                
                fig_targets.add_hline(
                    y=stop,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Stop Loss: ${stop:.2f}",
                    annotation_position="right"
                )
                
                fig_targets.add_hline(
                    y=fair_value,
                    line_dash="dot",
                    line_color="cyan",
                    line_width=1,
                    annotation_text=f"Fair Value: ${fair_value:.2f}",
                    annotation_position="left"
                )
                
                # Update layout with proper range and theme
                fig_targets.update_layout(
                    title=f"{ticker} - Entry/Exit Strategy ({st.session_state.portfolio_config['risk_tolerance']})",
                    yaxis_title="Price ($)",
                    yaxis=dict(range=[y_min, y_max]),
                    xaxis=dict(visible=False),
                    showlegend=False,
                    height=400,
                    template='plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white',
                    margin=dict(l=100, r=100, t=50, b=50)
                )
                
                st.plotly_chart(fig_targets, use_container_width=True)
            
            with col2:
                st.markdown("### Target Prices")
                
                st.metric("Current Price", f"${current:.2f}")
                st.metric("Fair Value", f"${price_targets['fair_value']:.2f}",
                         delta=f"{price_targets['upside_potential']:.1f}%")
                
                st.divider()
                
                st.markdown("**Entry Points:**")
                st.success(f"Conservative: ${conservative_entry:.2f}")
                st.info(f"Moderate: ${moderate_entry:.2f}")
                
                st.markdown("**Exit Strategy:**")
                st.success(f"Target: ${target:.2f}")
                st.error(f"Stop Loss: ${stop:.2f}")
                
                st.caption(f"Based on {st.session_state.portfolio_config['risk_tolerance']} profile")
            
            st.divider()
            
            # Filing sections
            if ticker in st.session_state.filing_context:
                st.subheader("📋 SEC Filing Sections")
                
                sections = st.session_state.filing_context[ticker]
                
                for section_name, content in sections.items():
                    with st.expander(f"📄 {section_name}"):
                        st.text_area(
                            "Content",
                            value=content[:2000] + "..." if len(content) > 2000 else content,
                            height=300,
                            key=f"analytics_section_{section_name}",
                            disabled=True
                        )
                        st.caption(f"Words: {len(content.split()):,}")

# ============================================================================
# TAB 3: VISUALIZATIONS
# ============================================================================

with tab3:
    st.header("Interactive Visualizations")
    
    # Use synced ticker and theme
    ticker = st.session_state.current_ticker
    plot_template = 'plotly_dark' if st.session_state.theme == 'dark' else 'plotly_white'
    
    if ticker:
        metrics = get_stock_metrics(ticker)
        
        if 'error' not in metrics:
            col1, col2 = st.columns([3, 1])
            with col2:
                period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2)
            
            hist = yf.Ticker(ticker).history(period=period)
            
            st.subheader("📈 Price Trend")
            
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Price'
            ))
            
            if len(hist) >= 20:
                hist['MA20'] = hist['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA20'], name='MA20',
                                        line=dict(color='orange', width=1.5)))
            
            if len(hist) >= 50:
                hist['MA50'] = hist['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=hist['MA50'], name='MA50',
                                        line=dict(color='red', width=1.5)))
            
            fig.update_layout(
                title=f'{ticker} - Price Analysis',
                yaxis_title='Price ($)',
                template=plot_template,
                height=500,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📊 Performance vs Sector")
            
            categories = ['Revenue Growth', 'Profit Margin', 'Operating Margin']
            values = [
                metrics['performance'].get('revenue_growth', 0) * 100,
                metrics['performance'].get('profit_margin', 0) * 100,
                metrics['performance'].get('operating_margin', 0) * 100
            ]
            
            fig2 = go.Figure(go.Bar(
                x=categories,
                y=values,
                marker_color=['#00CC96' if v > 0 else '#EF553B' for v in values],
                text=[f'{v:.1f}%' for v in values],
                textposition='auto',
            ))
            
            fig2.update_layout(
                title=f'{ticker} - Performance Metrics',
                yaxis_title='%',
                template=plot_template,
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab4:
    st.header("About FinChat AI")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    **FinChat AI** - Human-Centered Financial Analysis Assistant
    
    **Features:**
    - 📊 Real-time metrics with peer benchmarking
    - 📄 SEC 10-Q/10-K filing analysis
    - 🎯 Personalized price targets
    - 💼 Portfolio-aware recommendations
    - 🔍 Transparent citations
    - ⚠️ Uncertainty flagging
    
    ### 📤 How to Upload SEC Filings
    
    **Step 1: Get the Filing**
    - Visit [SEC EDGAR](https://www.sec.gov/edgar/searchedgar/companysearch.html)
    - Search for your company (e.g., "Apple Inc")
    - Find latest 10-Q filing
    - Download as TXT, PDF, or HTML
    
    **Step 2: Upload in Sidebar**
    - **Sidebar** → "📄 SEC Filing Upload"
    - Click **"Browse files"** or drag-and-drop
    - **Company Ticker field**: Enter the ticker symbol (e.g., AAPL for Apple)
      - *Why?* The PDF filename doesn't tell us which company it is
      - *This tags the filing* so we know it's Apple's data
    - Click **"🔄 Process Filing"** or press Enter
    - Wait for "✅ Processed" confirmation
    
    **Step 3: Ask Detailed Questions**
    Now you can ask:
    - "Which products are performing best?" ← Uses actual filing data
    - "What risks did management disclose?" ← Extracts risk section
    - "How does management view the outlook?" ← Uses MD&A section
    
    **What "Company Ticker" Means:**
    ```
    Example: You upload "quarterly-report-q3-2024.pdf"
    
    Question: Which company is this for?
    Answer: Enter "AAPL" in Company Ticker field
    
    Result: System knows this is Apple's filing
            Can match with Apple's metrics
            Uses it when you ask about AAPL
    ```
    
    ### 👥 Development Team
    
    **IST.688.M001.FALL25 - Building HC-AI Apps**
    
    - Bhushan Jain
    - Samiksha Singh
    - Anjali Kalra
    - Shraddha Aher
    
    ### 🛡️ Responsible AI Features
    
    ✓ **Citation Transparency**: All claims cite sources
    
    ✓ **Uncertainty Flagging**: Detects ambiguous language
    
    ✓ **Bias Detection**: Monitors overly optimistic/pessimistic statements
    
    ✓ **Peer Benchmarking**: Shows relative performance vs competitors
    
    ✓ **Portfolio Personalization**: Risk-adjusted recommendations
    
    ✓ **Source Traceability**: Filing excerpts linked to sections
    
    ### ⚖️ Disclaimer
    
    This tool provides AI-generated analysis for **educational and research purposes only**. 
    It does not constitute financial advice. Always consult qualified financial advisors 
    before making investment decisions. Past performance does not guarantee future results.
    
    **Data Sources:**
    - SEC EDGAR filings (user-uploaded)
    - Yahoo Finance API (real-time metrics)
    - Peer comparison data (sector averages)
    
    **Privacy:**
    - No data stored on external servers
    - Filings processed in-session only
    - Conversation history cleared on refresh
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Filings", len(st.session_state.uploaded_filings))
    with col3:
        st.metric("Mode", "Demo")

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85rem;'>
    <p>FinChat AI v1.0 | Built with LLaMA 3, LangChain & Streamlit</p>
    <p>IST.688.M001.FALL25 - Building Human-Centered AI Applications</p>
</div>
""", unsafe_allow_html=True)