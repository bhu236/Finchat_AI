import streamlit as st
import pandas as pd
import plotly.express as px

from data_fetchers import fetch_stock_info, fetch_company_metrics, fetch_news_sentiment
from agents import generate_advice, init_chat_memory
from embeddings_db import add_to_vector_db, retrieve_from_vector_db

# ----------------------
# Streamlit Config
# ----------------------
st.set_page_config(page_title="AI Investment Advisor", layout="wide")

# ----------------------
# Sidebar: User Profile
# ----------------------
st.sidebar.header("Your Investment Profile")
risk_tolerance = st.sidebar.selectbox("Risk Tolerance", ["Low", "Medium", "High"])
age_group = st.sidebar.selectbox("Age Group", ["18-30", "31-50", "50+"])
investment_horizon = st.sidebar.selectbox("Investment Horizon", ["Short-term", "Medium-term", "Long-term"])
budget = st.sidebar.number_input("Investment Budget ($)", min_value=100, step=100)
style_preference = st.sidebar.selectbox("Investment Style", ["Conservative", "Balanced", "Aggressive"])
knowledge_level = st.sidebar.selectbox("Financial Knowledge", ["Beginner", "Intermediate", "Expert"])

user_profile = {
    "risk_tolerance": risk_tolerance,
    "age_group": age_group,
    "investment_horizon": investment_horizon,
    "budget": budget,
    "style_preference": style_preference,
    "knowledge_level": knowledge_level
}

# ----------------------
# Initialize Chat Memory
# ----------------------
chat_memory = init_chat_memory()

# ----------------------
# Main Panel: Company Input
# ----------------------
st.title("AI Investment Advisor Chatbot")
ticker = st.text_input("Enter Company Ticker (e.g., AAPL, MSFT)")

if ticker:
    with st.spinner("Fetching company data..."):
        stock_info = fetch_stock_info(ticker)
        financial_metrics = fetch_company_metrics(ticker)
        news_sentiment = fetch_news_sentiment(ticker)

    # ----------------------
    # Stock Metrics + History
    # ----------------------
    st.subheader(f"{ticker} Stock Metrics")
    st.write(stock_info)

    # Plot stock price history
    hist = stock_info.get("history")
    if hist is not None:
        fig = px.line(hist, x="Date", y="Close", title=f"{ticker} Stock Price History")
        st.plotly_chart(fig)

    # Display financial metrics
    st.subheader("Key Financial Metrics (from latest 10-Q)")
    st.write(financial_metrics)

    # Add filings context to FAISS
    add_to_vector_db(financial_metrics.get("filing_text", ""), metadata={"ticker": ticker})

    # ----------------------
    # User Chat / Advice
    # ----------------------
    st.subheader("Ask Questions About This Stock")
    user_question = st.text_input("Your Question:")

    if user_question:
        # Retrieve relevant filings chunks
        context = retrieve_from_vector_db(user_question, top_k=3)

        # Generate advice
        advice = generate_advice(
            user_profile=user_profile,
            company_data={
                "ticker": ticker,
                "filings_context": context,
                "stock_info": stock_info,
                "financial_metrics": financial_metrics,
                "news_sentiment": news_sentiment
            },
            chat_memory=chat_memory,
            user_question=user_question
        )
        st.write(advice)
