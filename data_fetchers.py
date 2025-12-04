import yfinance as yf
import pandas as pd
from sec_edgar_downloader import Downloader
from utils import extract_financial_ratios

def fetch_stock_info(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    latest = stock.info
    return {
        "Current Price": latest.get("currentPrice"),
        "Market Cap": latest.get("marketCap"),
        "PE Ratio": latest.get("trailingPE"),
        "52 Week High": latest.get("fiftyTwoWeekHigh"),
        "52 Week Low": latest.get("fiftyTwoWeekLow"),
        "history": hist.reset_index()
    }

def fetch_company_metrics(ticker):
    dl = Downloader("filings")
    dl.get("10-Q", ticker, amount=1)
    # Placeholder: real implementation parses text
    filing_text = f"Parsed 10-Q text for {ticker}"
    metrics = extract_financial_ratios(filing_text)
    metrics["filing_text"] = filing_text
    return metrics

def fetch_news_sentiment(ticker):
    # Placeholder: integrate real news API + sentiment analysis
    return "Neutral sentiment from latest news."
