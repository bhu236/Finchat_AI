# finchat_app_fixed.py
import os
import io
import time
import json
import sqlite3
import hashlib
import requests
import traceback
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tqdm import tqdm

# Try to import new OpenAI SDK
try:
    from openai import OpenAI
    OPENAI_SDK_AVAILABLE = True
except Exception:
    OpenAI = None
    OPENAI_SDK_AVAILABLE = False

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="FinChat AI — Fixed", layout="wide", page_icon="💹")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"  # change if unavailable
DB_PATH = "finchat_embeddings.db"
VECTOR_TABLE = "embeddings"

# SEC requires a descriptive user agent: Company Name, App Name, Contact email
USER_AGENT = {
    "User-Agent": "FinChatAI/1.0 (FinChat AI for academic use; contact: your-email@example.com)"
}

# -------------------------
# DB helpers
# -------------------------
def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {VECTOR_TABLE} (
            id TEXT PRIMARY KEY,
            company TEXT,
            filing_url TEXT,
            section TEXT,
            chunk_index INTEGER,
            content TEXT,
            embedding BLOB,
            created_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def _to_bytes(array: np.ndarray) -> bytes:
    return array.astype(np.float32).tobytes()


def _from_bytes(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def upsert_embedding(conn: sqlite3.Connection, meta: dict, emb: np.ndarray):
    cur = conn.cursor()
    row_id = meta["id"]
    cur.execute(
        f"INSERT OR REPLACE INTO {VECTOR_TABLE} (id, company, filing_url, section, chunk_index, content, embedding, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            row_id,
            meta.get("company"),
            meta.get("filing_url"),
            meta.get("section"),
            meta.get("chunk_index"),
            meta.get("content"),
            _to_bytes(emb),
            datetime.utcnow().isoformat(),
        ),
    )
    conn.commit()


def fetch_all_embeddings(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.execute(f"SELECT id, company, filing_url, section, chunk_index, content, embedding FROM {VECTOR_TABLE}")
    rows = cur.fetchall()
    results = []
    for r in rows:
        try:
            emb = _from_bytes(r[6])
        except Exception:
            emb = np.array([], dtype=np.float32)
        results.append((r[0], r[1], r[2], r[3], r[4], r[5], emb))
    return results


# -------------------------
# EDGAR / CIK helpers
# -------------------------
def lookup_cik(company_or_cik: str) -> Optional[str]:
    if not company_or_cik:
        return None
    s = company_or_cik.strip()
    if s.isdigit():
        return s.zfill(10)
    # try official ticker registry
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        r = requests.get(url, headers=USER_AGENT, timeout=12)
        r.raise_for_status()
        catalog = r.json()
        s_lower = s.lower()
        # exact ticker or title match
        for v in catalog.values():
            title = v.get("title", "").strip().lower()
            ticker = v.get("ticker", "").strip().lower()
            cik = str(v.get("cik_str", "")).zfill(10)
            if s_lower == ticker or s_lower == title:
                return cik
        # substring fallback
        for v in catalog.values():
            title = v.get("title", "").strip().lower()
            cik = str(v.get("cik_str", "")).zfill(10)
            if s_lower in title:
                return cik
    except Exception:
        return None
    return None


def fetch_latest_10q_urls(cik: str, count: int = 2) -> List[str]:
    urls = []
    if not cik:
        return urls
    padded = cik.zfill(10)
    submissions_url = f"https://data.sec.gov/submissions/CIK{padded}.json"
    try:
        r = requests.get(submissions_url, headers=USER_AGENT, timeout=12)
        if r.status_code == 200:
            j = r.json()
            forms = j.get("filings", {}).get("recent", {}).get("form", [])
            accs = j.get("filings", {}).get("recent", {}).get("accessionNumber", [])
            for form, acc in zip(forms, accs):
                if isinstance(form, str) and form.lower().startswith("10-q"):
                    acc_no = acc.replace("-", "")
                    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/"
                    txt = base + acc_no + ".txt"
                    urls.append(txt)
                    if len(urls) >= count:
                        return urls
    except Exception:
        pass

    # fallback to browse-edgar search
    try:
        browse = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={int(cik)}&type=10-q&count=40"
        r = requests.get(browse, headers=USER_AGENT, timeout=12)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        table = soup.find("table", class_="tableFile2")
        if table:
            rows = table.find_all("tr")
            for tr in rows:
                a = tr.find("a", href=True)
                if a:
                    href = a["href"]
                    if href.startswith("/"):
                        href = "https://www.sec.gov" + href
                    urls.append(href)
                    if len(urls) >= count:
                        break
    except Exception:
        pass

    # dedupe
    unique = []
    for u in urls:
        if u not in unique:
            unique.append(u)
    return unique[:count]


def download_filing_text(url: str) -> Optional[str]:
    try:
        r = requests.get(url, headers=USER_AGENT, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception:
        # try alternative: if url is index.htm, try fetch page and find .txt link
        try:
            r2 = requests.get(url, headers=USER_AGENT, timeout=15)
            if r2.status_code == 200:
                s2 = BeautifulSoup(r2.text, "lxml")
                txt_link = s2.find("a", href=True, string=lambda t: t and ".txt" in t.lower())
                if txt_link:
                    href2 = txt_link["href"]
                    if href2.startswith("/"):
                        href2 = "https://www.sec.gov" + href2
                    r3 = requests.get(href2, headers=USER_AGENT, timeout=20)
                    r3.raise_for_status()
                    return r3.text
        except Exception:
            return None
    return None


def extract_html_sections_from_filing(raw_text: str) -> Dict[str, str]:
    soup = BeautifulSoup(raw_text, "lxml")
    full_text = soup.get_text(separator="\n")
    out = {"full_text": full_text, "item_1a": "", "tables": ""}
    lowered = full_text.lower()

    # look for item 1a patterns
    candidates = ["item 1a", "item 1a. risk factors", "item 1a - risk factors", "item 1a: risk factors"]
    idx = -1
    for c in candidates:
        idx = lowered.find(c)
        if idx != -1:
            break
    if idx != -1:
        # find next "item" heading
        import re
        m = re.search(r"\nitem\s+\d+\w*\b", full_text[idx + 1 :])
        if m:
            end = idx + 1 + m.start()
        else:
            end = min(len(full_text), idx + 20000)
        out["item_1a"] = full_text[idx:end].strip()

    # tables extraction
    tables = soup.find_all("table")
    table_texts = []
    for t in tables:
        try:
            table_texts.append(t.get_text(separator=" | ").strip())
        except Exception:
            continue
    out["tables"] = "\n\n".join(table_texts)
    return out


def simple_chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks


# -------------------------
# OpenAI wrappers (new SDK)
# -------------------------
def get_openai_key_from_input(api_key: str):
    if api_key:
        return api_key
    return os.getenv("OPENAI_API_KEY", "")


def make_openai_client(api_key: str):
    if not api_key:
        return None
    if not OPENAI_SDK_AVAILABLE:
        return None
    return OpenAI(api_key=api_key)


def openai_embed(client: OpenAI, texts: List[str]) -> List[np.ndarray]:
    if client is None:
        raise RuntimeError("OpenAI client not configured")
    embeds = []
    # batch calls
    for i in range(0, len(texts), 10):
        batch = texts[i : i + 10]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        for d in resp.data:
            vec = np.array(d.embedding, dtype=np.float32)
            embeds.append(vec)
    return embeds


def openai_chat(client: OpenAI, system_prompt: str, user_prompt: str, history: List[Dict] = None) -> str:
    if client is None:
        return "[LLM offline] OpenAI client not configured."
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history:
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user_prompt})
    try:
        resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, temperature=0.1)
        return resp.choices[0].message.content
    except Exception as e:
        return f"[LLM Error] {e}"


# -------------------------
# Vector search
# -------------------------
def build_index(conn: sqlite3.Connection):
    rows = fetch_all_embeddings(conn)
    if not rows:
        return None
    ids, comps, urls, secs, idxs, contents, embs = zip(*rows)
    X = np.vstack([e for e in embs])
    meta = [{"id": i, "company": c, "filing_url": u, "section": s, "chunk_index": ci, "content": cont} for i, c, u, s, ci, cont in zip(ids, comps, urls, secs, idxs, contents)]
    return {"X": X, "meta": meta}


def vector_search(conn: sqlite3.Connection, query: str, client: Optional[OpenAI], top_k: int = 4):
    idx = build_index(conn)
    if idx is None:
        return []
    if client is None:
        raise RuntimeError("OpenAI client required for query embeddings")
    qrep = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    q_emb = np.array(qrep.data[0].embedding, dtype=np.float32)
    X = idx["X"]
    sims = (X @ q_emb) / (np.linalg.norm(X, axis=1) * np.linalg.norm(q_emb) + 1e-10)
    top_idx = np.argsort(-sims)[:top_k]
    results = []
    for i in top_idx:
        m = idx["meta"][i]
        results.append({"score": float(sims[i]), "content": m["content"], "metadata": m})
    return results


# -------------------------
# Agent prompts & runners
# -------------------------
FINANCIAL_SYSTEM = "You are a Financial Analyst. Use provided metrics and RAG context. Cite chunk ids in brackets."
RISK_SYSTEM = "You are a Risk Insights Agent. Use Item 1A context and cite chunk ids."
RECOMMEND_SYSTEM = "You are a Recommendation Agent. Compare companies; avoid direct buy/sell advice. Cite chunks."
ETHICAL_SYSTEM = "You are an Ethical Verifier. Return JSON {safe, issues, missing_citations}."

def generate_rag_context(conn, company, query, client):
    if not company:
        return ""
    try:
        hits = vector_search(conn, f"{company} {query}", client, top_k=6)
    except Exception:
        return ""
    parts = []
    for h in hits:
        cid = h["metadata"]["id"]
        snippet = h["content"]
        parts.append(f"[{cid}] {snippet[:800]}")
    return "\n\n".join(parts)


def run_financial_agent(conn, client, query, company, metrics_df, history):
    metrics = metrics_df[metrics_df["company"] == company]
    metrics_md = metrics.to_markdown(index=False) if not metrics.empty else "No metrics available."
    rag = generate_rag_context(conn, company, query, client)
    prompt = f"Query: {query}\n\nCompany: {company}\n\nMetrics:\n{metrics_md}\n\nRAG:\n{rag}\n\nAnswer briefly and cite chunk ids."
    return openai_chat(client, FINANCIAL_SYSTEM, prompt, history)


def run_risk_agent(conn, client, query, company, history):
    rag = generate_rag_context(conn, company, query, client)
    prompt = f"Query: {query}\n\nCompany: {company}\n\nRAG:\n{rag}\n\nSummarize main risks with citations."
    return openai_chat(client, RISK_SYSTEM, prompt, history)


def run_recommendation_agent(conn, client, query, companies, metrics_df, history):
    sub = metrics_df[metrics_df["company"].isin(companies)]
    metrics_md = sub.to_markdown(index=False) if not sub.empty else "No metrics."
    rag_texts = [generate_rag_context(conn, c, query, client) for c in companies]
    prompt = f"Query: {query}\n\nCompanies: {', '.join(companies)}\n\nMetrics:\n{metrics_md}\n\nRAG contexts:\n" + "\n\n".join(rag_texts) + "\n\nProvide balanced comparison with citations."
    return openai_chat(client, RECOMMEND_SYSTEM, prompt, history)


def run_ethics(conn, client, assistant_text, query, history):
    prompt = f"User query: {query}\n\nAssistant text:\n{assistant_text}\n\nReturn JSON with keys safe (bool), issues (list), missing_citations (list)."
    out = openai_chat(client, ETHICAL_SYSTEM, prompt, history)
    try:
        parsed = json.loads(out)
        return parsed
    except Exception:
        return {"safe": True, "issues": [], "verifier_text": out}


# -------------------------
# App state init
# -------------------------
if "conn" not in st.session_state:
    st.session_state.conn = init_db()
if "metrics_df" not in st.session_state:
    st.session_state.metrics_df = pd.DataFrame({
        "company": ["AAPL", "AAPL", "MSFT", "MSFT", "GOOGL", "GOOGL", "AMZN", "AMZN"],
        "year": [2022, 2023, 2022, 2023, 2022, 2023, 2022, 2023],
        "revenue": [394.3, 383.3, 198.3, 211.9, 257.6, 282.8, 513.9, 575.0],
        "net_income": [99.8, 97.0, 72.7, 77.0, 60.0, 66.3, 11.6, 30.0],
        "eps": [5.61, 5.40, 9.65, 10.00, 4.50, 4.90, 1.12, 3.20],
        "debt_to_equity": [1.8, 1.75, 0.5, 0.48, 0.3, 0.32, 0.9, 0.85]
    })
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history_export" not in st.session_state:
    st.session_state.chat_history_export = []


# -------------------------
# Sidebar / ingestion
# -------------------------
st.sidebar.title("⚙️ Settings & Ingestion")
api_key_input = st.sidebar.text_input("OpenAI API Key (or set OPENAI_API_KEY)", type="password")
openai_key = get_openai_key_from_input(api_key_input)
openai_client = make_openai_client(openai_key)

company_input = st.sidebar.text_input("Company name or CIK (e.g., Apple Inc or 0000320193)")
ingest_button = st.sidebar.button("Fetch & Ingest latest 10-Qs")

conn = st.session_state.conn

if ingest_button:
    if not company_input:
        st.sidebar.error("Enter company name or CIK")
    else:
        with st.spinner("Resolving CIK..."):
            cik = lookup_cik(company_input)
        if not cik:
            st.sidebar.error("CIK not found")
        else:
            st.sidebar.success(f"Resolved CIK: {cik}")
            with st.spinner("Fetching latest 10-Q URLs..."):
                urls = fetch_latest_10q_urls(cik, count=2)
            if not urls:
                st.sidebar.warning("No 10-Q URLs found")
            else:
                total = 0
                for url in urls:
                    with st.spinner(f"Downloading {url} ..."):
                        txt = download_filing_text(url)
                    if not txt:
                        st.sidebar.warning(f"Unable to download {url}")
                        continue
                    sections = extract_html_sections_from_filing(txt)
                    to_store = []
                    # chunk and store item_1a, tables, and partial full text
                    if sections.get("item_1a"):
                        chunks = simple_chunk_text(sections["item_1a"], chunk_size=500, overlap=120)
                        for idx, c in enumerate(chunks):
                            uid = hashlib.sha1((company_input + url + "item_1a" + str(idx)).encode()).hexdigest()
                            meta = {"id": uid, "company": company_input, "filing_url": url, "section": "item_1a", "chunk_index": idx, "content": c}
                            to_store.append((meta, c))
                    if sections.get("tables"):
                        chunks = simple_chunk_text(sections["tables"], chunk_size=500, overlap=120)
                        for idx, c in enumerate(chunks):
                            uid = hashlib.sha1((company_input + url + "tables" + str(idx)).encode()).hexdigest()
                            meta = {"id": uid, "company": company_input, "filing_url": url, "section": "tables", "chunk_index": idx, "content": c}
                            to_store.append((meta, c))
                    if sections.get("full_text"):
                        chunks = simple_chunk_text(sections["full_text"][:25000], chunk_size=700, overlap=140)
                        for idx, c in enumerate(chunks):
                            uid = hashlib.sha1((company_input + url + "full" + str(idx)).encode()).hexdigest()
                            meta = {"id": uid, "company": company_input, "filing_url": url, "section": "full_text", "chunk_index": idx, "content": c}
                            to_store.append((meta, c))
                    if to_store:
                        texts = [t for (_, t) in to_store]
                        if openai_client:
                            try:
                                embs = openai_embed(openai_client, texts)
                            except Exception as e:
                                st.sidebar.error(f"Embedding error: {e}")
                                embs = []
                        else:
                            st.sidebar.warning("OpenAI key missing — skipping embedding (store only).")
                            embs = []
                        for (meta, _), emb in zip(to_store, embs):
                            upsert_embedding(conn, meta, emb)
                        total += len(to_store)
                st.sidebar.success(f"Ingested ~{total} chunks (approx).")


# -------------------------
# Main UI tabs
# -------------------------
st.title("💹 FinChat AI — EDGAR ingestion + RAG (Fixed)")
tabs = st.tabs(["💬 Chat", "📊 Metrics", "⚠️ Risk", "🏁 Compare", "📤 Export", "🔧 Admin"])

# Chat
with tabs[0]:
    st.header("Chat")
    for m in st.session_state.messages:
        if m["role"] == "user":
            st.markdown(f"**You:** {m['content']}")
        else:
            st.markdown(f"**Assistant:** {m['content']}")
    user_q = st.text_input("Enter a question and press Enter")
    c1, c2 = st.columns([1, 3])
    with c1:
        companies = sorted(set(st.session_state.metrics_df["company"].tolist()))
        primary_company = st.selectbox("Primary company", companies, index=0)
        mode = st.selectbox("Agent mode", ["Auto", "Financial", "Risk", "Recommendation"])
        compare = st.multiselect("Compare companies", options=companies, default=[primary_company])
    if user_q:
        st.session_state.messages.append({"role": "user", "content": user_q})
        st.session_state.chat_history_export.append({"timestamp": datetime.utcnow().isoformat(), "role": "user", "content": user_q})
        with st.spinner("Running agents..."):
            try:
                if mode == "Financial":
                    reply = run_financial_agent(conn, openai_client, user_q, primary_company, st.session_state.metrics_df, st.session_state.messages)
                elif mode == "Risk":
                    reply = run_risk_agent(conn, openai_client, user_q, primary_company, st.session_state.messages)
                elif mode == "Recommendation":
                    reply = run_recommendation_agent(conn, openai_client, user_q, compare, st.session_state.metrics_df, st.session_state.messages)
                else:
                    ql = user_q.lower()
                    if any(k in ql for k in ["risk", "downside", "exposure"]):
                        reply = run_risk_agent(conn, openai_client, user_q, primary_company, st.session_state.messages)
                    elif any(k in ql for k in ["compare", "versus", "vs", "better"]):
                        reply = run_recommendation_agent(conn, openai_client, user_q, compare, st.session_state.metrics_df, st.session_state.messages)
                    else:
                        reply = run_financial_agent(conn, openai_client, user_q, primary_company, st.session_state.metrics_df, st.session_state.messages)
            except Exception as e:
                reply = f"[Agent error] {e}\n{traceback.format_exc()}"
            verifier = run_ethics(conn, openai_client, reply, user_q, st.session_state.messages) if openai_client else {"safe": True, "issues": [], "verifier_text": "(verifier offline)"}
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.chat_history_export.append({"timestamp": datetime.utcnow().isoformat(), "role": "assistant", "content": reply})
            st.markdown("**Assistant:**")
            st.write(reply)
            st.markdown("**Verifier output:**")
            st.json(verifier)

# Metrics
with tabs[1]:
    st.header("Metrics")
    st.dataframe(st.session_state.metrics_df, use_container_width=True)
    metric = st.selectbox(
        "Metric",
        ["revenue", "net_income", "eps", "debt_to_equity"],
        key="metrics_metric_select"
    )
    comp = st.selectbox(
        "Company to plot",
        sorted(st.session_state.metrics_df["company"].unique()),
        key="metrics_company_select"
    )
    fig = px.line(
        st.session_state.metrics_df[st.session_state.metrics_df["company"] == comp],
        x="year",
        y=metric,
        markers=True
    )
    st.plotly_chart(fig, use_container_width=True)

# Risk
with tabs[2]:
    st.header("Risk / Item 1A retrieval")
    rows = fetch_all_embeddings(conn)
    companies_with_embeddings = sorted({r[1] for r in rows}) if rows else []
    if companies_with_embeddings:
        company_risk = st.selectbox(
            "Company",
            options=companies_with_embeddings,
            key="risk_company_select"
        )
        if openai_client:
            retrieved = vector_search(conn, company_risk + " risk", openai_client, top_k=6)
        else:
            retrieved = []
        if not retrieved:
            st.info("No retrieved chunks or OpenAI key missing.")
        else:
            for r in retrieved:
                st.markdown(f"**Chunk {r['metadata']['id']}** — score {r['score']:.3f}")
                st.write(r["content"][:1200])
                st.markdown("---")
    else:
        st.info("No ingested filings. Use sidebar to ingest.")

# Compare
with tabs[3]:
    st.header("Compare")
    comps = st.multiselect(
        "Companies",
        options=sorted(st.session_state.metrics_df["company"].unique()),
        default=sorted(st.session_state.metrics_df["company"].unique())[:2],
        key="compare_companies_select"
    )
    metric2 = st.selectbox(
        "Metric",
        ["revenue", "net_income", "eps", "debt_to_equity"],
        key="compare_metric_select"
    )

    if comps:
        df_sub = st.session_state.metrics_df[st.session_state.metrics_df["company"].isin(comps)]
        fig2 = px.line(df_sub, x="year", y=metric2, color="company", markers=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(df_sub, use_container_width=True)

# Export
with tabs[4]:
    st.header("Export")
    st.download_button(
        "Download metrics CSV",
        st.session_state.metrics_df.to_csv(index=False).encode("utf-8"),
        file_name="metrics.csv",
        mime="text/csv",
        key="download_metrics_csv"
    )
    if st.session_state.chat_history_export:
        st.download_button(
            "Download chat CSV",
            pd.DataFrame(st.session_state.chat_history_export).to_csv(index=False).encode("utf-8"),
            file_name="chat_history.csv",
            mime="text/csv",
            key="download_chat_csv"
        )
    if st.button("Generate PDF", key="generate_pdf_button"):
        b = io.BytesIO()
        c = canvas.Canvas(b, pagesize=letter)
        w, h = letter
        c.setFont("Helvetica", 11)
        c.drawString(40, h - 40, f"FinChat Report - {datetime.utcnow().isoformat()}")
        y = h - 80
        for m in st.session_state.chat_history_export[-20:]:
            txt = f"{m['timestamp']} - {m['role']}: {m['content']}"
            c.drawString(40, y, txt[:120])
            y -= 14
            if y < 80:
                c.showPage()
                y = h - 40
        c.save()
        b.seek(0)
        st.download_button(
            "Download PDF",
            b,
            file_name="finchat_report.pdf",
            mime="application/pdf",
            key="download_pdf_button"
        )

# Admin
with tabs[5]:
    st.header("Admin / DB")
    rows = fetch_all_embeddings(conn)
    st.markdown(f"Total indexed chunks: {len(rows)}")
    if rows:
        df_dbg = pd.DataFrame([{"id": r[0], "company": r[1], "section": r[3], "chunk_index": r[4], "snippet": r[5][:300]} for r in rows])
        st.dataframe(df_dbg, use_container_width=True)
    st.markdown("Notes:")
    st.write(
        "- Uses submissions JSON primarily; falls back to index scrape.\n"
        "- Chunking + embeddings stored in SQLite. For scale, switch to FAISS/Chroma and Postgres for metadata.\n"
        "- For production add authentication and rate limiting."
    )

# End
st.sidebar.caption("FinChat AI — Fixed. Ensure you set a valid OpenAI API key for embeddings/chat.")

