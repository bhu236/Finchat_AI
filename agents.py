from openai import OpenAI

client = OpenAI(api_key="YOUR_API_KEY")

def init_chat_memory():
    return []

def generate_advice(user_profile, company_data, chat_memory, user_question):
    context_text = " ".join([chunk["text"] for chunk in company_data.get("filings_context", [])])

    prompt = f"""
    You are a friendly AI financial advisor for beginners.
    User profile: {user_profile}
    Company: {company_data.get('ticker')}
    Financial Metrics: {company_data.get('financial_metrics')}
    Stock Info: {company_data.get('stock_info')}
    News Sentiment: {company_data.get('news_sentiment')}
    Relevant Filing Context: {context_text}
    User Question: {user_question}

    Provide:
    1. Beginner-friendly explanation
    2. Risk analysis and alternatives
    3. Disclaimers that advice is educational only
    """

    messages = [{"role": "system", "content": "You are a helpful AI financial advisor."}]
    for msg in chat_memory:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.7
    )

    advice_text = response.choices[0].message.content
    chat_memory.append({"role": "user", "content": user_question})
    chat_memory.append({"role": "assistant", "content": advice_text})
    return advice_text
