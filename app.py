import streamlit as st
from rag_pipeline import run_financial_rag

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Financial AI Chatbot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #0E1117;
    color: white;
}

/* Main title */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
    margin-bottom: 0;
}

.subtitle {
    font-size: 16px;
    color: #A0A0A0;
    margin-top: -10px;
    margin-bottom: 25px;
}

/* Chat container */
[data-testid="stChatMessage"] {
    padding: 14px;
    border-radius: 18px;
    margin-bottom: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* User message */
[data-testid="stChatMessage"]:has(.user) {
    background-color: #1E293B;
}

/* Assistant message */
[data-testid="stChatMessage"]:has(.assistant) {
    background-color: #111827;
}

/* Chat input */
.stChatInputContainer {
    border-top: 1px solid #333;
    background-color: #0E1117;
}

/* Buttons */
.stButton button {
    background-color: #2563EB;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.5rem 1rem;
    font-weight: 600;
}

.stButton button:hover {
    background-color: #1D4ED8;
    color: white;
}

/* Small caption */
.small-text {
    color: #9CA3AF;
    font-size: 14px;
}

/* Reduce markdown heading sizes */
h1 {
    font-size: 28px !important;
}

h2 {
    font-size: 24px !important;
}

h3 {
    font-size: 20px !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">📈 Financial Market AI Chatbot</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Agentic Financial Intelligence powered by LangGraph + Groq</div>',
    unsafe_allow_html=True
)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ System Info")

    st.markdown("""
    ### AI Stack
    
    - LangGraph Workflow
    - Groq LLM
    - FAISS Vector DB
    - Financial RAG
    - Streamlit UI
    """)

    st.markdown("---")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================================================
# BASIC CHAT RESPONSES
# =========================================================

def basic_chat_response(prompt):

    prompt = prompt.lower()

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "good morning",
        "good evening",
        "hola",
        "namaste"
    ]

    if prompt in greetings:

        return """
👋 Hello!

I'm your AI Financial Assistant.

You can ask me about:

- Apple financial performance
- Investment opportunities
- Revenue & profits
- Market risks
- Stock analysis
- Financial forecasting
"""

    elif "who are you" in prompt:

        return """
I'm an AI-powered Financial Market Intelligence Assistant built using:

- LangGraph
- Groq LLM
- FAISS Vector Search
- Multi-Agent Financial Reasoning
"""

    elif "thank you" in prompt or "thanks" in prompt:

        return "You're welcome 😊"

    return None

# =========================================================
# QUERY MODE
# =========================================================

def get_query_mode(query):

    query = query.lower()

    short_keywords = [
        "what",
        "who",
        "when",
        "revenue",
        "profit",
        "employees",
        "dividend",
        "stock"
    ]

    if len(query.split()) <= 8:
        return "simple"

    if any(word in query for word in short_keywords):
        return "simple"

    return "detailed"

# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "Ask about Apple financials, investments, risks, or market trends..."
)

# =========================================================
# USER MESSAGE
# =========================================================

if prompt:

    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # =====================================================
    # BASIC CHAT HANDLING
    # =====================================================

    basic_response = basic_chat_response(prompt)

    if basic_response:

        with st.chat_message("assistant"):
            st.markdown(basic_response)

        st.session_state.messages.append({
            "role": "assistant",
            "content": basic_response
        })

    # =====================================================
    # FINANCIAL RAG PIPELINE
    # =====================================================

    else:

        with st.chat_message("assistant"):

            with st.spinner("📊 Analyzing financial intelligence..."):

                try:

                    mode = get_query_mode(prompt)

                    response = run_financial_rag(
                        prompt,
                        mode=mode
                    )

                    answer = response["answer"]

                    latency = response["latency"]

                    st.markdown(answer)

                    st.markdown(
                        f'<div class="small-text">⏱️ Response Time: {latency} sec</div>',
                        unsafe_allow_html=True
                    )

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(f"❌ Error: {str(e)}")