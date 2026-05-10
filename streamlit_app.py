import streamlit as st
from rag_pipeline import run_financial_rag

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Financial AI Chatbot",
    page_icon="💬",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stChatMessage {
    border-radius: 15px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("Financial Market AI Chatbot")
st.caption("Agentic Financial Market Intelligence Assistant")

# =========================================================
# CLEAR CHAT
# =========================================================

if st.button("Clear Chat"):
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
# SIMPLE CHATBOT RESPONSES
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
        "holla",
        "Namaste"
    ]

    if prompt in greetings:

        return """
Hello 👋

I'm your Financial AI Assistant.

You can ask me about:

- Apple financial performance
- Investment opportunities
- Market risks
- Revenue growth
- Financial forecasting
- Portfolio insights
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
# CHAT INPUT
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

    # Short question
    if len(query.split()) <= 8:
        return "simple"

    # Small factual question
    if any(word in query for word in short_keywords):
        return "simple"

    return "detailed"

prompt = st.chat_input(
    "Ask about financial markets, Apple stock, investments, risks..."
)

# =========================================================
# USER MESSAGE
# =========================================================

if prompt:
    # Show user message
    st.chat_message("user").markdown(prompt)

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
            with st.spinner("Analyzing financial data..."):
                try:

                    mode = get_query_mode(prompt)
                    response = run_financial_rag(
                        prompt,
                        mode=mode
                    )
                    answer = response["answer"]
                    latency = response["latency"]
                    st.markdown(answer)
                    st.caption(
                        f"Response Time: {latency} sec"
                    )

                    # Save assistant response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")