import os
from dotenv import load_dotenv

from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv()

# =========================================================
# GEMINI LLM
# =========================================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.7,
    max_tokens=500
)

# =========================================================
# LOAD VECTORSTORE
# =========================================================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# =========================================================
# GRAPH STATE
# =========================================================

class FinancialState(TypedDict):
    query: str
    mode: str
    retrieved_docs: List[Document]
    context: str
    retrieval_analysis: str
    market_analysis: str
    portfolio_analysis: str
    risk_analysis: str
    final_response: str


# =========================================================
# AGENT 1 : RETRIEVER AGENT
# =========================================================

retriever_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
        You are an elite Financial Market Intelligence Retrieval Agent.
        You work for a hedge fund level AI financial intelligence system.
        Your task is to retrieve and summarize the most relevant financial information from:
        - SEC filings
        - Earnings reports
        - Financial statements
        - Stock market reports
        - Apple financial dataset
        - Economic indicators
        - Financial PDFs

        GOAL:
        Provide the most useful retrieved information for downstream AI agents.

        USER QUERY:
        {query}

        RETRIEVED CONTEXT:
        {context}

        INSTRUCTIONS:
        1. Extract only highly relevant information
        2. Remove duplicate insights
        3. Focus on:
        - Revenue
        - Profitability
        - Market performance
        - Competitive signals
        - Investment trends
        - Financial growth
        4. Mention important financial indicators
        5. Mention numerical evidence if available
        6. Keep the response factual
        7. Do not hallucinate

        OUTPUT FORMAT:
        1. Key Financial Findings
        2. Important Numbers
        3. Market Indicators
        4. Business Insights
        5. Strategic Signals
        """
)

def retrieval_agent(state):
    query = state["query"]
    docs = retriever.invoke(query)
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    prompt = retriever_prompt.format(
        query=query,
        context=context
    )
    response = llm.invoke(prompt)
    return {
        "retrieved_docs": docs,
        "context": context,
        "retrieval_analysis": response.content
    }

# =========================================================
# AGENT 2 : MARKET ANALYST AGENT
# =========================================================

market_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are a Senior Wall Street Financial Market Analyst AI Agent.
            BACKSTORY:
            You specialize in:
            - equity research
            - market forecasting
            - earnings analysis
            - institutional investment strategy
            - macroeconomic intelligence

            Your job is to deeply analyze retrieved financial data.

            USER QUERY:
            {query}

            FINANCIAL CONTEXT:
            {context}

            TASKS:
            1. Analyze company performance
            2. Analyze market growth
            3. Detect financial strengths
            4. Detect financial weaknesses
            5. Analyze investor confidence
            6. Analyze growth opportunities
            7. Analyze future outlook

            IMPORTANT:
            - Use evidence from context
            - Mention financial metrics
            - Mention trends
            - Mention possible business implications

            OUTPUT FORMAT:
            ## Executive Summary 
            ## Financial Performance
            ## Growth Indicators
            ## Weaknesses
            ## Investment Signals
            ## Future Outlook
            """
)

def market_analyst_agent(state):
    prompt = market_prompt.format(
        query=state["query"],
        context=state["retrieval_analysis"]
    )
    response = llm.invoke(prompt)
    return {
        "market_analysis": response.content
    }

# =========================================================
# AGENT 3 : PORTFOLIO AGENT
# =========================================================

portfolio_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are an Institutional Portfolio Management AI Agent.
            BACKSTORY:
            You work for a global investment management firm.
            Your responsibilities:
            - portfolio allocation
            - investment diversification
            - stock recommendations
            - growth portfolio optimization
            - financial forecasting

            USER QUERY:
            {query}

            FINANCIAL CONTEXT:
            {context}

            TASK:
            1. Suggest investment strategy
            2. Determine whether:
            - Buy
            - Hold
            - Sell
            3. Mention risk-adjusted strategy
            4. Mention diversification signals
            5. Mention long-term outlook
            6. Mention portfolio impact

            OUTPUT FORMAT:
            ## Portfolio Recommendation
            ## Suggested Allocation
            ## Risk vs Reward
            ## Long-Term Investment Outlook
            ## Final Investment Decision
            """
)

def portfolio_agent(state):
    prompt = portfolio_prompt.format(
        query=state["query"],
        context=state["market_analysis"]
    )
    response = llm.invoke(prompt)

    return {
        "portfolio_analysis": response.content
    }

# =========================================================
# AGENT 4 : RISK ASSESSMENT AGENT
# =========================================================

risk_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are a Financial Risk Assessment AI Agent.
            BACKSTORY:
            You specialize in:
            - market volatility
            - investment risk
            - debt risk
            - liquidity risk
            - operational risk
            - macroeconomic threats

            USER QUERY:
            {query}

            CONTEXT:
            {context}

            TASK:
            1. Identify major financial risks
            2. Analyze volatility
            3. Analyze debt exposure
            4. Analyze operational uncertainty
            5. Analyze future financial threats
            6. Generate risk score

            RISK SCORE:
            1 = Very Low Risk
            10 = Extremely High Risk

            OUTPUT FORMAT:
            ## Risk Summary
            ## Financial Threats
            ## Volatility Analysis
            ## Risk Score
            ## Institutional Risk Outlook
            """
)

def risk_agent(state):
    prompt = risk_prompt.format(
        query=state["query"],
        context=state["market_analysis"]
    )
    response = llm.invoke(prompt)

    return {
        "risk_analysis": response.content
    }


# =========================================================
# AGENT 6 : FINAL DECISION AGENT
# =========================================================

final_prompt = PromptTemplate(
    input_variables=[
    "query",
    "retrieval",
    "market",
    "portfolio",
    "risk"
],
    template="""You are the Chief Financial Intelligence AI Agent.
            BACKSTORY:
            You are the final decision-making AI in an enterprise-grade Financial Market Intelligence multi-agent system.
            Your job is to combine:
            - Retrieval intelligence
            - Financial analysis
            - Portfolio analysis
            - Risk assessment

            Generate a professional institutional-level financial intelligence report.
            USER QUERY:
            {query}

            RESPONSE MODE:
            {mode}
            
            RETRIEVAL ANALYSIS:
            {retrieval}

            MARKET ANALYSIS:
            {market}

            PORTFOLIO ANALYSIS:
            {portfolio}

            RISK ANALYSIS:
            {risk}


            IMPORTANT INSTRUCTIONS:
            If RESPONSE MODE is "simple":
            - answer in 1-3 concise sentences
            - avoid markdown headings
            - avoid long analysis
            - keep conversational tone

            If RESPONSE MODE is "detailed":
            - provide structured financial analysis
            - use markdown sections
            - include investment insights
            - include risks and opportunities

            Do not hallucinate.

            OUTPUT FORMAT:
            ## Executive Summary
            ## Key Financial Insights
            ## Market Intelligence
            ## Investment Recommendation
            ## Risk Assessment
            ## Strategic Opportunities
            ## Final Institutional Recommendation
            ## Confidence Score
            """
)

def final_agent(state):
    prompt = final_prompt.format(
        query=state["query"],
        mode=state["mode"],
        retrieval=state["retrieval_analysis"],
        market=state["market_analysis"],
        portfolio=state["portfolio_analysis"],
        risk=state["risk_analysis"]
    )
    response = llm.invoke(prompt)
    return {
        "final_response": response.content
    }

# =========================================================
# LANGGRAPH WORKFLOW
# =========================================================

workflow = StateGraph(FinancialState)

workflow.add_node("RetrieverAgent",retrieval_agent)
workflow.add_node("MarketAnalystAgent",market_analyst_agent)
workflow.add_node("PortfolioAgent",portfolio_agent)
workflow.add_node("RiskAgent",risk_agent)
workflow.add_node("FinalAgent",final_agent)

# =========================================================
# EDGES
# =========================================================

workflow.set_entry_point("RetrieverAgent")

workflow.add_edge("RetrieverAgent","MarketAnalystAgent")
workflow.add_edge("MarketAnalystAgent","PortfolioAgent")
workflow.add_edge("PortfolioAgent","RiskAgent")
workflow.add_edge("RiskAgent","FinalAgent")
workflow.add_edge("FinalAgent",END)

# =========================================================
# COMPILE GRAPH
# =========================================================

graph = workflow.compile()

# =========================================================
# MAIN FUNCTION
# =========================================================
import time

def run_financial_rag(
    query,
    mode="detailed",
    evaluation=False
):
    start = time.time()

    # SIMPLE MODE
    if mode == "simple":
        docs = retriever.invoke(query)
        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )
        simple_prompt = f"""
            You are a financial AI assistant.
            Answer the user's question briefly and directly.
            Rules:
            - Maximum 2-3 sentences
            - No markdown headings
            - No long explanations
            - Be factual
            - Use retrieved context only
            Question:
            {query}

            Context:
            {context}
            """

        response = llm.invoke(simple_prompt)
        final_answer = response.content

    # DETAILED MODE
    else:
        response = graph.invoke({
            "query": query,
            "mode": mode
        })

        context = response["context"]
        final_answer = response["final_response"]
        
    end = time.time()
    latency = round(end - start, 2)
    return {
        "answer": final_answer,
        "context": context,
        "latency": latency
    }
    

    # =====================================================
    # RETURN FOR EVALUATION
    # =====================================================

    if evaluation:

        return {
            "answer": final_answer,
            "context": response.get("context", ""),
            "latency": latency
        }

    # =====================================================
    # NORMAL RETURN
    # =====================================================

    return {
        "answer": final_answer,
        "latency": latency
    }
# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    query = """
    Analyze Apple's recent financial performance,
    investment opportunities,
    future growth,
    and overall market risks.
    """

    result = run_financial_rag(query)

    print("\n")
    print("=" * 80)
    print("FINAL FINANCIAL INTELLIGENCE REPORT")
    print("=" * 80)
    print("\n")

    print(result)