from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from rag_pipeline import run_financial_rag

# FASTAPI APP
app = FastAPI(
    title="Financial Market Intelligence API",
    description="Agentic Financial RAG API using LangGraph + Groq",
    version="1.0.0"
)

# REQUEST MODEL
class QueryRequest(BaseModel):
    query: str
    mode: Optional[str] = "detailed"

# RESPONSE MODEL
class QueryResponse(BaseModel):
    query: str
    answer: str
    latency: float

# HOME ROUTE
@app.get("/")
def home():
    return {
        "message": "Financial Market Intelligence API is running"
    }

# HEALTH CHECK
@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }

# MAIN RAG ENDPOINT
@app.post(
    "/analyze",
    response_model=QueryResponse
)

def analyze_financial_query(request: QueryRequest):
    try:
        result = run_financial_rag(
            query=request.query,
            mode=request.mode
        )
        return {
            "query": request.query,
            "answer": result["answer"],
            "latency": result["latency"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# RUN SERVER
if __name__ == "__main__":
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=True)