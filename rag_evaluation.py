import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.embeddings import HuggingFaceEmbeddings

# IMPORT YOUR RAG PIPELINE
from rag_pipeline import run_financial_rag

# EMBEDDING MODEL
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# GROUND TRUTH DATA
GROUND_TRUTH = [
    {
        "id": "Q1",
        "question": "What was Apple's total net sales in 2025 and how does it compare to 2024?",
        "reference": "Total net sales were $416,161 million in 2025, a 6% increase from $391,035 million in 2024."
    },
    {
        "id": "Q2",
        "question": "Which geographic segment saw a revenue decrease in 2025?",
        "reference": "Greater China net sales decreased to $64,377 million in 2025 from $66,952 million in 2024."
    },
    {
        "id": "Q3",
        "question": "Explain the 2024 income tax charge related to the European Commission.",
        "reference": "Apple recorded a $10.2 billion one-time income tax charge in 2024 following the ECJ decision on the State Aid Decision."
    },
    {
        "id": "Q4",
        "question": "Who will become the Principal Accounting Officer on January 1, 2026?",
        "reference": "Ben Borders will assume the role of Principal Accounting Officer, reporting to CFO Kevan Parekh."
    },
    {
        "id": "Q5",
        "question": "What was the year-over-year growth for the Services segment in 2025?",
        "reference": "Services net sales grew 14% in 2025, reaching $109,158 million."
    },
    {
        "id": "Q6",
        "question": "What new trade risks were identified starting in the second quarter of 2025?",
        "reference": "New U.S. Tariffs were announced on imports from China, India, Japan, South Korea, Taiwan, Vietnam, and the EU."
    },
    {
        "id": "Q7",
        "question": "What is the status of the Epic Games litigation as of the 2025 report?",
        "reference": "On April 30, 2025, a court found Apple in violation of the 2021 Injunction."
    },
    {
        "id": "Q8",
        "question": "How many full-time equivalent employees did Apple have at the end of fiscal 2025?",
        "reference": "Apple had approximately 166,000 full-time equivalent employees."
    },
    {
        "id": "Q9",
        "question": "What was the dividend amount per share declared in May 2025?",
        "reference": "Apple raised its quarterly cash dividend from $0.25 to $0.26 per share in May 2025."
    },
    {
        "id": "Q10",
        "question": "What drove the increase in R&D expenses during fiscal 2025?",
        "reference": "R&D growth was primarily driven by higher headcount-related expenses and infrastructure-related costs."
    }
]

# GENERATE ANSWERS
records = []

print("\n" + "=" * 80)
print("RUNNING FINANCIAL RAG EVALUATION")
print("=" * 80)

for item in GROUND_TRUTH:
    print(f"\nProcessing {item['id']}...")
    try:
        # CALL RAG PIPELINE
        result = run_financial_rag(
            query=item["question"],
            mode="simple",
            evaluation=True
        )
        
        # EXTRACT RESPONSE
        generated = result.get("answer", "")
        retrieved_context = result.get("context", "")
        latency = result.get("latency", 0)

        print(f"Generated Answer: {generated[:120]}...")

        records.append({
            "id": item["id"],
            "question": item["question"],
            "reference": item["reference"],
            "generated": generated,
            "context": retrieved_context,
            "latency": latency
        })

    except Exception as e:
        print(f"Error on {item['id']}: {str(e)}")
        records.append({
            "id": item["id"],
            "question": item["question"],
            "reference": item["reference"],
            "generated": "",
            "context": "",
            "latency": 0
        })


# ROUGE-L SCORE

print("CALCULATING ROUGE-L SCORES---------------------")

scorer = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)

for r in records:
    try:
        rouge_score = scorer.score(
            r["reference"],
            r["generated"]
        )
        r["rouge_l"] = round(
            rouge_score["rougeL"].fmeasure,
            4
        )
    except Exception as e:
        print(f"ROUGE Error: {e}")
        r["rouge_l"] = 0.0


# BLEU SCORE

print("CALCULATING BLEU SCORES------------------------")
smoothie = SmoothingFunction().method1

for r in records:
    try:
        reference = [r["reference"].split()]
        generated = r["generated"].split()

        bleu = sentence_bleu(
            reference,
            generated,
            smoothing_function=smoothie
        )

        r["bleu"] = round(bleu, 4)

    except Exception as e:
        print(f"BLEU Error: {e}")
        r["bleu"] = 0.0


# RELEVANCE SCORE (COSINE SIMILARITY)
print("CALCULATING RELEVANCE SCORES---------------------")

for r in records:
    try:
        ref_embedding = embedding_model.embed_query(
            r["reference"]
        )
        gen_embedding = embedding_model.embed_query(
            r["generated"]
        )
        relevance = cosine_similarity(
            [ref_embedding],
            [gen_embedding]
        )[0][0]

        r["relevance"] = round(
            float(relevance),
            4
        )

    except Exception as e:
        print(f"Relevance Error: {e}")
        r["relevance"] = 0.0


# CREATE DATAFRAME
df = pd.DataFrame(records)

# DISPLAY RESULTS
print("\n" + "=" * 80)
print("RAG EVALUATION RESULTS")
print("=" * 80)

print(
    df[
        [
            "id",
            "bleu",
            "rouge_l",
            "relevance",
            "latency"
        ]
    ]
)

# MEAN PERFORMANCE
print("\n" + "=" * 80)
print("MEAN PERFORMANCE")
print("=" * 80)

print(
    df[
        [
            "bleu",
            "rouge_l",
            "relevance",
            "latency"
        ]
    ].mean()
)

# SAVE RESULTS
output_file = "financial_rag_evaluation_results.csv"

df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 80)
print("EVALUATION COMPLETED")
print("=" * 80)

print(f"\nResults saved to: {output_file}")