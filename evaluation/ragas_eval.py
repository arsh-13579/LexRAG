import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline

load_dotenv()


# ─────────────────────────────────────────
# 1. TEST QUESTIONS
# ─────────────────────────────────────────

test_cases = [
    {
        "question": "What are the penalties for breaching this agreement?",
        "ground_truth": "Breach of this agreement will result in damages of up to 10 million rupees and the breaching party shall cover all legal costs incurred."
    },
    {
        "question": "What is the term of this agreement?",
        "ground_truth": "This agreement shall remain in effect for a period of two years from the date of signing. Either party may terminate with 30 days written notice."
    },
    {
        "question": "Which law governs this agreement?",
        "ground_truth": "This agreement shall be governed by the laws of India. Any disputes shall be resolved in the courts of New Delhi."
    },
    {
        "question": "Who are the parties in this agreement?",
        "ground_truth": "The agreement is between ABC Corporation as the Disclosing Party and XYZ Limited as the Receiving Party."
    },
    {
        "question": "What is considered confidential information?",
        "ground_truth": "Confidential information means any data or information that is proprietary to the Disclosing Party. The Receiving Party agrees not to disclose it to third parties and shall protect it with strict confidentiality."
    },
]


# ─────────────────────────────────────────
# 2. RUN PIPELINE ON EACH TEST CASE
# ─────────────────────────────────────────

def run_evaluation():
    print("Initializing pipeline...")
    pipeline = RAGPipeline()

    # check documents are loaded
    if pipeline.retriever is None:
        print("ERROR: No documents found in ChromaDB.")
        print("Please run the server and ingest sample.pdf first.")
        return

    print(f"Pipeline ready. {len(pipeline.chunks)} chunks loaded.")
    print("Running test cases...\n")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, test in enumerate(test_cases):
        print(f"Test {i+1}/{len(test_cases)}: {test['question'][:50]}...")

        # run full pipeline
        result = pipeline.query(
            question=test["question"],
            k=6,
            top_k=3
        )

        # collect results
        questions.append(test["question"])
        answers.append(result["answer"])
        ground_truths.append(test["ground_truth"])

        # extract context strings from sources
        if result["sources"]:
            context_list = [s["content"] for s in result["sources"]]
        else:
            context_list = [""]
        contexts.append(context_list)

        print(f"  Answer: {result['answer'][:80]}...")
        print(f"  Cached: {result['cached']}\n")

    # ─────────────────────────────────────────
    # 3. BUILD RAGAS DATASET
    # ─────────────────────────────────────────

    dataset = Dataset.from_dict({
        "question":   questions,
        "answer":     answers,
        "contexts":   contexts,
        "ground_truth": ground_truths,
    })

    # ─────────────────────────────────────────
    # 4. SETUP RAGAS LLM AND EMBEDDINGS
    # ─────────────────────────────────────────

    print("Setting up RAGAS evaluator...")

    ragas_llm = LangchainLLMWrapper(
        ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
            temperature=0.0
        )
    )

    ragas_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    )

    # ─────────────────────────────────────────
    # 5. RUN RAGAS EVALUATION
    # ─────────────────────────────────────────

    print("Running RAGAS evaluation...\n")

    results = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        raise_exceptions=False,
    )

    # ─────────────────────────────────────────
    # 6. PRINT RESULTS
    # ─────────────────────────────────────────

    print("\n" + "="*50)
    print("LEXRAG EVALUATION RESULTS")
    print("="*50)
    print(f"Faithfulness:      {results['faithfulness']:.4f}")
    print(f"Answer Relevancy:  {results['answer_relevancy']:.4f}")
    print(f"Context Precision: {results['context_precision']:.4f}")
    print(f"Context Recall:    {results['context_recall']:.4f}")
    print("="*50)

    avg = sum([
        results['faithfulness'],
        results['answer_relevancy'],
        results['context_precision'],
        results['context_recall']
    ]) / 4

    print(f"Overall Average:   {avg:.4f}")
    print("="*50)

    return results


if __name__ == "__main__":
    run_evaluation()