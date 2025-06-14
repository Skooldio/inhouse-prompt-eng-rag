from typing import Dict, List
import numpy as np
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy,
)
from ragas import evaluate
from datasets import Dataset


def evaluate_rag_response(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, float]:
    """
    Evaluate a RAG response using RAGAS metrics.

    Args:
        question: The original question that was asked
        answer: The generated answer to evaluate
        contexts: List of context strings used to generate the answer
    Returns:
        Dict containing metric names and their scores
    """
    # 1. Define metrics to use
    metrics = [faithfulness, answer_relevancy, context_relevancy]

    print(f"\n🧪 Evaluating with metrics: {[m.name for m in metrics]}")

    # 2. Prepare dataset
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }
    dataset = Dataset.from_dict(data)

    try:
        # 3. Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            raise_exceptions=False,
        )

        # 4. Process results
        scores = {}
        for metric in metrics:
            metric_name = metric.name
            if metric_name in result and result[metric_name] is not None:
                scores[metric_name] = float(np.mean(result[metric_name]))
        print(f"✅ Evaluation complete. Scores: {scores}")
        return scores

    except Exception as e:
        print(f"🔥 Evaluation failed: {str(e)}")
        pass
    return {}
