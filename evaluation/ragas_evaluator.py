from typing import Dict, Any, List, Optional
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_relevancy,
    answer_similarity,
)
from ragas import evaluate
from datasets import Dataset
import numpy as np
from typing import List, Dict, Optional
from ragas import evaluate
from datasets import Dataset
import numpy as np

class RagasEvaluator:
    """
    A class to evaluate RAG responses using RAGAS metrics.
    """
    
    def __init__(self):
        # Only include metrics that don't require ground truth by default
        self.metrics = [
            answer_relevancy,
            faithfulness,
            context_relevancy,
        ]
        
        # Metrics that require ground truth
        self.gt_metrics = [
            answer_similarity,
        ]
    
    def prepare_dataset(
        self, 
        question: str, 
        answer: str, 
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Prepare the dataset for evaluation.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of context strings used for generation
            ground_truth: The ground truth answer (if available)
            
        Returns:
            Dictionary containing the evaluation results
        """
        if contexts is None:
            contexts = []
            
        # Create dataset in the format expected by RAGAS
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        if ground_truth:
            data["ground_truth"] = [ground_truth]
            
        return Dataset.from_dict(data)
    
    def evaluate_response(
        self,
        question: str,
        answer: str,
        contexts: Optional[List[str]] = None,
        ground_truth: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate a single response using RAGAS metrics.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of context strings used for generation
            ground_truth: The ground truth answer (if available)
            
        Returns:
            Dictionary containing the evaluation metrics
        """
        # Prepare metrics based on availability of ground truth
        metrics_to_use = self.metrics.copy()
        if ground_truth is not None:
            metrics_to_use.extend(self.gt_metrics)
            
        dataset = self.prepare_dataset(question, answer, contexts, ground_truth)
        
        scores = {}
        if metrics_to_use:  # Only evaluate if we have metrics to run
            # Run evaluation
            try:
                result = evaluate(
                    dataset=dataset,
                    metrics=metrics_to_use,
                    raise_exceptions=False
                )
                
                # Convert to a simple dictionary of scores
                for metric in metrics_to_use:
                    metric_name = metric.name
                    if metric_name in result and result[metric_name] is not None:
                        scores[metric_name] = float(np.mean(result[metric_name]))
            except Exception as e:
                # If evaluation fails, we'll return an empty dict and let the caller handle it
                pass
        
        return scores
