import math
from typing import Dict, List
from logging_util import loggers

class EvaluationService:

    @staticmethod
    def _discounted_cumulative_gain_at_k(
        retrieved_ids: List[str], ground_truth_ids: List[str], k: int
    ) -> float:
        """
        Computes the Discounted Cumulative Gain (DCG) @ K.
        """
        try:
            if (
                not isinstance(retrieved_ids, list)
                or not isinstance(ground_truth_ids, list)
                or not isinstance(k, int)
            ):
                raise TypeError(
                    "Invalid input types. Expected (List[Dict], List[str], int)."
                )

            if k <= 0:
                raise ValueError("Parameter 'k' should be a positive integer.")

            relevant = [_id for _id in retrieved_ids[:k]]
            relevance_scores = [
                1 if doc in ground_truth_ids else 0 for doc in relevant
            ]

            if not relevance_scores:
                return 0.0

            dcg_at_k = sum(
                relevance_scores[i] / math.log2(i + 2) for i in range(len(relevance_scores))
            )

            return dcg_at_k

        except Exception as e:
            loggers['MainLogger'].error(f"Error in discounted_cumulative_gain_at_k: {str(e)}")
            raise e

    @staticmethod
    def _ideal_discounted_cumulative_gain_at_k(
        ground_truth_ids: List[str], k: int
    ) -> float:
        """
        Computes the Ideal Discounted Cumulative Gain (IDCG) @ K.
        """
        try:
            if not isinstance(ground_truth_ids, list) or not isinstance(k, int):
                raise TypeError(
                    "Invalid input types. Expected (List[str], int)."
                )

            if k <= 0:
                raise ValueError("Parameter 'k' should be a positive integer.")

            ideal_relevance_scores = [1] * min(len(ground_truth_ids), k)

            if not ideal_relevance_scores:
                return 0.0

            idcg_at_k = sum(
                ideal_relevance_scores[i] / math.log2(i + 2) for i in range(len(ideal_relevance_scores))
            )

            return idcg_at_k

        except Exception as e:
            loggers['MainLogger'].error(f"Error in ideal_discounted_cumulative_gain_at_k: {str(e)}")
            return {"error": "couldnt calculate rr"}

    @staticmethod
    def normalized_discounted_cumulative_gain_at_k(
        retrieved_ids: List[str], ground_truth_ids: List[str], k: int
    ) -> float:
        """
        Computes the Normalized Discounted Cumulative Gain (NDCG) @ K.
        """
        try:
            if (
                not isinstance(retrieved_ids, list)
                or not isinstance(ground_truth_ids, list)
                or not isinstance(k, int)
            ):
                raise TypeError(
                    "Invalid input types. Expected (List[str], List[str], int)."
                )

            if k <= 0:
                raise ValueError("Parameter 'k' should be a positive integer.")

            ndcg_at_k = {}
            for i in range(k):
                dcg_at_i = EvaluationService._discounted_cumulative_gain_at_k(
                    retrieved_ids, ground_truth_ids, i + 1
                )
                idcg_at_i = (
                    EvaluationService._ideal_discounted_cumulative_gain_at_k(
                        ground_truth_ids, i + 1
                    )
                )
                ndcg_at_i = dcg_at_i / idcg_at_i if idcg_at_i > 0 else 0.0

                ndcg_at_k[f"NDCG@{i+1}"] = ndcg_at_i

            loggers['EvaluationLogger'].info(f"NDCG Result: {ndcg_at_k}")
            return ndcg_at_k

        except Exception as e:
            loggers['MainLogger'].error(f"Error in normalized_discounted_cumulative_gain_at_k: {str(e)}")
            raise e
        
    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str], ground_truth_ids: List[str], max_k: int = None
    ) -> Dict[str, float]:
        """
        Computes Precision@K for all k values from 1 to max_k.

        Args:
            retrieved_docs: List of retrieved documents, each with an "id" field
            ground_truth: List of ground truth documents, each with an "id" field
            max_k: Maximum k value to compute. If None, uses length of retrieved_docs

        Returns:
            Dictionary mapping 'Precision@k' to precision scores for k from 1 to max_k
        """
        try:
            if not isinstance(retrieved_ids, list) or not isinstance(
                ground_truth_ids, list
            ):
                raise TypeError(
                    "Invalid input types. Expected (List[Dict], List[Dict])."
                )

            if max_k is None:
                max_k = len(retrieved_ids)
            elif max_k <= 0:
                raise ValueError(
                    "Parameter 'max_k' should be a positive integer."
                )

            precision_results = {}

            for k in range(1, max_k + 1):
                retrieved_at_k = retrieved_ids[:k]
                matches = sum(
                    1 for doc_id in retrieved_at_k if doc_id in ground_truth_ids
                )
                precision_results[f"Precision@{k}"] = (
                    round(matches / float(k), 2) if k > 0 else 0.0
                )
            loggers['EvaluationLogger'].info(f"Precision Result: {precision_results}")
            return precision_results

        except Exception as e:
            loggers['MainLogger'].error(f"Error in precision_at_k: {e}")
            raise e

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str], ground_truth_ids: List[str], max_k: int = None
    ) -> Dict[str, float]:
        """
        Computes Recall@K for all k values from 1 to max_k.

        Args:
            retrieved_docs: List of retrieved documents, each with an "id" field
            ground_truth: List of ground truth documents, each with an "id" field
            max_k: Maximum k value to compute. If None, uses length of retrieved_docs

        Returns:
            Dictionary mapping 'Recall@k' to recall scores for k from 1 to max_k
        """
        try:
            if not isinstance(retrieved_ids, list) or not isinstance(
                ground_truth_ids, list
            ):
                raise TypeError(
                    "Invalid input types. Expected (List[Dict], List[Dict])."
                )

            if max_k is None:
                max_k = len(retrieved_docs)
            elif max_k <= 0:
                raise ValueError(
                    "Parameter 'max_k' should be a positive integer."
                )

            ground_truth_ids = set(ground_truth_ids)
            recall_results = {}

            if len(ground_truth_ids) == 0:
                return {f"Recall@{k}": 0.0 for k in range(1, max_k + 1)}

            for k in range(1, max_k + 1):
                retrieved_at_k = set(retrieved_ids[:k])
                matches = len(ground_truth_ids.intersection(retrieved_at_k))
                recall_results[f"Recall@{k}"] = round(
                    matches / float(len(ground_truth_ids)), 2
                )
            loggers['EvaluationLogger'].info(f"Recall Results: {recall_results}")
            return recall_results

        except Exception as e:
            loggers['MainLogger'].error(f"Error in recall_at_k: {e}")
            raise e