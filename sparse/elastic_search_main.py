import json
import os
import asyncio
from elasticsearch import AsyncElasticsearch, Elasticsearch
from typing import List, Dict, Any
import aiofiles
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation import EvaluationService


# Constants
TOP_K = 15
RESULTS_DIR = "/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/sparse/sparse_results/with_context"
JSON_FILE = "/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/cache_data/final_chunks_with_context.json"
# JSON_FILE = "/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/cache_data/final_chunks.json"

def calculate_evaluation_metrics(relevant_ids, ground_truth_ids, top_k=TOP_K):
    """Calculate various evaluation metrics for the search results."""
    try:
        return {
            "precision_at_k": EvaluationService.precision_at_k(
                retrieved_ids=relevant_ids,
                ground_truth_ids=ground_truth_ids,
                max_k=top_k,
            ),
            "recall_at_k": EvaluationService.recall_at_k(
                retrieved_ids=relevant_ids,
                ground_truth_ids=ground_truth_ids,
                max_k=top_k,
            ),
            "ndcg_at_k": EvaluationService.normalized_discounted_cumulative_gain_at_k(
                retrieved_ids=relevant_ids, ground_truth_ids=ground_truth_ids, k=top_k
            )
        }
    except Exception as e:
        print(f"Error calculating evaluation metrics: {e}")
        raise e


def generate_evaluation_metrics(relevant_ids, ground_truth_ids, top_k=TOP_K):
    """Calculate various evaluation metrics for the search results using thread pool."""
    with ThreadPoolExecutor() as executor:
        evaluation_metrics = executor.submit(
            calculate_evaluation_metrics,
            relevant_ids,
            ground_truth_ids,
            top_k
        ).result()
    return evaluation_metrics

def average_metrics(metrics_list, top_k = TOP_K):
    
        sum_dict = {
        "precision_at_k": {},
        "recall_at_k": {},
        "ndcg_at_k": {}
        }

        for k in range(1, top_k + 1):
            sum_dict["precision_at_k"][f"Precision@{k}"] = 0.0
            sum_dict["recall_at_k"][f"Recall@{k}"] = 0.0
            sum_dict["ndcg_at_k"][f"NDCG@{k}"] = 0.0
        
        avg_dict = sum_dict.copy()
        
        count_dict = {
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "ndcg_at_k": 0.0,
        }
        
        for metric_dict in metrics_list:
            for key, value in metric_dict.items():
                if isinstance(value, dict):
                    
                    for sub_key, sub_value in value.items():
                        sum_dict[key][sub_key] += sub_value
                    count_dict[key] += 1
                elif isinstance(value, (int, float)):
                    
                    sum_dict[key] += value
                    count_dict[key] += 1
        
        for key, value in sum_dict.items():
            if isinstance(value, dict): 
                avg_dict[key] = {sub_key: sub_value / count_dict[key] 
                                for sub_key, sub_value in value.items()}
            else:
                avg_dict[key] = value / count_dict[key]
        
        return avg_dict

async def connect_to_elasticsearch(
    host="http://localhost:9200", username="elastic", password="zKRURlMb"
):
    """Connect to Elasticsearch server asynchronously"""
    try:
        es = AsyncElasticsearch(host, basic_auth=(username, password))
        if not await es.ping():
            print("Could not connect to Elasticsearch")
            return None
        print("Successfully connected to Elasticsearch")
        return es
    except Exception as e:
        print(f"Connection error: {e}")
        return None


def connect_to_elasticsearch_sync(
    host="http://localhost:9200", username="elastic", password="zKRURlMb"
):
    """Connect to Elasticsearch server (synchronous version for indexing)"""
    try:
        es = Elasticsearch(host, basic_auth=(username, password))
        if not es.ping():
            print("Could not connect to Elasticsearch")
            exit(1)
        print("Successfully connected to Elasticsearch")
        return es
    except Exception as e:
        print(f"Connection error: {e}")
        exit(1)


async def check_existing_index(es, index_name):
    """Check if index exists and list available indices"""
    indices = await es.indices.get(index="*")
    print(f"Available indices: {', '.join(indices.keys())}")

    if await es.indices.exists(index=index_name):
        count = await es.count(index=index_name)
        print(f"Index '{index_name}' already exists with {count['count']} documents")
        return True
    return False


def check_existing_index_sync(es, index_name):
    """Check if index exists and list available indices (synchronous version)"""
    indices = es.indices.get(index="*")
    print(f"Available indices: {', '.join(indices.keys())}")

    if es.indices.exists(index=index_name):
        count = es.count(index=index_name)
        print(f"Index '{index_name}' already exists with {count['count']} documents")
        return True
    return False


def create_index(es, index_name):
    """Create a new index with BM25 analyzer"""
    index_settings = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "custom_bm25_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase"],
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "file_path": {"type": "text"},
                "file_name": {"type": "keyword"},
                "start_line": {"type": "integer"},
                "end_line": {"type": "integer"},
                "content": {"type": "text", "analyzer": "custom_bm25_analyzer"},
                "size": {"type": "integer"},
                "parent-class": {"type": "keyword", "null_value": "null"},
                "function_name": {"type": "keyword", "null_value": "null"},
            }
        },
    }

    es.indices.create(index=index_name, body=index_settings)
    print(f"Created new index '{index_name}'")


def load_json_data(file_path):
    """Load chunk data from JSON file"""
    if not os.path.exists(file_path):
        print(f"Error: JSON file not found at {file_path}")
        exit(1)

    try:
        with open(file_path, "r") as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from JSON file")

        if not chunks:
            print("Warning: JSON file is empty or contains no valid chunks")
            exit(1)

        # Print a sample document to verify format
        print("\nSample document structure:")
        print(json.dumps(chunks[0], indent=2)[:500] + "...\n")

        return chunks

    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)


def index_documents(es, index_name, chunks):
    """Index documents into Elasticsearch"""
    success_count = 0
    error_count = 0

    for i, chunk in enumerate(chunks):
        try:
            response = es.index(index=index_name, document=chunk)
            if response["result"] == "created":
                success_count += 1
            else:
                error_count += 1
                print(f"Warning: Document {i} not properly created: {response}")
        except Exception as e:
            error_count += 1
            print(f"Error indexing document {i}: {e}")

    print(f"Indexing complete: {success_count} successful, {error_count} failed")

    # Force refresh to make sure all documents are searchable
    es.indices.refresh(index=index_name)

    # Check document count after indexing
    count_after = es.count(index=index_name)
    print(f"Total documents indexed: {count_after['count']}")

    return count_after["count"]


async def load_questions(file_path="/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/cache_data/questions.json"):
    """Load questions from JSON file"""
    if not os.path.exists(file_path):
        print(f"Error: Questions file not found at {file_path}")
        return None

    try:
        async with aiofiles.open(file_path, "r") as f:
            content = await f.read()
            questions = json.loads(content)

        print(f"Loaded {len(questions)} questions from JSON file")
        return questions

    except json.JSONDecodeError:
        print(f"Error: The file {file_path} contains invalid JSON")
        return None
    except Exception as e:
        print(f"Error loading questions: {str(e)}")
        return None


async def perform_search(es, index_name, query, size=TOP_K):
    """Perform BM25 search on indexed documents"""
    search_body = {
        "query": {
            "match": {
                "content": {
                    "query": query,
                    "operator": "or",
                    "minimum_should_match": "30%",
                }
            }
        },
        "size": size,
    }

    try:
        results = await es.search(index=index_name, body=search_body)
        return results
    except Exception as e:
        print(f"Error searching for query '{query}': {e}")
        return None


async def evaluate_question(es, index_name, question_data, size=TOP_K):
    """Evaluate a single question against the index"""
    question_text = question_data["question"]
    ground_truth_ids = question_data["chunk_ids"]
    
    # Perform search
    results = await perform_search(es, index_name, question_text, size)
    
    if not results:
        return {
            "question": question_text,
            "ground_truth_chunk_ids": ground_truth_ids,
            "retrieved_chunks": [],
            "metrics": {},
            "found_ground_truth": False
        }
    
    # Extract retrieved chunks
    retrieved_chunks = []
    retrieved_ids = []
    found_ground_truth = False
    
    for i, hit in enumerate(results["hits"]["hits"]):
        chunk_id = hit["_source"].get("id", "unknown")
        retrieved_ids.append(chunk_id)
        
        chunk = {
            "id": chunk_id,
            "score": hit["_score"],
            "rank": i + 1,
            "file_name": hit["_source"].get("file_name", "N/A"),
            "content_preview": hit["_source"].get("content", "")[:200] + "..." if hit["_source"].get("content") else ""
        }
        retrieved_chunks.append(chunk)
        
        # Check if this is a ground truth chunk
        if chunk_id in ground_truth_ids:
            found_ground_truth = True
    
    # Generate evaluation metrics
    metrics = generate_evaluation_metrics(retrieved_ids, ground_truth_ids, size)
    
    return {
        "question": question_text,
        "ground_truth_chunk_ids": ground_truth_ids,
        "retrieved_chunks": retrieved_chunks,
        "metrics": metrics,
        "found_ground_truth": found_ground_truth
    }


# async def evaluate_all_questions(es, index_name, questions, size=TOP_K):
#     """Evaluate all questions against the index"""
#     tasks = [evaluate_question(es, index_name, q, size) for q in questions]
#     results = await asyncio.gather(*tasks)
    
#     # Calculate aggregate metrics
#     total_questions = len(questions)
#     questions_with_found_ground_truth = sum(1 for r in results if r["found_ground_truth"])
    
#     # Average metrics across all questions
#     avg_precision = {}
#     avg_recall = {}
#     avg_ndcg = {"ndcg@k": 0}
    
#     for result in results:
#         if not result["metrics"]:
#             continue
            
#         # Precision
#         for k, v in result["metrics"].get("precision_at_k", {}).items():
#             if k not in avg_precision:
#                 avg_precision[k] = 0
#             avg_precision[k] += v
            
#         # Recall
#         for k, v in result["metrics"].get("recall_at_k", {}).items():
#             if k not in avg_recall:
#                 avg_recall[k] = 0
#             avg_recall[k] += v
            
#         # NDCG
#         avg_ndcg["ndcg@k"] += result["metrics"].get("ndcg_at_k", {}).get("ndcg@k", 0)
    
#     # Compute averages
#     for k in avg_precision:
#         avg_precision[k] /= total_questions
    
#     for k in avg_recall:
#         avg_recall[k] /= total_questions
    
#     avg_ndcg["ndcg@k"] /= total_questions
    
#     aggregate_metrics = {
#         "total_questions": total_questions,
#         "questions_with_found_ground_truth": questions_with_found_ground_truth,
#         "hit_rate": questions_with_found_ground_truth / total_questions if total_questions > 0 else 0,
#         "avg_precision": avg_precision,
#         "avg_recall": avg_recall,
#         "avg_ndcg": avg_ndcg
#     }
    
#     return {
#         "results": results,
#         "aggregate_metrics": aggregate_metrics
#     }

async def evaluate_all_questions(es, index_name, questions, size=TOP_K):
    """Evaluate all questions against the index"""
    tasks = [evaluate_question(es, index_name, q, size) for q in questions]
    results = await asyncio.gather(*tasks)
    
    # Calculate aggregate metrics
    total_questions = len(questions)
    questions_with_found_ground_truth = sum(1 for r in results if r["found_ground_truth"])
    
    metrics_list = [result["metrics"] for result in results if result["metrics"]]
    
    final_metrics = average_metrics(metrics_list = metrics_list)
    
    return {
        "results": results,
        "aggregate_metrics": final_metrics
    }

    
async def save_results(evaluation_results, output_dir=RESULTS_DIR):
    """Save evaluation results to JSON files"""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual question results
    question_results_file = os.path.join(output_dir, "question_results.json")
    aggregate_metrics_file = os.path.join(output_dir, "aggregate_metrics.json")
    
    try:
        # Save individual question results
        async with aiofiles.open(question_results_file, "w") as f:
            await f.write(json.dumps(evaluation_results["results"], indent=2))
        print(f"Individual question results saved to {question_results_file}")
        
        # Save aggregate metrics
        async with aiofiles.open(aggregate_metrics_file, "w") as f:
            await f.write(json.dumps(evaluation_results["aggregate_metrics"], indent=2))
        print(f"Aggregate metrics saved to {aggregate_metrics_file}")
    except Exception as e:
        print(f"Error saving results: {e}")


# async def evaluate_questions_mode():
#     """Run in question evaluation mode"""
#     # Connect to Elasticsearch asynchronously
#     es = await connect_to_elasticsearch()
#     if not es:
#         return
    
#     # Define index name
#     index_name = "code_chunks"
    
#     # Check if index exists
#     index_exists = await check_existing_index(es, index_name)
#     if not index_exists:
#         print(f"Index '{index_name}' doesn't exist. Please create and populate it first.")
#         await es.close()
#         return
    
#     # Load questions from file (using static path)
#     questions = await load_questions()
#     if not questions:
#         await es.close()
#         return
    
#     # Set the number of results to retrieve
#     size = TOP_K
    
#     # Evaluate all questions
#     print(f"Evaluating {len(questions)} questions against the index...")
#     evaluation_results = await evaluate_all_questions(es, index_name, questions, size)
    
#     # Display summary metrics
#     metrics = evaluation_results["aggregate_metrics"]
#     print("\nEvaluation Summary:")
#     print(f"Total questions: {metrics['total_questions']}")
#     print(f"Questions with found ground truth: {metrics['questions_with_found_ground_truth']}")
#     print(f"Hit rate: {metrics['hit_rate']:.4f}")
#     print(f"Average precision@{size}: {metrics['avg_precision'].get(f'p@{size}', 0):.4f}")
#     print(f"Average recall@{size}: {metrics['avg_recall'].get(f'r@{size}', 0):.4f}")
#     print(f"Average NDCG@{size}: {metrics['avg_ndcg']['ndcg@k']:.4f}")
    
#     # Save results to specified directory
#     await save_results(evaluation_results)
    
#     # Close Elasticsearch connection
#     await es.close()

async def evaluate_questions_mode():
    """Run in question evaluation mode"""
    # Connect to Elasticsearch asynchronously
    es = await connect_to_elasticsearch()
    if not es:
        return
    
    # Define index name
    index_name = "code_chunks"
    
    # Check if index exists
    index_exists = await check_existing_index(es, index_name)
    if not index_exists:
        print(f"Index '{index_name}' doesn't exist. Please create and populate it first.")
        await es.close()
        return
    
    # Load questions from file (using static path)
    questions = await load_questions()
    if not questions:
        await es.close()
        return
    
    # Set the number of results to retrieve
    size = TOP_K
    
    # Evaluate all questions
    print(f"Evaluating {len(questions)} questions against the index...")
    evaluation_results = await evaluate_all_questions(es, index_name, questions, size)
    
    # Display summary metrics
    metrics = evaluation_results["aggregate_metrics"]

    # Display NDCG with the correct key format
    ndcg_key = f"NDCG@{size}"
    
    # Save results to specified directory
    await save_results(evaluation_results)
    
    # Close Elasticsearch connection
    await es.close()

def indexing_mode():
    """Run in indexing mode"""
    # Connect to Elasticsearch (synchronous for indexing)
    es = connect_to_elasticsearch_sync()

    # Define index name
    index_name = "code_chunks"

    # Check if index exists
    index_exists = check_existing_index_sync(es, index_name)

    if index_exists:
        # Delete the existing index if we're reindexing
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted")

    # Create new index
    create_index(es, index_name)

    # Load JSON data
    # json_file = "/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/cache_data/final_chunks_with_context.json"
    # json_file = "/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/cache_data/final_chunks.json"
    chunks = load_json_data(JSON_FILE)

    # Index documents
    indexed_count = index_documents(es, index_name, chunks)
    
    if indexed_count <= 0:
        print("No documents were indexed.")
        return


async def main():
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Define index name
    index_name = "code_chunks"
    
    # Connect to Elasticsearch asynchronously just to check if index exists
    es = await connect_to_elasticsearch()
    if not es:
        return
    
    # Check if index exists
    index_exists = await check_existing_index(es, index_name)
    await es.close()
    
    if index_exists:
        # Ask if we should reindex
        if input(f"Do you want to reindex {index_name}? (y/n): ").lower() == "y":
            # Run indexing mode (synchronous)
            indexing_mode()
    else:
        print(f"Index '{index_name}' doesn't exist. Creating it now.")
        # Run indexing mode (synchronous)
        indexing_mode()
    
    # After indexing (or if we decided not to reindex), run evaluation mode
    await evaluate_questions_mode()


if __name__ == "__main__":
    asyncio.run(main())