import json
import re
import httpx
import pickle
from settings import Config
from pinecone import Pinecone
from logging_util import loggers
from typing import List
from evaluation import EvaluationService
import asyncio
from concurrent.futures import ThreadPoolExecutor

with open("../bm25_encoder.pkl", "rb") as f:
    bm25 = pickle.load(f)
    
groq_chat_url = "https://api.groq.com/openai/v1/chat/completions"
EMBEDDING_BATCH = 90
TOP_K = 10

class SearchDB:
    def __init__(self, pinecone_api_key: str = None, voyage_api_key: str = None):
        self.pinecone_api_key = pinecone_api_key
        if not pinecone_api_key:
            self.pinecone_api_key = Config.PINECONE_API_KEY
            
        self.voyage_api_key = voyage_api_key
        if not voyage_api_key:
            self.voyage_api_key = Config.VOYAGE_API_KEY
            
        self.embedding_model = "voyage-code-3"
        self.dimension = 1024
        self.embedding_url = "https://api.voyageai.com/v1/embeddings" 
        self.question_embedding = []
        self.question_sprase_embedding = []
        self.timeout = httpx.Timeout(
            connect=60.0,
            read=300.0,
            write=300.0,
            pool=60.0
        )
        
        self.query_url = "https://{}/query"
        self.pinecone_api_version = "2025-04"
        self.index_metric = "dotproduct"
        self.index_name = f"demo-{self.index_metric}"
        self.namespace_name = f"demo-namespace-with-context"
        self.index_host = ""
        self.pinecone_client = Pinecone(api_key = self.pinecone_api_key)
        self.semaphore = asyncio.Semaphore(50)
        
    async def voyageai_dense_embedding(self, inputs: List[dict], input_type: str = "query"):
        
        questions = [q['question'] for q in inputs]
        
        headers = {
            "Authorization": f"Bearer {self.voyage_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": questions,
            "model": self.embedding_model,
            "input_type": input_type,
            "output_dimension": self.dimension
        }
        
        try:
            async with httpx.AsyncClient(verify = False, timeout = self.timeout) as client:
                response = await client.post(self.embedding_url, headers=headers, json=data)
                response.raise_for_status()
                response = response.json()
                loggers["VoyageLogger"].info(f"voyage embedding usage: {response['usage']['total_tokens']} tokens")
                embedding_list = [item["embedding"] for item in response["data"]]
                return embedding_list
            
        except httpx.HTTPStatusError as e:
            loggers["VoyageLogger"].error(f"detail message of voyage embed failure: {e.response.text}")
            raise e
        
    async def pinecone_hybrid_search(
        self,
        dense_embedding: List[float],
        sparse_embedding: dict,
        top_k: int = TOP_K,
        alpha: float = 0.5,
        include_metadata: bool = True,
        filter_dict: dict = None
    ):
        
        async with self.semaphore:
            
            if not dense_embedding:
                raise ValueError("question embedding is empty")
            if not sparse_embedding:
                raise ValueError("question sparse embedding is empty")
            
            if not self.pinecone_client.has_index(name = self.index_name):
                raise ValueError(f"index {self.index_name} does not exist")
            
            self.index_host = self.pinecone_client.describe_index(name = self.index_name).get("host")

            headers = {
                "Api-Key": self.pinecone_api_key,
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-Pinecone-API-Version": self.pinecone_api_version,
            }
            
            hdense, hsparse = self.hybrid_scale(
                dense_embedding, sparse_embedding, alpha
            )
            
            payload = {
                "includeValues": False,
                "includeMetadata": include_metadata,
                "vector": hdense, 
                "sparseVector": {
                    "indices": hsparse.get(
                        "indices"
                    ),  
                    "values": hsparse.get(
                        "values"
                    ),  
                },
                "topK": top_k,
                "namespace": self.namespace_name,
            }
            
            if filter_dict:
                payload["filter"] = filter_dict
                
            query_url = self.query_url.format(self.index_host)
            
            try:
                async with httpx.AsyncClient(verify = False, timeout = self.timeout) as client:
                    response = await client.post(query_url, headers=headers, json=payload)
                    loggers["PineconeLogger"].info(f"pinecone hybrid query read units: {response.json()['usage']}")
                    return response.json()
                
            except httpx.HTTPStatusError as e:
                loggers["PineconeLogger"].error(f"detail message of pinecone search failure: {e.response.text}")
                raise e
        
    async def pinecone_dense_search(
        self,
        dense_embedding: List[float],
        top_k: int = TOP_K,
        include_metadata: bool = True,
        filter_dict: dict = None
    ): 
        
        async with self.semaphore:
            if not dense_embedding:
                raise ValueError("question embedding is empty")
            
            if not self.pinecone_client.has_index(name = self.index_name):
                raise ValueError(f"index {self.index_name} does not exist")
            
            self.index_host = self.pinecone_client.describe_index(name = self.index_name).get("host")
            
            headers = {
                "Api-Key": self.pinecone_api_key,
                "Content-Type": "application/json",
                "X-Pinecone-API-Version": self.pinecone_api_version,
            }
            
            payload = {
                "namespace": self.namespace_name,
                "topK": top_k,
                "vector": dense_embedding,
                "includeValues": False,
                "includeMetadata": include_metadata,
            }
            
            if filter_dict:
                payload["filter"] = filter_dict
                
            query_url = self.query_url.format(self.index_host)
            
            try:
                async with httpx.AsyncClient(verify = False, timeout = self.timeout) as client:
                    response = await client.post(query_url, headers=headers, json=payload)
                    # loggers["PineconeLogger"].info(f"pinecone dense query result: {response.json()}")
                    loggers["PineconeLogger"].info(f"pinecone hybrid query read units: {response.json()['usage']}")
                    return response.json()
                
            except httpx.HTTPStatusError as e:
                loggers["PineconeLogger"].error(f"detail message of pinecone search failure: {e.response.text}")
                raise e
        
    def hybrid_scale(self, dense, sparse, alpha: float):

        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")

        hsparse = {
            "indices": sparse["indices"],
            "values": [v * (1 - alpha) for v in sparse["values"]],
        }
        hdense = [v * alpha for v in dense]
        return hdense, hsparse
                
    def pinecone_sparse_embedding(self, inputs: list):
        try:
            sparse_vector = bm25.encode_documents(inputs)
            return sparse_vector
        
        except Exception as e:
            loggers["PineconeLogger"].error(f"detail message of pinecone sparse embedding failure: {e}")
            raise e
        
    def is_question_embedding_generated(self):
        return bool(self.question_embedding)
    
    def is_question_sparse_embedding_generated(self):
        return bool(self.question_sprase_embedding)
    
def save_results_to_json(results, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")
    
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
        loggers['evaluation'].error(f"Error calculating evaluation metrics: {e}")
        raise e

def average_metrics(metrics_list, top_k):
    
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
    
async def generate_evaluation_metrics(relevant_ids, ground_truth_ids, top_k=TOP_K):
    """Calculate various evaluation metrics for the search results."""
    
    async with asyncio.Semaphore(50):
        with ThreadPoolExecutor() as executor:
            evaluation_metrics = executor.submit(
                calculate_evaluation_metrics,
                relevant_ids,
                ground_truth_ids,
                top_k
            ).result()
            
        return evaluation_metrics
            
def generate_hypothetical_code(question: str):
    
    HyDE_SYSTEM_PROMPT = """
    You are an expert software engineer specializing in knowledge retrieval systems. 
    Your task is to generate hypothetical code that would be relevant to answering the given query, 
    which will be used for embedding-based retrieval.
    
    Instructions:
    1. Analyze the query carefully, focusing on key concepts and requirements.
    2. Generate concise, idiomatic code that represents a potential solution to the query.
    3. Include domain-specific terminology, method names, class names, and key concepts that would appear in ideal documentation for this topic.
    4. Incorporate relevant imports, class definitions, and function signatures that would be expected in high-quality results.
    5. Focus on capturing the essence of what would make a good search result, rather than a complete implementation.
    6. Use common libraries and frameworks appropriate for the task.
    7. The code should be detailed enough to generate meaningful embeddings but remain focused on the query's core concepts.
    
    Output format:
    - Wrap your entire code snippet in <code> and </code> XML tags.
    - Provide only the hypothetical code snippet without explanations.
    - Include comments that contain key terminology and concepts relevant to the query.
    - Ensure code is representative of what you would expect to find in a relevant document.
    - Use proper formatting, indentation, and naming conventions.
    """
    
    try:
        
        messages = [
            {
                "role": "system",
                "content": HyDE_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            }
        ]
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {Config.GROQ_API_KEY}",
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": messages
        }
    
        with httpx.Client(verify = False) as client:
            response = client.post(groq_chat_url, headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()["choices"][0]["message"]["content"]
            return response_data
        
    except httpx.HTTPStatusError as e:
        loggers["GroqLogger"].error(f"An HTTP error occurred while generating the response. {e.response.text}")
        raise e
        
    except Exception as e:
        loggers["GroqLogger"].error(f"An error occurred while generating the response.")
        raise e
    
async def main():
    
    with open("../questions_data/questions.json", "r") as f:
        questions = json.load(f)
        
    
    # hypothetical_code =  generate_hypothetical_code(question = question[0])
    
    # pattern = r'<code>(.*?)</code>'
    
    # matches = re.findall(pattern, hypothetical_code, re.DOTALL)
    # if matches:
    #     hypothetical_code = matches[0]
    # else:
    #     raise ValueError("No code found in the response")
    
    # print(f"hypothetical code: {hypothetical_code}")
    # print(f"=*="*20)
    
    questions_batches = [questions[i: min(i + EMBEDDING_BATCH, len(questions))]
        for i in range(0, len(questions), EMBEDDING_BATCH)
    ]
    
    obj = SearchDB()
        
    embedding_task = [obj.voyageai_dense_embedding(inputs = questions) for questions in questions_batches]
    
    embeddings_batches = await asyncio.gather(*embedding_task, return_exceptions = True)
    dense_embeddings = []
    for batch in embeddings_batches:
        if isinstance(batch, Exception):
            loggers["VoyageLogger"].error(f"Error in embedding batch: {batch}")
            continue
        dense_embeddings.extend(batch)
        
    loggers["MainLogger"].info(f"total number of questions: {len(questions)}")
    loggers["MainLogger"].info(f"total number of embeddings: {len(dense_embeddings)}")
    
    is_embedding_generated = True if dense_embeddings else False
    print(f"is dense embedding generated: {is_embedding_generated}")
    print(f"=*="*20)

    sparse_embeddings = []
    for question in questions:
        sparse_embeddings.append(obj.pinecone_sparse_embedding(inputs = question['question']))

    is_sparse_embedding_generated = True if sparse_embeddings else False
    print(f"is sparse embedding generated: {is_sparse_embedding_generated}")
    print(f"=*="*20)

    search_tasks = [obj.pinecone_hybrid_search(dense_embedding = dense_embedding, sparse_embedding = sparse_embedding) 
        for dense_embedding, sparse_embedding in zip(dense_embeddings, sparse_embeddings)
    ]
    
    print(f"total number of search tasks: {len(search_tasks)}")
    print(f"=*="*20)
    
    # search_tasks = [obj.pinecone_dense_search(dense_embedding = dense_embedding)
    #     for dense_embedding in dense_embeddings
    # ]
    
    # print(f"total number of search tasks: {len(search_tasks)}")
    # print(f"=*="*20)
    
    search_results = await asyncio.gather(*search_tasks, return_exceptions = True)
    
    print(f"total number of search results: {len(search_results)}")
    print(f"=*="*20)
    
    formated_search_results = []
    retrieved_ids = []
    
    for i, result in enumerate(search_results):
        retrieved = []
        ids = []
        for match in result["matches"]:
            retrieved.append({
                "id": match["id"],
                "score": match["score"],
                "content": match["metadata"]["content"]
            })
            ids.append(match["id"])
            
        formated_search_results.append({
            "question": questions[i]['question'],
            "ground_truth": questions[i]['chunk_ids'],
            "retrieved": retrieved
        })
        retrieved_ids.append(ids)
        
    print(f"total number of formated search results: {len(formated_search_results)}")
    print(f"=*="*20)
    
    save_results_to_json(formated_search_results, "../results/first_stage/hybrid_with_context.json")
    
    evaluation_tasks = [generate_evaluation_metrics(relevant_ids = retrieved, ground_truth_ids = question["chunk_ids"])
        for retrieved, question in zip(retrieved_ids, questions)
    ]
    
    print(f"total number of evaluation tasks: {len(evaluation_tasks)}")
    print(f"=*="*20)
    
    evaluation_metrices = await asyncio.gather(*evaluation_tasks, return_exceptions = True)
    
    print(f"total number of evaluation metrics: {len(evaluation_metrices)}")
    print(f"=*="*20)
    
    with ThreadPoolExecutor() as executor:
        average_evaluation_metrics = executor.submit(
            average_metrics,
            evaluation_metrices,
            TOP_K
        ).result()
    
    save_results_to_json(average_evaluation_metrics, "../evaluation/first_stage/hybrid_with_context.json")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
    