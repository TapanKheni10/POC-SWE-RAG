import json
import time
import httpx
from logging_util import loggers
from settings import Config
from search import generate_evaluation_metrics, average_metrics
from concurrent.futures import ThreadPoolExecutor

TOP_N = 5

def save_results_to_json(results, output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")
        
class ReRanker:
    def __init__(self, cohere_api_key: str = None):
        self.cohere_api_key  = None
        if not cohere_api_key:
            self.cohere_api_key = Config.COHERE_API_KEY
    
        self.model = "rerank-v3.5"
        self.cohere_rerank_url = "https://api.cohere.com/v2/rerank"
        self.input_path = "../results/first_stage/hybrid_without_context.json"
        self.question_path = "../questions_data/questions.json"
        self.result_output_path = "../results/second_stage/hybrid_without_context.json"
        self.evaluation_output_path = "../evaluation/second_stage/hybrid_without_context.json"
        self.semaphores = asyncio.Semaphore(15)
        self.timeout = httpx.Timeout(connect=60.0, read=120.0, write=120.0, pool=60.0)
        
    async def cohere_reranker(self, query: str, documents: list):
        async with self.semaphores:
            
            headers = {
                "content-type": "application/json",
                "accept": "application/json",
                "Authorization": f"bearer {self.cohere_api_key}",
            }

            payload = {
                "model": self.model,
                "query": query,
                "top_n": TOP_N,
                "documents": documents,
            }
            
            try:
                async with httpx.AsyncClient(verify=False, timeout = self.timeout) as client:
                    response = await client.post(
                        self.cohere_rerank_url,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    loggers["MainLogger"].info("reranking done by cohere")
                    loggers["CohereLogger"].info(f"Reranking model hosted by Cohere tokens usage : {response.json().get('meta',{}).get('billed_units', {})}")
                    return response.json()
                
            except httpx.HTTPStatusError as e:
                loggers["CohereLogger"].error(f"httpx status error in anthropic api call : {str(e)} - {e.response.text}")
                if e.response.status_code == 429:
                    print("Rate limit exceeded. Waiting for 60 seconds...")
                    time.sleep(70)
                    return self.cohere_reranker(query, documents)
                else:
                    raise e

    async def rerank(self):
        
        with open(self.input_path, "r") as f:
            data = json.load(f)
            
        with open(self.question_path, "r") as f:
            questions_data = json.load(f)
        
        questions = []
        relevant_docs = []
        for item in data:
            questions.append(item["question"])
            relevant_docs.append([doc["content"] for doc in item["retrieved"]])
            
        print(f"Number of queries: {len(questions)}")
        print(f"Number of documents: {len(relevant_docs)}")
        print("=*="*20)
        
        reranking_tasks = [self.cohere_reranker(query = question, documents = docs)
            for question, docs in zip(questions, relevant_docs)
        ]
        
        print(f'reranked tasks: {len(reranking_tasks)}')
        print("=*="*20)
        
        rerank_results = await asyncio.gather(*reranking_tasks, return_exceptions = True)
        
        print(f"rerank results: {len(rerank_results)}")
        print("=*="*20)
        
        formatted_results = []
        reranked_ids = []
        
        for i, result in enumerate(rerank_results):
            if isinstance(result, Exception):
                loggers["MainLogger"].error(f"Error in reranking task {i}: {result}")
                continue
            
            reranked_docs = []
            ids = []
            for item in result["results"]:
                index = item["index"]
                score = item["relevance_score"]
                _id = data[i]["retrieved"][index]["id"]
                content = data[i]["retrieved"][index]["content"]
                
                reranked_docs.append({
                    "id" : _id,
                    "score" : score,
                    "content" : content,
                })
                ids.append(_id)
            
            formatted_results.append({
                "question": questions[i],
                "ground_truth": questions_data[i]["chunk_ids"],
                "reranked": reranked_docs,
            })
            reranked_ids.append(ids)
            
        print(f"formatted results: {len(formatted_results)}")
        print("=*="*20)
        print(f"reranked ids: {len(reranked_ids)}")
        print("=*="*20)
        
        save_results_to_json(formatted_results, self.result_output_path)
        
        evaluation_tasks = [generate_evaluation_metrics(relevant_ids = reranked, ground_truth_ids = question["chunk_ids"], top_k = TOP_N)
            for reranked, question in zip(reranked_ids, questions_data)
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
                TOP_N,
            ).result()
    
        save_results_to_json(average_evaluation_metrics, self.evaluation_output_path)
        
async def main():
    
    reranker = ReRanker()
    
    await reranker.rerank()
    print("Reranking completed successfully.")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())