import json
import re
import httpx
import pickle
from settings import Config
from pinecone import Pinecone
from logging_util import loggers

with open("bm25_encoder.pkl", "rb") as f:
    bm25 = pickle.load(f)
    
groq_chat_url = "https://api.groq.com/openai/v1/chat/completions"

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
        
    async def voyageai_dense_embedding(self, inputs: list, input_type: str = "query"):
        
        headers = {
            "Authorization": f"Bearer {self.voyage_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "input": inputs,
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
                self.question_embedding = embedding_list
                
                return embedding_list
            
        except httpx.HTTPStatusError as e:
            loggers["VoyageLogger"].error(f"detail message of voyage embed failure: {e.response.text}")
            raise e
        
    async def pinecone_hybrid_search(
        self,
        top_k: int,
        alpha: float,
        include_metadata: bool,
        filter_dict: dict = None
    ):
        
        if not self.question_embedding:
            raise ValueError("question embedding is empty")
        if not self.question_sprase_embedding:
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
            self.question_embedding[0], self.question_sprase_embedding[0], alpha
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
            self.question_sprase_embedding = sparse_vector
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
    
    question = ["""I'm working with a microservice architecture and need to implement distributed tracing. Can you explain how to properly set up request context propagation between services with FastAPI middleware and show me the best practices for trace ID generation?"""]
    
    hypothetical_code =  generate_hypothetical_code(question = question[0])
    
    pattern = r'<code>(.*?)</code>'
    
    matches = re.findall(pattern, hypothetical_code, re.DOTALL)
    if matches:
        hypothetical_code = matches[0]
    else:
        raise ValueError("No code found in the response")
    
    print(f"hypothetical code: {hypothetical_code}")
    print(f"=*="*20)
    
    obj = SearchDB()
    await obj.voyageai_dense_embedding(inputs = question)
    obj.pinecone_sparse_embedding(inputs = question)
    print('=*='*20)
    
    is_embedding_generated = obj.is_question_embedding_generated()
    print(f"is dense embedding generated: {is_embedding_generated}")
    print(f"=*="*20)
    
    is_sparse_embedding_generated = obj.is_question_sparse_embedding_generated()
    print(f"is sparse embedding generated: {is_sparse_embedding_generated}")
    print(f"=*="*20)
    
    results = await obj.pinecone_hybrid_search(
        top_k = 5,
        alpha = 0.3,
        include_metadata = True 
    )
    print(results.keys())
    print(f"=*="*20)
    
    retrieved = []
    for match in results["matches"]:
        retrieved.append({
            "score": match["score"],
            "content": match["metadata"]["content"],
            "start_line": match["metadata"]["start_line"],
            "end_line": match["metadata"]["end_line"],
            "file_name": match["metadata"]["file_name"],
            "file_path": match["metadata"]["file_path"],
        })
        
        print(f"file_name: {match['metadata']['file_name']}")
        print(f"{match['metadata']['content']}")
        print(f"=*="*20)
        
    # save_results_to_json(retrieved, "retrieved_results_with_context.json")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
    
    