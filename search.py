import httpx
import json
import pickle
from settings import Config
from pinecone import Pinecone
from logging_util import loggers

with open("bm25_encoder.pkl", "rb") as f:
    bm25 = pickle.load(f)

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
    
async def main():
    
    question = ["""How does the LangfuseClient interact with the Langfuse service to create generation, span, and trace objects, 
                and how does it handle metadata and cost details during operations such as vector database queries, embedding, and reranking?"""]
    
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
    
    