from settings import Config
import httpx
from logging_util import loggers
from pinecone import Pinecone
from typing import List, Dict, Any
import time
import json
from datetime import datetime
import asyncio
import pickle
import hashlib

with open("bm25_encoder.pkl", "rb") as f:
    bm25 = pickle.load(f)

class VectorDB:
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
        self.embeddings = []
        self.sparse_embeddings = []
        self.embedding_cache_file = "embedding_cache_with_context.json"
        self.timeout = httpx.Timeout(
            connect=60.0,
            read=300.0,
            write=300.0,
            pool=60.0
        )
        self.pinecone_api_version = "2025-04"
        self.create_index_url = "https://api.pinecone.io/indexes"
        self.upsert_index_url = "https://{}/vectors/upsert"
        self.index_metric = "dotproduct"
        self.index_name = f"demo-{self.index_metric}"
        self.pinecone_client = Pinecone(api_key = self.pinecone_api_key)
        self.index_host = ""
        self.namespace_name = f"demo-namespace-with-context"
        
    async def voyageai_dense_embedding(self, inputs: list, input_type: str = "document", use_cache: bool = True):
        
        if use_cache and self.load_embeddings_from_cache(inputs):
            loggers["VoyageLogger"].info(f"Loaded embeddings from cache for {len(inputs)} inputs")
            return self.embeddings
        
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
            # async with httpx.AsyncClient(verify = False, timeout = self.timeout) as client:
            #     response = await client.post(self.embedding_url, headers=headers, json=data)
            #     response.raise_for_status()
            #     response = response.json()
            #     loggers["VoyageLogger"].info(f"pinecone hosted embedding model tokens usage: {response['usage']}")
            #     embedding_list = [item["embedding"] for item in response["data"]]
            #     self.embeddings = embedding_list
                
            #     self.save_embeddings_to_cache(inputs, embedding_list)
                
            #     return embedding_list
            pass
            
        except httpx.HTTPStatusError as e:
            loggers["VoyageLogger"].error(f"detail message of voyage embed failure: {e.response.text}")
            raise e
        
    async def create_pinecone_index(self):
        if not self.pinecone_client.has_index(name = self.index_name):
            index_data = {
                "name" : self.index_name,
                "dimension" : self.dimension,
                "metric" : self.index_metric,
                "spec": {"serverless": {"cloud": "aws", "region": "us-east-1"}},
            }
            
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Api-Key": self.pinecone_api_key,
                "X-Pinecone-API-Version": self.pinecone_api_version,
            }
            
            try:
                async with httpx.AsyncClient(verify = False) as client:
                    response = await client.post(self.create_index_url, headers=headers, json=index_data)
                    response.raise_for_status()
                    
                    time.sleep(2)
                    if self.pinecone_client.has_index(name = self.index_name):
                        self.index_host = self.pinecone_client.describe_index(name = self.index_name).get("host")
                        
                    loggers["PineconeLogger"].info(f"pinecone index creation successful.")
                    return response.json()
                
            except httpx.HTTPStatusError as e:
                loggers["PineconeLogger"].error(f"detail message of pinecone index creation failure: {e.response.text}")
                raise e
            
        else:
            loggers["PineconeLogger"].info(f"index already exists.")
            self.index_host = self.pinecone_client.describe_index(name = self.index_name).get("host")

    async def upsert_to_pinecone(self, chunks: List[Dict[str, Any]]):
        
        if not self.embeddings:
            raise ValueError("Embeddings are not available. Please generate embeddings first.")
        if not self.sparse_embeddings:
            raise ValueError("Sparse embeddings are not available. Please generate sparse embeddings first.")
        if not self.index_host:
            raise ValueError("Pinecone index host is not available. Please create the index first.")
        if not chunks:
            raise ValueError("No chunks to upsert.")
        
        results = []
        for i, chunk in enumerate(chunks):
            metadata = {key: value for key, value in chunk.items() if key != "id" and value is not None}
            
            metadata["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            result = {
                "id": chunk["id"],
                "values": self.embeddings[i],
                "metadata": metadata,
                "sparse_values": {
                    "indices" : self.sparse_embeddings[i]["indices"],
                    "values" : self.sparse_embeddings[i]["values"]
                }
                
            }
            results.append(result)
            
        headers = {
            "Api-Key": self.pinecone_api_key,
            "Content-Type": "application/json",
            "X-Pinecone-API-Version": self.pinecone_api_version,
        }
        
        upsert_url = self.upsert_index_url.format(self.index_host)
        
        payload = {"vectors" : results, "namespace" : self.namespace_name}
        
        try:
            async with httpx.AsyncClient(verify = False, timeout = self.timeout) as client:
                response = await client.post(url=upsert_url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            loggers["PineconeLogger"].error(f"detail message of pinecone upsert failure: {e.response.text}")
            raise e
        
    def pinecone_sparse_embedding(self, inputs: list):
        try:
            sparse_vector = bm25.encode_documents(inputs)
            self.sparse_embeddings = sparse_vector
            return sparse_vector
        
        except Exception as e:
            loggers["PineconeLogger"].error(f"detail message of pinecone sparse embedding failure: {e}")
            raise e
        
    def save_embeddings_to_cache(self, inputs: list, embeddings: list):
        try:
            cache_data = {}
            
            for i, input_text in enumerate(inputs):
                input_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
                cache_data[input_hash] = {
                    "embedding" : embeddings[i]
                }
                
            with open(self.embedding_cache_file, 'w') as f:
                json.dump(cache_data, f)
                
            loggers["VoyageLogger"].info(f"Embeddings saved to cache file: {self.embedding_cache_file}")
            
        except Exception as e:
            loggers["VoyageLogger"].error(f"Failed to save embeddings to cache: {str(e)}")

    def load_embeddings_from_cache(self, inputs: list):
        try:
            try:
                with open(self.embedding_cache_file, 'r') as f:
                    cache_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                loggers["VoyageLogger"].error(f"Cache file not found or is not a valid JSON file.")
                return False
            
            embeddings = []
            for input_text in inputs:
                input_hash = hashlib.sha256(input_text.encode('utf-8')).hexdigest()
                if input_hash not in cache_data:
                    loggers["VoyageLogger"].info(f"Input not found in cache: {input_hash}")
                    return False
                
                cached_item = cache_data[input_hash]
                
                if "embedding" not in cached_item:
                    loggers["VoyageLogger"].info(f"Embedding not found in cache for input: {input_hash}")
                    return False
                
                embeddings.append(cached_item["embedding"])
                
            self.embeddings  = embeddings
            return True
            
        except Exception as e:
            loggers["VoyageLogger"].error(f"Failed to load embeddings from cache: {str(e)}")
            return False
        
    def is_embedding_generated(self):
        return bool(self.embeddings), len(self.embeddings)
    
    def is_sparse_embedding_generated(self):
        return bool(self.sparse_embeddings), len(self.sparse_embeddings)


def load_json(file_path: str):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        
async def main():
    chunk_file_path = "final_chunks_with_context.json"
    chunks = load_json(chunk_file_path)
    
    print(f"type of chunks: {type(chunks)}")
    print(f"=*="*20)
    print(f"type of chunks[0]: {type(chunks[0])}")
    print(f"=*="*20)
    print(f"length of chunks: {len(chunks)}")
    print(f"=*="*20)
    if not isinstance(chunks, list):
        raise ValueError("The loaded data is not a list.")
    
    content = []
    for chunk in chunks:
        content.append(chunk["content"])
    
    obj = VectorDB()
    
    await obj.voyageai_dense_embedding(
        inputs = content,
        use_cache = True
    )
    
    obj.pinecone_sparse_embedding(
        inputs = content
    )
    
    is_embedding_generated, total_embeddings = obj.is_embedding_generated()
    print(f"is embedding generated: {is_embedding_generated}")
    print(f"=*="*20)
    print(f"total embeddings: {total_embeddings}")
    print(f"=*="*20)
    
    is_sparse_embedding_generated, total_sparse_embeddings = obj.is_sparse_embedding_generated()
    print(f"is embedding generated: {is_sparse_embedding_generated}")
    print(f"=*="*20)
    print(f"total embeddings: {total_sparse_embeddings}")
    print(f"=*="*20)
    
    await obj.create_pinecone_index()
    
    await obj.upsert_to_pinecone(chunks = chunks)
    
if __name__ == "__main__":
    asyncio.run(main())
    