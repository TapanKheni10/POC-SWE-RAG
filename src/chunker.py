import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import uuid
import json
import os
import hashlib
import re
from google import genai
from google.genai import types
import time
import httpx
from settings import Config
from datetime import datetime, timedelta
from logging_util import loggers

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

example_function_code = """
async def execute(self, request_data: QueryEndPointRequest):
        '''
            Main execution function for processing query endpoint requests.
            Breaks down the request handling into smaller, focused functions.
        '''
        
        total_chunks = await self.index_repository.fetch_total_chunks(request_data.file_name)
        if total_chunks < request_data.top_k:
            raise HTTPException(
                status_code=400,
                detail=f"Top K value cannot be greater than the total number of chunks: {total_chunks}",
            )
        
        try:
            model = self.embeddings_provider_mapping[request_data.embedding_model]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail="Invalid embedding model. Please provide a valid model.",
            )
            
        if request_data.dimension not in self.model_to_dimensions[request_data.embedding_model]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dimension. Please provide a valid dimension for embedding model: {request_data.embedding_model}",
            )
            
        namespace_name, host = await self._get_namespace_and_host(request_data)
        questions_with_ground_turth_chunks = await self.index_repository.fetch_questions(request_data.file_name)
        
        batches = [questions_with_ground_turth_chunks[i:min(i + self.embedding_batch, len(questions_with_ground_turth_chunks))]
            for i in range(0, len(questions_with_ground_turth_chunks), self.embedding_batch)
        ]
        
        embedding_tasks = [self._generate_batch_embeddings(batch, request_data) for batch in batches]
        embeddings = await asyncio.gather(*embedding_tasks, return_exceptions=True)
        
        dense_embeddings = []
        for x in embeddings:
            dense_embeddings.extend(x)

        loggers['evaluation'].info(f"Total number of questions: {len(questions_with_ground_turth_chunks)}")
        loggers['evaluation'].info(f"Total number of dense embeddings: {len(dense_embeddings)}")

        
        s = time.time()
        tasks = [self._process_question(embedding, question, namespace_name, host, request_data) for question, embedding in zip(questions_with_ground_turth_chunks, dense_embeddings)]
        processed_questions = await asyncio.gather(*tasks)
        
        e = time.time()
        loggers['evaluation'].info(f"Total Time to execute evaluation metrices: {e-s}")
        if processed_questions:
            loggers['evaluation'].info(f"processed_questions: {len(processed_questions)}")
        
        questions, evaluation_metrics_list = zip(*processed_questions)

        s = time.time()
        with ThreadPoolExecutor() as executor:
            average_evaluation_metrics = executor.submit(
                self.average_metrics,
                evaluation_metrics_list,
                request_data.top_k,
            ).result()
        e = time.time()
        loggers['evaluation'].info(f"Total Time to execute evaluation metrices (Average): {e-s}")

        os.makedirs("results", exist_ok=True)
        with open("results/first_stage_retrieval.json", "w") as f:
            json.dump(list(questions), f, indent=4)
        
        with open("results/first_stage_evaluation.json", "w") as f:
            json.dump(average_evaluation_metrics, f, indent=4)
            
        return {"questions": questions, "evaluation_result" : average_evaluation_metrics}
        """

class RateLimiter:
    def __init__(self, rpm_limit=15):
        self.rpm_limit = rpm_limit
        self.request_timestamps = []
    
    def wait_if_needed(self):
        """Wait if we've exceeded the RPM limit"""
        now = datetime.now()
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < timedelta(minutes=1)]
        
        if len(self.request_timestamps) >= self.rpm_limit:
            oldest_timestamp = min(self.request_timestamps)
            wait_time = 60 - (now - oldest_timestamp).total_seconds()
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        
        self.request_timestamps.append(datetime.now())
        
rate_limiter = RateLimiter(rpm_limit=50)

def create_cache_for_file(google_client, file_path, code_bytes):
    
    system_prompt = """
    You are a specialized context extraction assistant designed to analyze document fragments and their relationship to whole documents. Your purpose is to create concise contextual descriptions that help enhance document retrieval systems.

    When presented with a whole document and a specific chunk from that document, your task is to:

    1. Carefully analyze both the complete document and the specific chunk.
    2. Identify the key themes, concepts, or information in the chunk.
    3. Determine how this chunk relates to and fits within the broader document structure.
    4. Generate a brief, information-dense contextual description (typically 1-3 sentences) that explains:
    - What the chunk contains
    - Where it fits in the document's overall narrative or structure
    - Why this chunk is significant within the document

    Your response should be ONLY the contextual description, with no additional commentary, explanations, or other text. The description should be concise but informative enough to improve search retrieval systems that need to match the chunk with relevant queries.

    Do not include phrases like "This chunk describes..." or "This section covers..." - simply provide the contextual information directly. Focus on content and context rather than document structure markers.
    """
    
    if google_client.caches.get(name = f"{file_path}_cache"):
        print(f"Cache for {file_path} already exists.")
        return google_client.caches.get(name = f"{file_path}_cache")
    
    rate_limiter.wait_if_needed()
        
    cache = google_client.caches.create(
        model = 'gemini-1.5-flash-002',
        config = types.CreateCachedContentConfig(
            display_name = f"{file_path}_cache",
            system_instruction = system_prompt,
            contents = [code_bytes],
            ttl = "300s"
        )
    )
    
    return cache
    
def call_llm_service(code_bytes, function_code):
    
    google_client = genai.Client(api_key = GEMINI_API_KEY)
    
    rate_limiter.wait_if_needed()
    
    response = google_client.models.generate_content(
        model = 'gemini-2.0-flash',
        contents = (f"""
            "<document> {code_bytes} </document> "
            Here is the chunk we want to situate within the whole document

            <chunk> {function_code} </chunk>
            
            Here's one example of how to do this:
            <chunk-example>
            {example_function_code}
            </chunk-example>
            
            <description-example>
            This function is the main entry point of the QueryUseCase class that orchestrates the entire vector search evaluation process. It validates input parameters against model constraints, retrieves questions with ground truth data, processes them in batches to generate embeddings, performs search operations (hybrid or dense vector), calculates retrieval evaluation metrics (precision, recall, NDCG, etc.), aggregates the results, and saves the evaluation output to JSON files for analysis. The function manages concurrency through asyncio for efficient processing of multiple search queries.
            </description-example>
            """
        ),
        config = types.GenerateContentConfig(
            temperature = 0.1,
        )
    )

    loggers["GeminiLogger"].info(f"usage: {response.usage_metadata}")
    loggers["GeminiLogger"].info(f"{response.text}")
    
    return response.text

def call_anthropic_service(code_bytes, function_code):
    
    headers = {
        "x-api-key": Config.ANTHROPIC_API_KEY,
        "anthropic-version": Config.ANTHROPIC_VERSION,
        "content-type": "application/json"
    }
    
    payload = {
        "model" : "claude-3-7-sonnet-20250219",
        "system" : [
            {
                "type": "text",
                "text": """
                    You are a specialized code analysis assistant that helps developers understand how specific code chunks fit within larger codebases. Your task is to provide brief, precise context for code snippets to improve search retrieval.

                    When presented with a full document and a specific chunk from that document, you will:
                    1. Analyze the relationship between the chunk and the overall document
                    2. Identify the chunk's purpose, functionality, and connections to other components
                    3. Focus only on factual technical context, not opinions or suggestions
                    4. Include key terminology that would help in search retrieval
                    
                    Never use phrases like "This code is" or "This function does" - simply state the context directly and succinctly. Prioritize clarity, precision, and search-relevance in your descriptions.
                    Always wrap the description in <description> xml tags. Only provide the description, no other text.
                """
            },
            {
                "type": "text",
                "text": f"<document> {code_bytes} </document>",
                "cache_control" : {
                    "type" : "ephemeral"
                }
            }
        ],
        "messages" : [
            {
                "role": "user",
                "content" : f"""
                    Here is the chunk we want to situate within the whole document

                    <chunk> {function_code} </chunk>
                """
            }
        ],
        "max_tokens" : 1024,
    }
    
    rate_limiter.wait_if_needed()
    timeout = httpx.Timeout(connect=60.0, read=300.0, write=300.0, pool=60.0)
    
    try:
        with httpx.Client(timeout = timeout) as client:
            response = client.post(f"{Config.ANTHROPIC_BASE_URL}messages", headers=headers, json=payload)
            response.raise_for_status()
            response_data = response.json()
            loggers["AnthropicLogger"].info(f"usage: {response_data['usage']}")
            return response_data["content"][0]["text"]
        
    except httpx.HTTPStatusError as e:
        loggers["AnthropicLogger"].error(f"httpx status error in anthropic api call : {str(e)} - {e.response.text}")
        if e.response.status_code == 429:
            print("Rate limit exceeded. Waiting for 60 seconds...")
            time.sleep(70)
            return call_anthropic_service(code_bytes, function_code)
        else:
            raise e
    
def extract_chunks(node, code_bytes, file_path, current_class=None, import_statements=None):
    if import_statements is None:
        import_statements = []

    chunks = []

    if node.type in ['import_statement', 'import_from_statement']:
        start = node.start_byte
        end = node.end_byte
        import_statements.append(code_bytes[start:end])
    elif node.type == 'class_definition':
        class_name_node = node.child_by_field_name('name')
        if class_name_node:
            current_class = code_bytes[class_name_node.start_byte:class_name_node.end_byte]
        for child in node.children:
            chunks.extend(extract_chunks(child, code_bytes, file_path, current_class, import_statements))
    elif node.type == 'function_definition':
        start = node.start_byte
        end = node.end_byte
        function_code = code_bytes[start:end]
        
        function_name_node = node.child_by_field_name('name')
        function_name = None
        if function_name_node:
            function_name = code_bytes[function_name_node.start_byte:function_name_node.end_byte]
        
        start_line = node.start_point[0] + 1  
        end_line = node.end_point[0] + 1  
        chunk_description = call_anthropic_service(code_bytes, function_code)
        
        pattern = r'<description>(.*?)</description>'
        
        matches = re.findall(pattern, chunk_description, re.DOTALL)
        if matches:
            chunk_description = matches[0]
        else:
            raise ValueError("No code found in the response")
        
        chunks.append({
            'code': function_code,
            'metadata': {
                'class': current_class,
                'function_name': function_name,
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line
            },
            "description": chunk_description
        })
    else:
        for child in node.children:
            chunks.extend(extract_chunks(child, code_bytes, file_path, current_class, import_statements))

    if node.type == 'module' and import_statements:
        combined_imports = '\n'.join(import_statements)
        start_line = node.start_point[0] + 1 
        end_line = start_line + len(import_statements) - 1
        chunk_description = call_anthropic_service(code_bytes, combined_imports)
        
        pattern = r'<description>(.*?)</description>'
        
        matches = re.findall(pattern, chunk_description, re.DOTALL)
        if matches:
            chunk_description = matches[0]
        else:
            raise ValueError("No code found in the response")
        
        chunks.insert(0, {
            'code': combined_imports,
            'metadata': {
                'class': None,
                'function_name': None,
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line
            },
            "description": chunk_description
        })

    return chunks

def chunk_codebase(file_path):
    try:
        tree = parse_code(file_path)
        root_node = tree.root_node
        
        with open(file_path, 'r', encoding='utf-8') as file:
            code_bytes = file.read()
        
        chunks = extract_chunks(root_node, code_bytes, file_path=file_path)
        return chunks
    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return []

def format_chunks_for_json(chunks):
    formatted_chunks = []
    for chunk in chunks:
        code = chunk['code'] + f"\n\n{chunk['description']}"
        metadata = chunk['metadata']
        file_path = metadata['file_path']
        file_name = os.path.basename(file_path)
        formatted_chunk = {
            "id": hashlib.sha256(chunk['code'].encode('utf-8')).hexdigest(),
            "file_path": file_path,
            "file_name": file_name,
            "start_line": metadata['start_line'],
            "end_line": metadata['end_line'],
            "content": code,
            "size": len(code),
            "parent-class": metadata['class'],
            "function_name": metadata['function_name']
        }
        formatted_chunks.append(formatted_chunk)
    return formatted_chunks

def save_chunks_to_json(chunks, output_file="chunks.json"):
    """
    Saves the formatted chunks to a JSON file.
    
    Parameters:
        chunks (list): List of formatted chunk dictionaries.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=4)
    
    print(f"Chunks saved to {output_file}")

def parse_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        code = file.read()
    tree = parser.parse(code.encode('utf-8'))
    return tree

def process_directory(directory_path):
    all_chunks = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                print(f"Processing: {file_path}")
                file_chunks = chunk_codebase(file_path)
                
                for chunk in file_chunks:
                    code = chunk['code']
                    metadata = chunk['metadata']
                    print(f"File: {metadata['file_path']}")
                    print(f"Class: {metadata['class']}")
                    print(f"Lines: {metadata['start_line']} to {metadata['end_line']}")
                    print(code)
                    loggers["ChunkLogger"].info(code)
                    print('=*='*20)
                    
                all_chunks.extend(file_chunks)
    return all_chunks
    
if __name__ == '__main__':
    
    directory_path = "/Users/tapankheni/Developer/POC-SWE-RAG/code_repo"
    all_chunks = process_directory(directory_path=directory_path)
    
    print(f"Extracted {len(all_chunks)} chunks from all files")
    
    formatted_chunks = format_chunks_for_json(all_chunks)
    save_chunks_to_json(formatted_chunks, "../cache_data/final_chunks_with_context.json")