import tree_sitter_python as tspython
from tree_sitter import Language, Parser
import uuid
import json
import os
from google import genai
from google.genai import types
import time
from datetime import datetime, timedelta
from logging_util import loggers

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)
GEMINI_API_KEY = "AIzaSyA79zHVtbEZ2AzaW8GlRsGoNCWjxnSDuVc"

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
        
rate_limiter = RateLimiter(rpm_limit=15)

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
        model = 'gemini-1.5-flash-002',
        contents = (f"""
            "<document> {code_bytes} </document> "
            Here is the chunk we want to situate within the whole document

            <chunk> {function_code} </chunk>

            Please give a short succinct context to situate this chunk within
            the overall document for the purposes of improving search retrieval
            of the chunk. Answer only with the succinct context and nothing else.
            """
        ),
        config = types.GenerateContentConfig(
            temperature = 0.1,
        )
    )

    loggers["GeminiLogger"].info(f"usage: {response.usage_metadata}")
    loggers["GeminiLogger"].info(f"{response.text}")
    
    return response.text

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
        # chunk_description = call_llm_service(code_bytes, function_code)
        
        # function_code += f"\n{chunk_description}"
        chunks.append({
            'code': function_code,
            'metadata': {
                'class': current_class,
                'function_name': function_name,
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line
            }
        })
    else:
        for child in node.children:
            chunks.extend(extract_chunks(child, code_bytes, file_path, current_class, import_statements))

    if node.type == 'module' and import_statements:
        combined_imports = '\n'.join(import_statements)
        start_line = node.start_point[0] + 1 
        end_line = start_line + len(import_statements) - 1
        # chunk_description = call_llm_service(code_bytes, combined_imports)
        
        # combined_imports += f"\n{chunk_description}"
        chunks.insert(0, {
            'code': combined_imports,
            'metadata': {
                'class': None,
                'function_name': None,
                'file_path': file_path,
                'start_line': start_line,
                'end_line': end_line
            }
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
        code = chunk['code']
        metadata = chunk['metadata']
        file_path = metadata['file_path']
        file_name = os.path.basename(file_path)
        formatted_chunk = {
            "id": str(uuid.uuid4()),
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
    
    directory_path = "/Users/tapankheni/Developer/POC-SWE-RAG/observe_traces"
    all_chunks = process_directory(directory_path=directory_path)
    
    print(f"Extracted {len(all_chunks)} chunks from all files")
    
    formatted_chunks = format_chunks_for_json(all_chunks)
    save_chunks_to_json(formatted_chunks, "final_chunks.json")