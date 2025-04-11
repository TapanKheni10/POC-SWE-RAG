import json
import os

from elasticsearch import Elasticsearch


def connect_to_elasticsearch(
    host="http://localhost:9200", username="elastic", password="zKRURlMb"
):
    """Connect to Elasticsearch server"""
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


def check_existing_index(es, index_name):
    """Check if index exists and list available indices"""
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


def perform_search(es, index_name, query, size=15):
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

    # Check document count before search
    count = es.count(index=index_name)
    print(f"Document count before search: {count['count']}")

    if count["count"] > 0:
        results = es.search(index=index_name, body=search_body)

        # Convert ObjectApiResponse to dictionary
        results_dict = dict(results)

        # Save results to file
        with open("results.json", "w") as f:
            print("Writing results to POC-SWE-RAG/sparse/results.json")
            json.dump(results_dict, f, indent=2)

        return results
    else:
        print("\nSkipping search because no documents are indexed.")
        return None


def display_results(results):
    """Display search results in a readable format"""
    if not results:
        return

    print("\nTop Search Results:\n")
    if results["hits"]["total"]["value"] > 0:
        for hit in results["hits"]["hits"]:
            print(
                f"File: {hit['_source'].get('file_name', 'N/A')} - Score: {hit['_score']}"
            )
            content = hit["_source"].get("content", "")
            if content:
                print(f"Snippet: {content[:200]}...\n")
            else:
                print("No content available\n")
    else:
        print("No results found. Try adjusting your search parameters.")


def main():
    # Connect to Elasticsearch
    es = connect_to_elasticsearch()

    # Define index name
    index_name = "code_chunks"

    # Check if index exists
    index_exists = check_existing_index(es, index_name)

    if index_exists:
        # Ask if we should reindex
        if input(f"Do you want to reindex {index_name}? (y/n): ").lower() != "y":
            print("Using existing index for search...")
            # Get query from user input
            query = input("Enter your search query: ")
            results = perform_search(es, index_name, query)
            display_results(results)
            return

        # Delete the existing index if we're reindexing
        es.indices.delete(index=index_name)
        print(f"Index '{index_name}' deleted")

    # Create new index
    create_index(es, index_name)

    # Load JSON data
    # json_file = "/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/json/final_chunks.json"
    json_file = "/Users/harshabajaj/Desktop/SWE-bench/POC-SWE-RAG/cache_data/final_chunks_with_context.json"
    chunks = load_json_data(json_file)

    # Index documents
    indexed_count = index_documents(es, index_name, chunks)

    # Perform search if documents were indexed
    if indexed_count > 0:
        # Get query from user input
        query = input("Enter your search query: ")
        results = perform_search(es, index_name, query)
        display_results(results)
    else:
        print("No documents were indexed, skipping search.")


if __name__ == "__main__":
    main()