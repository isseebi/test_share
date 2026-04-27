import os
from dotenv import load_dotenv
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Load environment variables
load_dotenv()

# Configure LLM and Embedding
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")

# Use Ollama with specified models
llm = Ollama(model="qwen3.5:9b", request_timeout=360.0)
embed_model = OllamaEmbedding(model_name="qwen3-embedding:8b")

Settings.llm = llm
Settings.embed_model = embed_model

def main():
    # 1. Connect to Memgraph Store
    print("Connecting to Memgraph...")
    graph_store = MemgraphPropertyGraphStore(
        username=os.getenv("MEMGRAPH_USERNAME", ""),
        password=os.getenv("MEMGRAPH_PASSWORD", ""),
        url=os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    )

    # 2. Reconstruct Index from existing store
    print("Initializng Graph RAG index...")
    index = PropertyGraphIndex.from_existing(
        property_graph_store=graph_store
    )

    # 3. Create Query Engine
    # You can combine vector search and graph search here
    query_engine = index.as_query_engine(
        include_text=True, # Include text chunks in retrieval
        streaming=True
    )

    print("\n--- Graph RAG Query Engine Ready ---")
    print("Type 'exit' to quit.")
    
    while True:
        query_str = input("\nQuery: ")
        if query_str.lower() in ["exit", "quit"]:
            break
        
        print("\nResponse: ", end="", flush=True)
        response = query_engine.query(query_str)
        response.print_response_stream()
        print()

if __name__ == "__main__":
    main()
