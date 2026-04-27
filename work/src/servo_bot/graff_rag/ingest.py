import os
from dotenv import load_dotenv
from llama_index.core import PropertyGraphIndex, SimpleDirectoryReader, Settings
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

# Load environment variables
load_dotenv()

# Configure LLM and Embedding
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in .env")

# Use Ollama with specified models
llm = Ollama(model="qwen3.5:0.8b", request_timeout=3600.0)
embed_model = OllamaEmbedding(model_name="qwen3-embedding:8b")

Settings.llm = llm
Settings.embed_model = embed_model
Settings.chunk_size = 512
def main():
    # 1. Load documents
    print("Loading documents from doc.pdf...")
    documents = SimpleDirectoryReader(input_files=["doc.pdf"]).load_data()

    # 2. Setup Memgraph Store
    print("Connecting to Memgraph...")
    graph_store = MemgraphPropertyGraphStore(
        username=os.getenv("MEMGRAPH_USERNAME", ""),
        password=os.getenv("MEMGRAPH_PASSWORD", ""),
        url=os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
    )

    # 3. Define Schema and Extractor
    # Based on initial inspection, the content is about ONE PIECE
    entities = ["Character", "Group", "Location", "Vessel", "Ability", "Event", "Concept"]
    relations = ["BELONGS_TO", "LOCATED_IN", "PART_OF", "ALLIED_WITH", "FOUGHT_WITH", "LEADER_OF", "MEMBER_OF", "OWNED_BY"]
    
    kg_extractor = [
        SchemaLLMPathExtractor(
            llm=llm,
            possible_entities=entities,
            possible_relations=relations,
            strict=False, # Allow LLM to discover more if needed
        )
    ]

    # 4. Create Property Graph Index
    print("Extracting entities and relationships (this may take a few minutes)...")
    index = PropertyGraphIndex.from_documents(
        documents,
        property_graph_store=graph_store,
        kg_extractors=kg_extractor,
        show_progress=True
    )
    
    # Save the index structure (metadata) locally if needed, 
    # but the graph data is already in Memgraph.
    # index.storage_context.persist(persist_dir="./storage")
    
    print("Ingestion complete! You can now visualize the graph in Memgraph Lab (http://localhost:3000).")

if __name__ == "__main__":
    main()
