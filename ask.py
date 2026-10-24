import sys
from src.ingestion.embedder import FastEmbedEmbedder
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever, format_context
from src.retrieval.generator import generate_answer

def main():
    if len(sys.argv) < 2:
        print("Usage: python ask.py 'Your question here'")
        return

    query = sys.argv[1]
    
    # 1. Initialize retrieval components
    embedder = FastEmbedEmbedder()
    store = VectorStore("chroma_db", embedding_model=embedder.name)
    retriever = Retriever(embedder, store)

    # 2. Search the vector database
    print(f"Searching for: '{query}'...")
    retrieval_result = retriever.retrieve(query)
    
    # 3. Format context and generate answer
    context = format_context(retrieval_result)
    print("Generating answer...\n")
    
    answer = generate_answer(query, context)
    print(answer)

if __name__ == "__main__":
    main()
