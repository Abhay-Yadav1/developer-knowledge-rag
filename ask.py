import sys
from src.ingestion.embedder import FastEmbedEmbedder
from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever, format_context
from src.retrieval.generator import generate_answer
import re
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
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    print("\nSources:")
    for n in cited:
       if 1 <= n <= len(retrieval_result.chunks):
           c = retrieval_result.chunks[n - 1]
           print(f"[{n}] {c.file_path} L{c.start_line}-L{c.end_line}")

if __name__ == "__main__":
    main()
