import argparse
import numpy as np
from tqdm import tqdm

# Import from Day 1 and Day 2
from day1_main import read_repo_data
from day2_main import chunk_documents


def create_text_index(chunks: list[dict]):
    """
    Create a text search index using minsearch.

    Args:
        chunks: List of chunk dictionaries with 'chunk', 'title', 'filename' keys

    Returns:
        Fitted minsearch Index
    """
    from minsearch import Index

    index = Index(
        text_fields=["chunk", "title", "description", "filename"],
        keyword_fields=[]
    )
    index.fit(chunks)
    return index


def text_search(index, query: str, num_results: int = 5) -> list[dict]:
    """
    Perform text (lexical) search.

    Args:
        index: minsearch Index
        query: Search query string
        num_results: Number of results to return

    Returns:
        List of matching chunks
    """
    results = index.search(query, num_results=num_results)
    return results


def create_embeddings(chunks: list[dict], model_name: str = 'multi-qa-distilbert-cos-v1'):
    """
    Create vector embeddings for all chunks.

    Args:
        chunks: List of chunk dictionaries
        model_name: Sentence transformer model name

    Returns:
        Tuple of (embedding_model, embeddings_array)
    """
    from sentence_transformers import SentenceTransformer

    print(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_name)

    print("Creating embeddings...")
    embeddings = []
    for chunk in tqdm(chunks):
        text = chunk.get('chunk', chunk.get('content', ''))
        v = embedding_model.encode(text)
        embeddings.append(v)

    return embedding_model, np.array(embeddings)


def vector_search(query: str, embedding_model, embeddings: np.ndarray,
                  chunks: list[dict], num_results: int = 5) -> list[dict]:
    """
    Perform vector (semantic) search using embeddings.

    Args:
        query: Search query string
        embedding_model: SentenceTransformer model
        embeddings: Pre-computed embeddings array
        chunks: Original chunk dictionaries
        num_results: Number of results to return

    Returns:
        List of matching chunks with similarity scores
    """
    # Encode the query
    query_embedding = embedding_model.encode(query)

    # Compute cosine similarity (dot product on normalized vectors)
    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = query_embedding / np.linalg.norm(query_embedding)

    # Compute similarities
    similarities = np.dot(embeddings_norm, query_norm)

    # Get top results
    top_indices = np.argsort(similarities)[::-1][:num_results]

    results = []
    for idx in top_indices:
        chunk = chunks[idx].copy()
        chunk['similarity_score'] = float(similarities[idx])
        results.append(chunk)

    return results


def hybrid_search(query: str, text_index, embedding_model, embeddings: np.ndarray,
                  chunks: list[dict], num_results: int = 5) -> list[dict]:
    """
    Perform hybrid search combining text and vector search.

    Runs both searches and deduplicates results.

    Args:
        query: Search query string
        text_index: minsearch Index
        embedding_model: SentenceTransformer model
        embeddings: Pre-computed embeddings array
        chunks: Original chunk dictionaries
        num_results: Number of results per search method

    Returns:
        Combined deduplicated results
    """
    # Get results from both methods
    text_results = text_search(text_index, query, num_results)
    vector_results = vector_search(query, embedding_model, embeddings, chunks, num_results)

    # Deduplicate by filename + chunk content
    seen_ids = set()
    combined_results = []

    for result in text_results + vector_results:
        # Create unique ID from filename and chunk start
        chunk_id = f"{result.get('filename', '')}:{result.get('chunk', '')[:50]}"
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            combined_results.append(result)

    return combined_results


def print_results(results: list[dict], method: str):
    """Print search results in a readable format."""
    print(f"\n{'='*60}")
    print(f"  {method.upper()} SEARCH RESULTS ({len(results)} results)")
    print(f"{'='*60}")

    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"File: {result.get('filename', 'N/A')}")
        if 'title' in result:
            print(f"Title: {result.get('title', 'N/A')}")
        if 'similarity_score' in result:
            print(f"Similarity: {result['similarity_score']:.4f}")
        chunk = result.get('chunk', result.get('content', ''))
        print(f"Content: {chunk[:200]}...")


def demo_search():
    """Demo all search methods on sample data."""
    print("="*60)
    print("  DAY 3: SEARCH DEMO")
    print("="*60)

    # Step 1: Ingest data (from Day 1)
    print("\n[Step 1] Ingesting data from GitHub...")
    evidently_docs = read_repo_data('evidentlyai', 'docs')
    print(f"Downloaded {len(evidently_docs)} documents")

    # Step 2: Chunk documents (from Day 2)
    print("\n[Step 2] Chunking documents...")
    chunks = chunk_documents(evidently_docs, method='sliding', chunk_size=500, step_size=250)
    print(f"Created {len(chunks)} chunks")

    # Step 3: Create text search index
    print("\n[Step 3] Creating text search index...")
    text_index = create_text_index(chunks)
    print("Text index ready")

    # Step 4: Create vector embeddings
    print("\n[Step 4] Creating vector embeddings...")
    embedding_model, embeddings = create_embeddings(chunks)
    print(f"Created {len(embeddings)} embeddings")

    # Step 5: Demo searches
    query = "how to detect data drift"
    print(f"\n[Step 5] Searching for: '{query}'")

    # Text search
    text_results = text_search(text_index, query, num_results=3)
    print_results(text_results, "text")

    # Vector search
    vector_results = vector_search(query, embedding_model, embeddings, chunks, num_results=3)
    print_results(vector_results, "vector")

    # Hybrid search
    hybrid_results = hybrid_search(query, text_index, embedding_model, embeddings, chunks, num_results=3)
    print_results(hybrid_results, "hybrid")

    print("\n" + "="*60)
    print("  DEMO COMPLETE")
    print("="*60)


def demo_text_only():
    """Demo text search only (no heavy dependencies)."""
    print("="*60)
    print("  TEXT SEARCH DEMO")
    print("="*60)

    # Ingest and chunk
    print("\n[Step 1] Ingesting data...")
    evidently_docs = read_repo_data('evidentlyai', 'docs')
    print(f"Downloaded {len(evidently_docs)} documents")

    print("\n[Step 2] Chunking documents...")
    chunks = chunk_documents(evidently_docs, method='sliding', chunk_size=500, step_size=250)
    print(f"Created {len(chunks)} chunks")

    print("\n[Step 3] Creating text index...")
    text_index = create_text_index(chunks)

    # Search queries
    queries = [
        "how to detect data drift",
        "install evidently",
        "classification metrics"
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: '{query}'")
        results = text_search(text_index, query, num_results=3)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] {r.get('filename', 'N/A')}")
            chunk = r.get('chunk', '')[:150]
            print(f"      {chunk}...")


def main():
    parser = argparse.ArgumentParser(description='Day 3: Search - Text, Vector, and Hybrid')
    parser.add_argument('--demo', action='store_true', help='Run full demo with all search methods')
    parser.add_argument('--demo-text', action='store_true', help='Run text search demo only (lighter)')
    parser.add_argument('--query', type=str, help='Search query')
    parser.add_argument('--method', type=str, choices=['text', 'vector', 'hybrid'],
                        default='hybrid', help='Search method')
    parser.add_argument('--results', type=int, default=5, help='Number of results')

    args = parser.parse_args()

    if args.demo:
        demo_search()
    elif args.demo_text:
        demo_text_only()
    else:
        # Default: run text search demo
        demo_text_only()


if __name__ == "__main__":
    main()
