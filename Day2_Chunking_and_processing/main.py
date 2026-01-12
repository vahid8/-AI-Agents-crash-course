"""
Day 2: Chunking and Intelligent Processing for Data

This module implements various chunking strategies for breaking down
large documents into manageable pieces for AI agents.

Chunking Methods:
- Simple: Fixed-size chunks without overlap
- Sliding Window: Overlapping chunks for context preservation
- Paragraph: Split by natural paragraph breaks
- Section: Split by markdown headers
- LLM: AI-powered semantic chunking using OpenRouter
"""

import re
import json
import sys
import os

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
# Check both current directory and parent directory
load_dotenv()  # Current dir
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))  # Parent dir

# Add Day1 to path to reuse read_repo_data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Day1_Ingest_and_index_your_data'))
from day1_ingest import read_repo_data


# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-001')


def simple_chunking(text: str, size: int = 2000) -> list[dict]:
    """
    Split text into fixed-size chunks without overlap.

    Args:
        text: The text to chunk
        size: Maximum characters per chunk

    Returns:
        List of chunk dictionaries with start position and content
    """
    if size <= 0:
        raise ValueError("size must be positive")

    chunks = []
    for i in range(0, len(text), size):
        chunk = text[i:i + size]
        chunks.append({
            'start': i,
            'end': i + len(chunk),
            'chunk': chunk
        })
    return chunks


def sliding_window_chunking(text: str, size: int = 2000, step: int = 1000) -> list[dict]:
    """
    Split text into overlapping chunks using a sliding window.

    This method preserves context at chunk boundaries by creating
    overlap between consecutive chunks.

    Args:
        text: The text to chunk
        size: Window size (characters per chunk)
        step: Step size (how far to move the window each iteration)

    Returns:
        List of chunk dictionaries with start position and content
    """
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    n = len(text)
    chunks = []

    for i in range(0, n, step):
        chunk = text[i:i + size]
        chunks.append({
            'start': i,
            'end': i + len(chunk),
            'chunk': chunk
        })
        # Stop if we've captured the end of the text
        if i + size >= n:
            break

    return chunks


def paragraph_chunking(text: str) -> list[dict]:
    """
    Split text by natural paragraph breaks (double newlines).

    This method respects the natural structure of the document
    by splitting at paragraph boundaries.

    Args:
        text: The text to chunk

    Returns:
        List of chunk dictionaries with content
    """
    # Split on double newlines (paragraph breaks)
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    position = 0

    for para in paragraphs:
        para = para.strip()
        if para:  # Skip empty paragraphs
            chunks.append({
                'start': position,
                'chunk': para
            })
        position += len(para) + 2  # Account for the newlines

    return chunks


def markdown_section_chunking(text: str, level: int = 2) -> list[dict]:
    """
    Split markdown text by header sections.

    This method extracts content under markdown headers at a specified
    level, grouping headers with their corresponding body content.

    Args:
        text: The markdown text to chunk
        level: Header level to split on (1 = #, 2 = ##, etc.)

    Returns:
        List of chunk dictionaries with header and content
    """
    # Create pattern for the specified header level
    # e.g., level=2 matches "## Header"
    pattern = r'^(#{' + str(level) + r'})\s+(.+)$'

    lines = text.split('\n')
    chunks = []
    current_header = None
    current_content = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            # Save previous section if exists
            if current_header is not None:
                chunks.append({
                    'header': current_header,
                    'chunk': '\n'.join(current_content).strip()
                })
            current_header = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    # Don't forget the last section
    if current_header is not None:
        chunks.append({
            'header': current_header,
            'chunk': '\n'.join(current_content).strip()
        })

    return chunks


def llm_chunking(text: str, max_chunks: int = 10) -> list[dict]:
    """
    Use OpenRouter LLM to intelligently chunk text based on semantic meaning.

    This method sends the text to an LLM which identifies natural semantic
    boundaries and creates meaningful chunks.

    Args:
        text: The text to chunk
        max_chunks: Maximum number of chunks to create

    Returns:
        List of chunk dictionaries with content and optional summary
    """
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not set. "
            "Copy .env.example to .env and add your API key."
        )

    # Truncate very long texts to avoid token limits
    max_text_length = 8000
    if len(text) > max_text_length:
        text = text[:max_text_length]
        print(f"  (Text truncated to {max_text_length} chars for LLM processing)")

    prompt = f"""Analyze the following text and split it into semantically meaningful chunks.
Each chunk should be a self-contained piece of information that makes sense on its own.

Rules:
- Create between 2 and {max_chunks} chunks
- Each chunk should focus on a single topic or concept
- Preserve the original text exactly (don't summarize or modify)
- Return ONLY a valid JSON array of objects with "chunk" and "topic" fields

Example output format:
[
  {{"topic": "Introduction", "chunk": "The original text for this section..."}},
  {{"topic": "Main concept", "chunk": "Another section of original text..."}}
]

Text to chunk:
---
{text}
---

Return ONLY the JSON array, no other text:"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/AI-Agent-CrashCourse",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
            },
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        content = result['choices'][0]['message']['content']

        # Parse JSON from response (handle markdown code blocks)
        content = content.strip()
        if content.startswith('```'):
            # Remove markdown code block
            lines = content.split('\n')
            content = '\n'.join(lines[1:-1])

        chunks_data = json.loads(content)

        # Convert to our standard format
        chunks = []
        for i, item in enumerate(chunks_data):
            chunks.append({
                'chunk': item.get('chunk', ''),
                'topic': item.get('topic', f'Section {i+1}'),
                'method': 'llm'
            })

        return chunks

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"OpenRouter API request failed: {e}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse LLM response as JSON: {e}")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response format: {e}")


def chunk_text(text: str, method: str = 'sliding',
               size: int = 2000, step: int = 1000,
               level: int = 2) -> list[dict]:
    """
    Unified interface for chunking text using various strategies.

    Args:
        text: The text to chunk
        method: Chunking method ('simple', 'sliding', 'paragraph', 'section', 'llm')
        size: Chunk size for simple/sliding methods
        step: Step size for sliding window method
        level: Header level for section method

    Returns:
        List of chunk dictionaries
    """
    methods = {
        'simple': lambda: simple_chunking(text, size),
        'sliding': lambda: sliding_window_chunking(text, size, step),
        'paragraph': lambda: paragraph_chunking(text),
        'section': lambda: markdown_section_chunking(text, level),
        'llm': lambda: llm_chunking(text)
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from: {list(methods.keys())}")

    return methods[method]()


def chunk_documents(docs: list[dict], method: str = 'sliding',
                    size: int = 2000, step: int = 1000,
                    level: int = 2) -> list[dict]:
    """
    Chunk multiple documents and preserve metadata.

    Args:
        docs: List of document dictionaries from read_repo_data
        method: Chunking method
        size: Chunk size
        step: Step size for sliding window
        level: Header level for section chunking

    Returns:
        List of chunks with source metadata
    """
    all_chunks = []

    for doc in docs:
        content = doc.get('content', '')
        if not content:
            continue

        chunks = chunk_text(content, method=method, size=size, step=step, level=level)

        for i, chunk in enumerate(chunks):
            chunk['source_file'] = doc.get('filename', 'unknown')
            chunk['chunk_index'] = i
            # Preserve any frontmatter metadata
            for key in doc:
                if key not in ['content', 'filename']:
                    chunk[f'meta_{key}'] = doc[key]
            all_chunks.append(chunk)

    return all_chunks


def print_chunk_stats(chunks: list[dict], method: str, target_size: int = 2000):
    """Print statistics about the chunks."""
    print(f"\n{'='*50}")
    print(f"Chunking Method: {method}")
    print(f"{'='*50}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        lengths = [len(c.get('chunk', '')) for c in chunks]
        full_size_chunks = [l for l in lengths if l == target_size]
        partial_chunks = [l for l in lengths if l < target_size]

        print(f"Average chunk size: {sum(lengths) / len(lengths):.0f} characters")
        print(f"Min chunk size: {min(lengths)} characters")
        print(f"Max chunk size: {max(lengths)} characters")

        if method in ['simple', 'sliding']:
            print(f"Full-size chunks ({target_size} chars): {len(full_size_chunks)}")
            print(f"Partial chunks (end of documents): {len(partial_chunks)}")


def print_sample_chunks(chunks: list[dict], n: int = 2):
    """Print sample chunks for inspection."""
    print(f"\n--- Sample Chunks (first {n}) ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\nChunk {i + 1}:")
        if 'header' in chunk:
            print(f"  Header: {chunk['header']}")
        if 'source_file' in chunk:
            print(f"  Source: {chunk['source_file']}")
        content = chunk.get('chunk', '')[:200]
        print(f"  Content: {content}...")
        print()


def demo_all_methods(text: str):
    """Demonstrate all chunking methods on sample text."""
    methods = ['simple', 'sliding', 'paragraph', 'section']

    print("\n" + "="*60)
    print("DEMONSTRATING ALL CHUNKING METHODS")
    print("="*60)

    for method in methods:
        try:
            chunks = chunk_text(text, method=method, size=500, step=250, level=2)
            print_chunk_stats(chunks, method, target_size=500)
            print_sample_chunks(chunks, n=1)
        except Exception as e:
            print(f"\nError with {method}: {e}")


def main():
    """Main function to demonstrate chunking on real data."""
    import argparse

    parser = argparse.ArgumentParser(description='Day 2: Document Chunking Strategies')
    parser.add_argument('--method', choices=['simple', 'sliding', 'paragraph', 'section', 'llm'],
                        default='sliding', help='Chunking method to use')
    parser.add_argument('--size', type=int, default=2000, help='Chunk size in characters')
    parser.add_argument('--step', type=int, default=1000, help='Step size for sliding window')
    parser.add_argument('--level', type=int, default=2, help='Header level for section chunking')
    parser.add_argument('--demo', action='store_true', help='Run demo of all methods (excluding LLM)')
    parser.add_argument('--demo-llm', action='store_true', help='Demo LLM chunking on first document')
    parser.add_argument('--save', type=str, help='Save chunks to JSON file')

    args = parser.parse_args()

    print("Day 2: Chunking and Intelligent Processing")
    print("-" * 45)

    # Load data from Day 1
    print("\nLoading repository data...")
    dtc_faq = read_repo_data('DataTalksClub', 'faq')
    print(f"Loaded {len(dtc_faq)} FAQ documents")

    if args.demo:
        # Demo all methods on first document (excluding LLM to avoid API costs)
        if dtc_faq:
            sample_text = dtc_faq[0].get('content', '')
            demo_all_methods(sample_text)
        return

    if args.demo_llm:
        # Demo LLM chunking on first document
        if dtc_faq:
            sample_text = dtc_faq[0].get('content', '')
            print("\n" + "="*60)
            print("DEMONSTRATING LLM-BASED CHUNKING (OpenRouter)")
            print(f"Model: {OPENROUTER_MODEL}")
            print("="*60)
            try:
                chunks = llm_chunking(sample_text)
                print_chunk_stats(chunks, 'llm')
                print("\n--- LLM Chunks ---")
                for i, chunk in enumerate(chunks):
                    print(f"\nChunk {i + 1}:")
                    print(f"  Topic: {chunk.get('topic', 'N/A')}")
                    content = chunk.get('chunk', '')[:300]
                    print(f"  Content: {content}...")
            except Exception as e:
                print(f"\nError: {e}")
        return

    # Chunk all documents with specified method
    print(f"\nChunking documents using '{args.method}' method...")
    chunks = chunk_documents(
        dtc_faq,
        method=args.method,
        size=args.size,
        step=args.step,
        level=args.level
    )

    print_chunk_stats(chunks, args.method, target_size=args.size)
    print_sample_chunks(chunks, n=3)

    # Optionally save to file
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"\nChunks saved to: {args.save}")


if __name__ == "__main__":
    main()
