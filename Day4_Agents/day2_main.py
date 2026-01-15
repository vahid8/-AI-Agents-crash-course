"""
Day 2: Chunking functions for Day 3 import.
Simplified version with core chunking logic.
"""

import re


def simple_chunking(text: str, size: int = 2000) -> list[dict]:
    """Split text into fixed-size chunks without overlap."""
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
    """Split text into overlapping chunks using a sliding window."""
    n = len(text)
    chunks = []

    for i in range(0, n, step):
        chunk = text[i:i + size]
        chunks.append({
            'start': i,
            'end': i + len(chunk),
            'chunk': chunk
        })
        if i + size >= n:
            break

    return chunks


def paragraph_chunking(text: str) -> list[dict]:
    """Split text by natural paragraph breaks."""
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks = []
    position = 0

    for para in paragraphs:
        para = para.strip()
        if para:
            chunks.append({
                'start': position,
                'chunk': para
            })
        position += len(para) + 2

    return chunks


def markdown_section_chunking(text: str, level: int = 2) -> list[dict]:
    """Split markdown text by header sections."""
    pattern = r'^(#{' + str(level) + r'})\s+(.+)$'

    lines = text.split('\n')
    chunks = []
    current_header = None
    current_content = []

    for line in lines:
        match = re.match(pattern, line)
        if match:
            if current_header is not None:
                chunks.append({
                    'header': current_header,
                    'chunk': '\n'.join(current_content).strip()
                })
            current_header = match.group(2).strip()
            current_content = []
        else:
            current_content.append(line)

    if current_header is not None:
        chunks.append({
            'header': current_header,
            'chunk': '\n'.join(current_content).strip()
        })

    return chunks


def chunk_text(text: str, method: str = 'sliding',
               chunk_size: int = 2000, step_size: int = 1000,
               level: int = 2) -> list[dict]:
    """Unified interface for chunking text."""
    methods = {
        'simple': lambda: simple_chunking(text, chunk_size),
        'sliding': lambda: sliding_window_chunking(text, chunk_size, step_size),
        'paragraph': lambda: paragraph_chunking(text),
        'section': lambda: markdown_section_chunking(text, level),
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}")

    return methods[method]()


def chunk_documents(docs: list[dict], method: str = 'sliding',
                    chunk_size: int = 2000, step_size: int = 1000,
                    level: int = 2) -> list[dict]:
    """
    Chunk multiple documents and preserve metadata.

    Args:
        docs: List of document dictionaries from read_repo_data
        method: Chunking method
        chunk_size: Chunk size
        step_size: Step size for sliding window
        level: Header level for section chunking

    Returns:
        List of chunks with source metadata
    """
    all_chunks = []

    for doc in docs:
        content = doc.get('content', '')
        if not content:
            continue

        chunks = chunk_text(content, method=method, chunk_size=chunk_size,
                           step_size=step_size, level=level)

        for i, chunk in enumerate(chunks):
            chunk['filename'] = doc.get('filename', 'unknown')
            chunk['chunk_index'] = i
            # Preserve frontmatter metadata
            for key in ['title', 'description', 'tags']:
                if key in doc:
                    chunk[key] = doc[key]
            all_chunks.append(chunk)

    return all_chunks
