# AI Agents Crash Course

## Day 1: Ingest and Index Your Data

Today I learned how to:
- Download repos as zip archives
- Parse frontmatter metadata
- Extract content from markdown files

### What is Frontmatter?

This format is called "frontmatter". The section between the `---` markers contains YAML metadata that describes the document, while everything below is regular Markdown content. This is very useful because we can extract structured information (like title, tags, difficulty level) along with the content.

```markdown
---
title: "Getting Started with AI"
author: "John Doe"
date: "2024-01-15"
tags: ["ai", "machine-learning", "tutorial"]
difficulty: "beginner"
---

# Getting Started with AI

This is the main content of the document written in **Markdown**.

You can include code blocks, links, and other formatting here.
```

### Run it

First build:
```bash
docker-compose build day1
```

Then run:
```bash
docker run --rm ai_agent_crashcoourse-day1
```

To see content of one example file:
```bash
docker run --rm ai_agent_crashcoourse-day1 python main.py --example
```

## Day 2: Chunking and Intelligent Processing

Today I learned about chunking strategies for preparing documents for AI agents:
- **Simple chunking**: Fixed-size chunks without overlap
- **Sliding window**: Overlapping chunks for context preservation
- **Paragraph-based**: Split by natural paragraph breaks
- **Section-based**: Split by markdown headers
- **LLM-based**: AI-powered semantic chunking using OpenRouter

### Why Chunking Matters

AI models have token limits, and large documents need to be broken into manageable pieces. The key insight: **start simple, evaluate, then iterate**. Most use cases don't require sophisticated chunking methods.

### Run it

First build:
```bash
docker-compose build day2
```

Then run with default settings (sliding window):
```bash
docker run --rm ai_agent_crashcoourse-day2
```

Demo all chunking methods (rule-based):
```bash
docker run --rm ai_agent_crashcoourse-day2 python main.py --demo
```

Try different methods:
```bash
# Simple chunking
docker run --rm ai_agent_crashcoourse-day2 python main.py --method simple --size 1000

# Sliding window with custom overlap
docker run --rm ai_agent_crashcoourse-day2 python main.py --method sliding --size 2000 --step 500

# Paragraph-based chunking
docker run --rm ai_agent_crashcoourse-day2 python main.py --method paragraph

# Section-based (by markdown headers)
docker run --rm ai_agent_crashcoourse-day2 python main.py --method section --level 2
```

Save chunks to a JSON file:
```bash
docker run --rm -v $(pwd)/output:/app/output ai_agent_crashcoourse-day2 python main.py --method sliding --save output/chunks.json
```

### LLM-Based Chunking (OpenRouter)

For intelligent semantic chunking using AI, you need an OpenRouter API key.

1. Get your API key from https://openrouter.ai/keys

2. Create your `.env` file:
```bash
cp .env.example .env
# Edit .env and add your OPENROUTER_API_KEY
```

3. Run with your `.env` file mounted:
```bash
# Demo LLM chunking on a single document
docker run --rm --env-file .env ai_agent_crashcoourse-day2 python main.py --demo-llm

# Use LLM method for all documents (costs API credits)
docker run --rm --env-file .env ai_agent_crashcoourse-day2 python main.py --method llm
```

4. Or pass the API key directly:
```bash
docker run --rm -e OPENROUTER_API_KEY=your_key_here ai_agent_crashcoourse-day2 python main.py --demo-llm
```

5. Optionally specify a different model:
```bash
docker run --rm --env-file .env -e OPENROUTER_MODEL=anthropic/claude-3-haiku ai_agent_crashcoourse-day2 python main.py --demo-llm
```

Default model: `google/gemini-2.0-flash-001` (fast and cheap)

## Day 3: Search - Text, Vector, and Hybrid

Today I learned about three search approaches for AI agents:
- **Text Search (Lexical)**: Fast keyword matching using minsearch
- **Vector Search (Semantic)**: Find similar content using embeddings with sentence-transformers
- **Hybrid Search**: Combine both approaches for best results

### Key Insight

Always start with the simplest approach. For search, that's text search. Add complexity (vector search) only when basic approaches prove insufficient.

### Run it

First build:
```bash
docker-compose build day3
```

Run text search demo (lightweight, no ML models):
```bash
docker run --rm ai_agent_crashcoourse-day3
```

Or explicitly:
```bash
docker run --rm ai_agent_crashcoourse-day3 python main.py --demo-text
```

Run full demo with all search methods (downloads ~100MB embedding model):
```bash
docker run --rm ai_agent_crashcoourse-day3 python main.py --demo
```

### Search Methods Explained

**Text Search**: Uses minsearch library for fast lexical matching. Great for exact keywords and specific terms.

```python
from minsearch import Index

index = Index(
    text_fields=["chunk", "title", "description", "filename"],
    keyword_fields=[]
)
index.fit(chunks)
results = index.search("data drift", num_results=5)
```

**Vector Search**: Uses sentence-transformers to encode text into embeddings. Finds semantically similar content even with different words.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('multi-qa-distilbert-cos-v1')
embeddings = [model.encode(chunk['text']) for chunk in chunks]
```

**Hybrid Search**: Runs both methods and deduplicates results for comprehensive coverage.

## Day 4: Agents and Tools

Today I learned how to build an actual AI agent with tool-calling capabilities:
- **What is an Agent**: An LLM that can invoke tools, not just generate text
- **Function Calling**: Define tools in JSON format for the LLM to use
- **Agent Loop**: Call LLM → check for tool calls → execute tools → repeat
- **System Prompts**: Guide agent behavior and tool usage

### Key Insight

**Tools are what distinguish agents from chatbots.** Without tools, an LLM can only respond from training data. With tools, it can access domain-specific, up-to-date information.

### Run it

First build:
```bash
docker-compose build day4
```

Run demo with sample questions:
```bash
docker run --rm -e OPENROUTER_API_KEY=your_key ai_agent_crashcoourse-day4
```

### Interactive Mode

Chat with the agent interactively:
```bash
docker run --rm -it -e OPENROUTER_API_KEY=your_key ai_agent_crashcoourse-day4 python main.py --interactive
```

### Ask a Single Question

```bash
docker run --rm -e OPENROUTER_API_KEY=your_key ai_agent_crashcoourse-day4 python main.py -q "How do I submit homework?"
```

### Verbose Mode

See what tools the agent is calling:
```bash
docker run --rm -e OPENROUTER_API_KEY=your_key ai_agent_crashcoourse-day4 python main.py -q "What are the prerequisites?" -v
```

### How It Works

1. **Define Tools**: Describe functions in JSON format so the LLM knows how to use them
2. **Send to LLM**: Include tools in the API call along with user message
3. **Handle Tool Calls**: When LLM wants to use a tool, execute it and return results
4. **Loop Until Done**: Continue until LLM provides final answer (no more tool calls)

```python
# Example tool definition
text_search_tool = {
    "type": "function",
    "function": {
        "name": "text_search",
        "description": "Search the FAQ database",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
}
```

Default model: `google/gemini-2.0-flash-001`
