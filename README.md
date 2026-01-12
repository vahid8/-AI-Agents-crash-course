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
