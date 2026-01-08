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
