"""
Day 5: Evaluation - Building Reliable AI Agents

This module implements:
- Logging system to track agent interactions
- Automated evaluation using LLM as a judge
- Test data generation
- Performance metrics calculation

Evaluation is critical for building reliable AI systems. Without proper
evaluation, you can't tell if your changes improve or hurt performance.
"""

import os
import json
import secrets
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any

import requests
import pandas as pd
from dotenv import load_dotenv
from minsearch import Index

from day1_main import read_repo_data
from day2_main import chunk_documents

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-001')

# Logging directory
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)


# =============================================================================
# Agent Setup (from Day 4)
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant for a course.

Use the search tool to find relevant information from the course materials before answering questions.

If you can find specific information through search, use it to provide accurate answers.
Always include references by citing the filename of the source material you used.

If the search doesn't return relevant results, let the user know and provide general guidance.
"""

TEXT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "text_search",
        "description": "Search the FAQ database for relevant information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find relevant FAQ entries"
                }
            },
            "required": ["query"]
        }
    }
}


class FAQAgent:
    """FAQ Agent with logging capabilities."""

    def __init__(self, name: str = "faq_agent", system_prompt: str = SYSTEM_PROMPT):
        self.name = name
        self.system_prompt = system_prompt
        self.index = None
        self.chunks = None

    def setup(self):
        """Initialize the search index with FAQ data."""
        print("Loading FAQ data...")
        faq_docs = read_repo_data('DataTalksClub', 'faq')
        print(f"Downloaded {len(faq_docs)} FAQ documents")

        print("Chunking documents...")
        self.chunks = chunk_documents(faq_docs, method='sliding', chunk_size=500, step_size=250)
        print(f"Created {len(self.chunks)} chunks")

        print("Building search index...")
        self.index = Index(
            text_fields=["chunk", "title", "description", "filename"],
            keyword_fields=[]
        )
        self.index.fit(self.chunks)
        print("Agent ready!")

    def text_search(self, query: str, num_results: int = 5) -> list[dict]:
        """Search the FAQ database."""
        results = self.index.search(query, num_results=num_results)
        return [
            {"text": r.get("chunk", ""), "filename": r.get("filename", "unknown")}
            for r in results
        ]

    def call_openrouter(self, messages: list[dict], tools: list[dict] = None) -> dict:
        """Make an API call to OpenRouter."""
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY not set")

        payload = {
            "model": OPENROUTER_MODEL,
            "messages": messages,
        }
        if tools:
            payload["tools"] = tools

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    def run(self, question: str, max_iterations: int = 5) -> tuple[str, list[dict]]:
        """Run the agent and return (answer, messages)."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]

        for _ in range(max_iterations):
            response = self.call_openrouter(messages, tools=[TEXT_SEARCH_TOOL])
            message = response["choices"][0]["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                return message.get("content", ""), messages + [message]

            messages.append(message)

            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])

                if func_name == "text_search":
                    result = self.text_search(func_args["query"])
                else:
                    result = {"error": f"Unknown function: {func_name}"}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })

        return "Max iterations reached", messages


# =============================================================================
# Logging System
# =============================================================================

def log_entry(agent: FAQAgent, messages: list[dict], source: str = "user") -> dict:
    """Create a log entry from agent interaction."""
    return {
        "agent_name": agent.name,
        "system_prompt": agent.system_prompt,
        "model": OPENROUTER_MODEL,
        "tools": ["text_search"],
        "messages": messages,
        "source": source,
        "timestamp": datetime.now().isoformat()
    }


def save_log(agent: FAQAgent, messages: list[dict], source: str = "user") -> Path:
    """Save interaction log to a JSON file."""
    entry = log_entry(agent, messages, source)

    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)
    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=str)

    return filepath


def load_log(log_file: Path) -> dict:
    """Load a log file."""
    with open(log_file, 'r') as f_in:
        log_data = json.load(f_in)
        log_data['log_file'] = str(log_file)
        return log_data


# =============================================================================
# LLM as a Judge - Evaluation
# =============================================================================

EVALUATION_PROMPT = """
Use this checklist to evaluate the quality of an AI agent's answer (<ANSWER>) to a user question (<QUESTION>).

For each item, check if the condition is met.

Checklist:
- instructions_follow: The agent followed the instructions
- answer_relevant: The response directly addresses the user's question
- answer_clear: The answer is clear and understandable
- answer_citations: The response includes proper citations or sources
- completeness: The response is complete and covers key aspects
- tool_usage: The search tool was used appropriately

Output JSON with this exact structure:
{
  "checks": [
    {"name": "check_name", "pass": true/false, "reason": "brief explanation"}
  ],
  "summary": "overall assessment"
}
"""


def evaluate_interaction(question: str, answer: str, messages: list[dict]) -> dict:
    """Evaluate an agent interaction using LLM as a judge."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Simplify messages for evaluation
    simplified_log = []
    for m in messages:
        if m.get("role") == "tool":
            simplified_log.append({"role": "tool", "content": "[search results]"})
        else:
            simplified_log.append(m)

    user_prompt = f"""
<QUESTION>{question}</QUESTION>
<ANSWER>{answer}</ANSWER>
<LOG>{json.dumps(simplified_log)}</LOG>
"""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": EVALUATION_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        },
        timeout=60
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse evaluation", "raw": content}


def evaluate_log_file(log_file: Path) -> dict:
    """Evaluate a single log file."""
    log_data = load_log(log_file)
    messages = log_data["messages"]

    # Extract question and answer
    question = None
    answer = None

    for m in messages:
        if m.get("role") == "user":
            question = m.get("content", "")
        elif m.get("role") == "assistant" and "content" in m:
            answer = m.get("content", "")

    if not question or not answer:
        return {"error": "Could not extract question/answer"}

    eval_result = evaluate_interaction(question, answer, messages)
    eval_result["log_file"] = str(log_file)
    eval_result["question"] = question
    eval_result["answer"] = answer[:200] + "..." if len(answer) > 200 else answer

    return eval_result


# =============================================================================
# Test Data Generation
# =============================================================================

QUESTION_GENERATION_PROMPT = """
You are helping to create test questions for an AI agent that answers questions about a course FAQ.

Based on the provided FAQ content, generate realistic questions that students might ask.

The questions should:
- Be natural and varied in style
- Range from simple to complex
- Include both specific technical questions and general course questions

Generate exactly {num_questions} questions. Return as JSON:
{{"questions": ["question1", "question2", ...]}}
"""


def generate_test_questions(chunks: list[dict], num_questions: int = 5) -> list[str]:
    """Generate test questions from FAQ content."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not set")

    # Sample some chunks for context
    sample_size = min(10, len(chunks))
    sample = random.sample(chunks, sample_size)
    content = json.dumps([c.get("chunk", "") for c in sample])

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": QUESTION_GENERATION_PROMPT.format(num_questions=num_questions)},
                {"role": "user", "content": content}
            ],
            "response_format": {"type": "json_object"}
        },
        timeout=60
    )
    response.raise_for_status()

    result = response.json()
    content = result["choices"][0]["message"]["content"]

    try:
        data = json.loads(content)
        return data.get("questions", [])
    except json.JSONDecodeError:
        return []


# =============================================================================
# Metrics Calculation
# =============================================================================

def calculate_metrics(eval_results: list[dict]) -> pd.DataFrame:
    """Calculate metrics from evaluation results."""
    rows = []

    for result in eval_results:
        if "error" in result:
            continue

        row = {
            "file": result.get("log_file", ""),
            "question": result.get("question", "")[:50],
        }

        checks = result.get("checks", [])
        for check in checks:
            row[check["name"]] = check["pass"]

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def print_metrics_summary(df: pd.DataFrame):
    """Print a summary of evaluation metrics."""
    if df.empty:
        print("No evaluation data available")
        return

    print("\n" + "=" * 60)
    print("  EVALUATION METRICS SUMMARY")
    print("=" * 60)

    # Calculate pass rates for each check
    numeric_cols = df.select_dtypes(include=[bool]).columns
    for col in numeric_cols:
        pass_rate = df[col].mean() * 100
        print(f"  {col}: {pass_rate:.1f}% pass rate")

    print("=" * 60)


# =============================================================================
# Demo Functions
# =============================================================================

def demo_logging():
    """Demo the logging system."""
    print("=" * 60)
    print("  LOGGING DEMO")
    print("=" * 60)

    agent = FAQAgent(name="faq_agent_v1")
    agent.setup()

    questions = [
        "How do I install Docker on Windows?",
        "Can I join the course late?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        answer, messages = agent.run(q)
        print(f"Answer: {answer[:200]}...")

        log_path = save_log(agent, messages, source="demo")
        print(f"Logged to: {log_path}")


def demo_evaluation():
    """Demo the evaluation system."""
    print("=" * 60)
    print("  EVALUATION DEMO")
    print("=" * 60)

    # Find existing logs
    log_files = list(LOG_DIR.glob("*.json"))
    if not log_files:
        print("No logs found. Run --demo-logging first.")
        return

    print(f"Found {len(log_files)} log files")

    # Evaluate first few logs
    eval_results = []
    for log_file in log_files[:3]:
        print(f"\nEvaluating: {log_file.name}")
        result = evaluate_log_file(log_file)

        if "error" not in result:
            print(f"  Summary: {result.get('summary', 'N/A')}")
            eval_results.append(result)
        else:
            print(f"  Error: {result['error']}")

    # Calculate and print metrics
    if eval_results:
        df = calculate_metrics(eval_results)
        print_metrics_summary(df)


def demo_generation():
    """Demo test data generation."""
    print("=" * 60)
    print("  TEST DATA GENERATION DEMO")
    print("=" * 60)

    agent = FAQAgent(name="faq_agent_v1")
    agent.setup()

    print("\nGenerating test questions...")
    questions = generate_test_questions(agent.chunks, num_questions=3)

    print(f"Generated {len(questions)} questions:")
    for i, q in enumerate(questions, 1):
        print(f"  {i}. {q}")

    print("\nRunning agent on generated questions...")
    for q in questions:
        print(f"\nQ: {q}")
        answer, messages = agent.run(q)
        print(f"A: {answer[:150]}...")
        save_log(agent, messages, source="ai-generated")


def demo_full():
    """Run complete evaluation pipeline."""
    print("=" * 60)
    print("  FULL EVALUATION PIPELINE")
    print("=" * 60)

    # Step 1: Setup agent
    print("\n[Step 1] Setting up agent...")
    agent = FAQAgent(name="faq_agent_eval")
    agent.setup()

    # Step 2: Generate test questions
    print("\n[Step 2] Generating test questions...")
    questions = generate_test_questions(agent.chunks, num_questions=5)
    print(f"Generated {len(questions)} questions")

    # Step 3: Run agent and log
    print("\n[Step 3] Running agent on test questions...")
    log_files = []
    for q in questions:
        print(f"  Processing: {q[:50]}...")
        answer, messages = agent.run(q)
        log_path = save_log(agent, messages, source="ai-generated")
        log_files.append(log_path)

    # Step 4: Evaluate
    print("\n[Step 4] Evaluating responses...")
    eval_results = []
    for log_file in log_files:
        result = evaluate_log_file(log_file)
        eval_results.append(result)
        if "summary" in result:
            print(f"  {log_file.name}: {result['summary'][:50]}...")

    # Step 5: Calculate metrics
    print("\n[Step 5] Calculating metrics...")
    df = calculate_metrics(eval_results)
    print_metrics_summary(df)

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Day 5: Evaluation')
    parser.add_argument('--demo-logging', action='store_true', help='Demo logging system')
    parser.add_argument('--demo-eval', action='store_true', help='Demo evaluation on existing logs')
    parser.add_argument('--demo-generate', action='store_true', help='Demo test data generation')
    parser.add_argument('--demo', action='store_true', help='Run full evaluation pipeline')

    args = parser.parse_args()

    if args.demo_logging:
        demo_logging()
    elif args.demo_eval:
        demo_evaluation()
    elif args.demo_generate:
        demo_generation()
    elif args.demo:
        demo_full()
    else:
        # Default: run full demo
        demo_full()


if __name__ == "__main__":
    main()
