"""
Day 4: Agents and Tools - Building the Conversational Agent

An agent is an LLM that can not only generate text, but also invoke tools.
Tools are external functions enabling information retrieval, calculations, or actions.
This is what distinguishes agents from simple chatbots.
"""

import os
import json
import argparse
import requests
from dotenv import load_dotenv

from minsearch import Index
from day1_main import read_repo_data
from day2_main import chunk_documents

# Load environment variables
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_MODEL = os.getenv('OPENROUTER_MODEL', 'google/gemini-2.0-flash-001')

# System prompt that guides agent behavior
SYSTEM_PROMPT = """You are a helpful FAQ assistant for the DataTalks.Club courses.

IMPORTANT RULES:
1. Always search for relevant information before answering
2. Make multiple searches if needed for comprehensive answers
3. If the search returns no results, say you don't have that information
4. Base your answers ONLY on search results, not your training data
5. Be concise but thorough
6. If asked about something not in the FAQ, clearly state that

When users ask questions, use the text_search tool to find relevant FAQ entries."""


# Tool definition for OpenRouter
TEXT_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "text_search",
        "description": "Search the FAQ database for relevant information about DataTalks.Club courses",
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
    """An agent that can search FAQ data and answer questions."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
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

    def text_search(self, query: str, num_results: int = 3) -> list[dict]:
        """Search the FAQ database."""
        if self.index is None:
            raise RuntimeError("Agent not initialized. Call setup() first.")

        results = self.index.search(query, num_results=num_results)
        # Return simplified results for the LLM
        return [
            {
                "text": r.get("chunk", ""),
                "source": r.get("filename", "unknown")
            }
            for r in results
        ]

    def call_openrouter(self, messages: list[dict], tools: list[dict] = None) -> dict:
        """Make an API call to OpenRouter."""
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not set. "
                "Set it in .env file or pass via environment variable."
            )

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
                "HTTP-Referer": "https://github.com/AI-Agent-CrashCourse",
            },
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()

    def run(self, question: str, max_iterations: int = 5) -> str:
        """
        Run the agent loop to answer a question.

        The loop:
        1. Send message to LLM with tools
        2. If LLM wants to call a tool, execute it and add result
        3. Repeat until LLM returns final answer (no tool calls)
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]

        for iteration in range(max_iterations):
            if self.verbose:
                print(f"\n[Iteration {iteration + 1}]")

            response = self.call_openrouter(messages, tools=[TEXT_SEARCH_TOOL])
            message = response["choices"][0]["message"]

            # Check if agent wants to call tools
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                # No tool calls - return final answer
                return message.get("content", "No response generated")

            # Process tool calls
            if self.verbose:
                print(f"Agent wants to call {len(tool_calls)} tool(s)")

            # Add assistant message with tool calls
            messages.append(message)

            for tool_call in tool_calls:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])

                if self.verbose:
                    print(f"  Calling: {func_name}({func_args})")

                # Execute the tool
                if func_name == "text_search":
                    result = self.text_search(func_args["query"])
                else:
                    result = {"error": f"Unknown function: {func_name}"}

                if self.verbose:
                    print(f"  Results: {len(result)} items found")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result)
                })

        return "Max iterations reached without final answer"


def demo_agent():
    """Demo the FAQ agent with sample questions."""
    print("=" * 60)
    print("  DAY 4: AGENTS AND TOOLS DEMO")
    print("=" * 60)

    agent = FAQAgent(verbose=True)
    agent.setup()

    # Sample questions
    questions = [
        "How do I submit my homework for the ML Zoomcamp?",
        "What are the prerequisites for the course?",
    ]

    for question in questions:
        print("\n" + "=" * 60)
        print(f"Question: {question}")
        print("=" * 60)

        answer = agent.run(question)

        print("\n--- Answer ---")
        print(answer)


def interactive_mode():
    """Run the agent in interactive mode."""
    print("=" * 60)
    print("  FAQ AGENT - INTERACTIVE MODE")
    print("=" * 60)
    print("Type 'quit' or 'exit' to stop.\n")

    agent = FAQAgent(verbose=False)
    agent.setup()

    while True:
        try:
            question = input("\nYou: ").strip()
            if not question:
                continue
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            answer = agent.run(question)
            print(f"\nAgent: {answer}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description='Day 4: Agents and Tools')
    parser.add_argument('--demo', action='store_true', help='Run demo with sample questions')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    parser.add_argument('--question', '-q', type=str, help='Ask a single question')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    if args.demo:
        demo_agent()
    elif args.interactive:
        interactive_mode()
    elif args.question:
        agent = FAQAgent(verbose=args.verbose)
        agent.setup()
        answer = agent.run(args.question)
        print(f"\nAnswer: {answer}")
    else:
        # Default: run demo
        demo_agent()


if __name__ == "__main__":
    main()
