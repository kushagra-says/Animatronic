"""
Tests the LLM module independently.
"""

from llm import LLM
from models import Message


def main():

    llm = LLM()

    print("=" * 60)
    print("LLM MODULE TEST")
    print("=" * 60)

    print("\nChecking Ollama server...")

    llm.health_check()

    print("Connection successful.\n")

    print("Installed Models:")

    for model in llm.list_models():
        print(f"  • {model}")

    print("\nSending test prompt...\n")

    messages = [
        Message(
            role="user",
            content="Say hello in one short sentence."
        )
    ]

    response = llm.generate(messages)

    print("Model Response:\n")
    print(response)

    print("\nLLM test completed successfully.")


if __name__ == "__main__":
    main()