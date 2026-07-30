"""
main.py

Entry point for the Animatronic AI prototype.

Run from the Cognitive Engine so the package imports resolve:
    cd "Software/Cognitive Engine"
    python ../Chatbot/main.py [--stub]

Flags:
    --stub     Skip ollama, gliner, lingua, sentence-transformers.
               Uses canned fakes so you can verify the CLI plumbing
               (perception -> observation -> belief -> emotion ->
               context -> LLM -> output) without installing anything.

The CLI prints the AI's state alongside each response so you can see
its memory forming and decaying in real time.
"""

import argparse
import sys
from pathlib import Path

# Ensure the Cognitive Engine directory is on sys.path so its
# package imports (Core.*, Utilities.*) resolve regardless of cwd.
ENGINE_DIR = Path(__file__).resolve().parent.parent / "Cognitive Engine"
sys.path.insert(0, str(ENGINE_DIR))

from chatbot import Chatbot  # noqa: E402


def main():

    parser = argparse.ArgumentParser(description="Animatronic AI Prototype")
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Run with fake LLM + perception/observation/belief engines "
             "(no ollama, gliner, lingua, or sentence-transformers needed).",
    )
    parser.add_argument(
        "--maintenance-interval",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Seconds between maintenance ticks (decay + emotion decay + "
             "occasional reflection). Default 60. Lower it to demo faster.",
    )
    args = parser.parse_args()

    print("=" * 60)
    if args.stub:
        print("Animatronic AI Prototype — STUB MODE")
        print("=" * 60)
        print("Fake LLM + engines. The plumbing is real; responses are canned.")
        print("Replies will show '[stub]' so you can tell this isn't a real LLM.")
    else:
        print("Animatronic AI Prototype")
    print("=" * 60)
    print(f"Maintenance interval: {args.maintenance_interval}s")
    print("Type 'exit' to quit.\n")

    chatbot = Chatbot(
        stub=args.stub,
        maintenance_interval_sec=args.maintenance_interval,
    )

    try:
        while True:
            user = input("You: ")
            if user.lower() in ("exit", "quit"):
                break

            response = chatbot.chat(user)
            print(f"\nAI: {response}\n")
    finally:
        chatbot.shutdown()


if __name__ == "__main__":
    main()