"""
main.py

Entry point for the chatbot prototype.
"""

from chatbot import Chatbot


def main():

    print("=" * 60)
    print("Animatronic AI Prototype")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    chatbot = Chatbot()

    while True:

        user = input("You: ")

        if user.lower() == "exit":
            break

        response = chatbot.chat(user)

        print(f"\nAI: {response}\n")


if __name__ == "__main__":
    main()