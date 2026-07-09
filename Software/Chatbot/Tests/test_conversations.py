"""
Tests the Conversation module independently.
"""

from conversations import Conversation


def main():

    conversation = Conversation()

    print("=" * 60)
    print("CONVERSATION MODULE TEST")
    print("=" * 60)

    conversation.add_system_message(
        "You are a helpful AI assistant."
    )

    conversation.add_user_message(
        "Hello!"
    )

    conversation.add_assistant_message(
        "Hello! How can I help you today?"
    )

    print("\nConversation History:\n")

    for message in conversation.get_messages():

        print(f"[{message.role.upper()}]")
        print(message.content)
        print()

    print(f"Total Messages: {conversation.message_count()}")

    print("\nConversation module test completed successfully.")


if __name__ == "__main__":
    main()