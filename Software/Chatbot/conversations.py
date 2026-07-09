"""
conversations.py

Maintains the current conversation history.
"""

from models import Message


class Conversation:
    """
    Stores the conversation history between
    the user and the assistant.
    """

    def __init__(self):

        self.messages = []

    def add_system_message(self, content: str):

        self.messages.append(
            Message(
                role="system",
                content=content
            )
        )

    def add_user_message(self, content: str):

        self.messages.append(
            Message(
                role="user",
                content=content
            )
        )

    def add_assistant_message(self, content: str):

        self.messages.append(
            Message(
                role="assistant",
                content=content
            )
        )

    def get_messages(self):

        return self.messages

    def clear(self):

        self.messages.clear()

    def message_count(self):

        return len(self.messages)