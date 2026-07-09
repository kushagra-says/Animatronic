"""
chatbot.py

Coordinates the conversation and the language model.
"""

from conversations import Conversation
from llm import LLM


class Chatbot:

    def __init__(self):

        self.conversation = Conversation()
        self.llm = LLM()

        # Optional system prompt
        self.conversation.add_system_message(
            "You are a helpful AI assistant."
        )

    def chat(self, user_input: str) -> str:
        """
        Handles one turn of the conversation.
        """

        # Store user's message
        self.conversation.add_user_message(user_input)

        # Generate response
        response = self.llm.generate(
            self.conversation.get_messages()
        )

        # Store assistant response
        self.conversation.add_assistant_message(response)

        return response

    def clear(self):

        self.conversation.clear()