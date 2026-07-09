"""
memory.py

Manages persistent memory for the chatbot prototype.
"""

import json
from pathlib import Path
from datetime import datetime


class MemoryManager:
    """
    Handles loading and saving memories
    to the local JSON database.
    """

    def __init__(self):

        self.memory_file = (
            Path(__file__).parent
            / "Data"
            / "memories.json"
        )

        # Ensure the file exists
        if not self.memory_file.exists():

            self.memory_file.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            with open(self.memory_file, "w") as file:
                json.dump([], file, indent=4)

    def _load(self):
        """
        Loads all memories from disk.
        """

        with open(
            self.memory_file,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    def _save(self, memories):
        """
        Saves all memories to disk.
        """

        with open(
            self.memory_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                memories,
                file,
                indent=4,
                ensure_ascii=False
            )
    
    def add(self, memory_type: str, content: str):

        memories = self._load()

        new_memory = {
            "id": len(memories) + 1,
            "type": memory_type,
            "content": content,
            "created_at": datetime.now().isoformat()
        }

        memories.append(new_memory)

        self._save(memories)

    def get_all(self):

        return self._load()
    
    def search(self, query: str):

        memories = self.get_all()

        results = []

        query = query.lower()

        for memory in memories:

            if query in memory["content"].lower():

                results.append(memory)

        return results
    
    def delete(self, memory_id: int):

        memories = self.get_all()

        updated_memories = [
            memory
            for memory in memories
            if memory["id"] != memory_id
        ]

        self._save(updated_memories)

    def clear(self):

        self._save([])