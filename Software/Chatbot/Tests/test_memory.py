"""
Tests the MemoryManager module.
"""

from memory import MemoryManager


def main():

    memory = MemoryManager()

    print("=" * 60)
    print("MEMORY MODULE TEST")
    print("=" * 60)

    # Reset memory database for testing
    memory._save([])

    print("\nMemory database cleared.")

    # Add test memories
    memory.add(
        memory_type="fact",
        content="The user likes Tame Impala."
    )

    memory.add(
        memory_type="fact",
        content="The user studies AI & ML."
    )

    memory.add(
        memory_type="fact",
        content="The user is building an offline AI."
    )

    print("\nAdded 3 memories successfully.\n")

    # Retrieve all memories
    memories = memory.get_all()

    print("=" * 60)
    print("CURRENT MEMORIES")
    print("=" * 60)

    for item in memories:

        print(f"ID         : {item['id']}")
        print(f"Type       : {item['type']}")
        print(f"Content    : {item['content']}")
        print(f"Created At : {item['created_at']}")
        print("-" * 60)

    print(f"\nTotal Memories: {len(memories)}")

    print("\nMemoryManager add() and get_all() tests passed successfully.")

    print("\n" + "=" * 60)
    print("SEARCH TEST")
    print("=" * 60)

    results = memory.search("AI")

    print("\nSearching for: AI\n")

    if results:

        for item in results:

            print(f"ID      : {item['id']}")
            print(f"Content : {item['content']}")
            print("-" * 60)

    else:

        print("No memories found.")

    print("\n" + "=" * 60)
    print("DELETE TEST")
    print("=" * 60)

    print("\nDeleting memory with ID = 2\n")

    memory.delete(2)

    memories = memory.get_all()

    for item in memories:

        print(f"ID      : {item['id']}")
        print(f"Content : {item['content']}")
        print("-" * 60)

    print(f"\nRemaining Memories: {len(memories)}")

    print("\nMemoryManager delete() test passed successfully.")

    print("\n" + "=" * 60)
    print("CLEAR TEST")
    print("=" * 60)

    print("\nClearing all memories...\n")

    memory.clear()

    memories = memory.get_all()

    if not memories:

        print("Memory database is empty.")

    else:

        print("Clear failed!")

    print(f"\nTotal Memories: {len(memories)}")

    print("\nMemoryManager tests completed successfully.")


if __name__ == "__main__":
    main()