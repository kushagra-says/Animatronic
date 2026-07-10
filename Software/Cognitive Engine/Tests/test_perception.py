"""
Integration test for the Perception Engine.
"""

import time

from Core.perception import PerceptionEngine


TEXT = """
My name is Kushagra.

I study AI & ML at Ramaiah Institute of Technology.

My favorite band is Tame Impala.

I am building an offline animatronic AI.
"""


def main():

    print("=" * 70)
    print("PERCEPTION ENGINE TEST")
    print("=" * 70)

    engine = PerceptionEngine()

    start = time.perf_counter()

    perception = engine.perceive(TEXT)

    elapsed = time.perf_counter() - start

    print("\nRaw Text\n")

    print(perception.raw_text)

    print("\nLanguage")

    print("-" * 50)

    print(perception.language.language)

    print(
        f"Confidence : "
        f"{perception.language.confidence:.3f}"
    )

    print(
    f"Extractor   : "
    f"{perception.language.extractor}"
    )

    print(
        f"Model       : "
        f"{perception.language.model}"
    )

    print(
        f"Version     : "
        f"{perception.language.version}"
    )

    print("\nEntities")

    print("-" * 50)

    for entity in perception.entities:

        print(f"Entity      : {entity.text}")

        print(f"Label       : {entity.label}")

        print(f"Confidence  : {entity.confidence:.3f}")

        print(f"Extractor   : {entity.extractor}")
    
        print(f"Model       : {entity.model}")

        print(f"Version     : {entity.version}")

        print("-" * 40)

    print("\nEmbedding")

    print("-" * 50)

    print(
        f"Dimension : "
        f"{perception.semantics.dimension}"
    )

    print(
    f"Extractor : "
    f"{perception.semantics.extractor}"
    )

    print(
        f"Model     : "
        f"{perception.semantics.model}"
    )

    print(
        f"Version   : "
        f"{perception.semantics.version}"
    )

    print("\nPipeline Latency")

    print("-" * 50)

    print(

        f"{elapsed*1000:.2f} ms"

    )

    print("\nTimestamp")

    print("-" * 50)

    print(perception.timestamp)

    print("\nPerception successfully created.")


if __name__ == "__main__":

    main()