import json
import statistics
import time

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from config import PERFORMANCE_LOGS_DIR
from logger import get_logger
from performance_monitor import PerformanceMonitor


logger = get_logger("sentence_transformer")


SENTENCES = [

    "My favorite band is Tame Impala.",

    "I love listening to Tame Impala.",

    "The weather is sunny today.",

    "I am building an offline animatronic AI.",

    "My project is an offline cognitive engine."
]


def main():

    print("=" * 70)
    print("SENTENCE TRANSFORMER TEST")
    print("=" * 70)

    monitor = PerformanceMonitor()

    # --------------------------------------------------
    # Load Model
    # --------------------------------------------------

    monitor.start()

    load_start = time.perf_counter()

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    load_time = time.perf_counter() - load_start

    monitor.stop()

    logger.info("Sentence Transformer loaded successfully.")

    # --------------------------------------------------
    # Benchmark Inference
    # --------------------------------------------------

    runs = 5

    inference_times = []

    embeddings = None

    for _ in range(runs):

        start = time.perf_counter()

        embeddings = model.encode(
            SENTENCES,
            convert_to_tensor=True
        )

        inference_times.append(
            time.perf_counter() - start
        )

    average_latency = statistics.mean(inference_times)

    report = monitor.report()

    report.update({

        "model": "all-MiniLM-L6-v2",

        "load_time_sec": round(load_time, 3),

        "average_latency_ms": round(
            average_latency * 1000,
            2
        ),

        "embedding_dimension": embeddings.shape[1],

        "runs": runs
    })

    # --------------------------------------------------
    # Similarity Tests
    # --------------------------------------------------

    similarity_band = float(

        cos_sim(

            embeddings[0],

            embeddings[1]

        )

    )

    similarity_random = float(

        cos_sim(

            embeddings[0],

            embeddings[2]

        )

    )

    similarity_project = float(

        cos_sim(

            embeddings[3],

            embeddings[4]

        )

    )

    report["similarities"] = {

        "Band Similarity":

            round(similarity_band, 3),

        "Random Similarity":

            round(similarity_random, 3),

        "Project Similarity":

            round(similarity_project, 3)
    }

    # --------------------------------------------------
    # Console Output
    # --------------------------------------------------

    print(f"\nModel                : {report['model']}")

    print(f"Load Time            : {report['load_time_sec']} s")

    print(f"Average Latency      : {report['average_latency_ms']} ms")

    print(f"Embedding Dimension  : {report['embedding_dimension']}")

    print(f"RAM Before           : {report['ram_before_mb']} MB")

    print(f"RAM After            : {report['ram_after_mb']} MB")

    print(f"RAM Delta            : {report['ram_delta_mb']} MB")

    print(f"CPU Usage            : {report['cpu_percent']} %")

    if report["gpu_percent"] is not None:

        print(f"GPU Usage            : {report['gpu_percent']} %")

        print(f"GPU Memory           : {report['vram_after_mb']:.2f} MB")

    else:

        print("GPU                  : Not Available")

    print("\nSemantic Similarity")

    print("-" * 40)

    print(

        f"Tame Impala ↔ Music      : "

        f"{similarity_band:.3f}"

    )

    print(

        f"Tame Impala ↔ Weather    : "

        f"{similarity_random:.3f}"

    )

    print(

        f"Offline AI ↔ Cognitive Engine : "

        f"{similarity_project:.3f}"

    )

    # --------------------------------------------------
    # Recommendation
    # --------------------------------------------------

    print("\nRecommendation")

    if similarity_band > 0.80:

        recommendation = "Excellent semantic representation."

    elif similarity_band > 0.60:

        recommendation = "Good semantic representation."

    else:

        recommendation = "Needs further evaluation."

    print(recommendation)

    report["recommendation"] = recommendation

    # --------------------------------------------------
    # Save Report
    # --------------------------------------------------

    output_file = (

        PERFORMANCE_LOGS_DIR

        / "sentence_transformer.json"

    )

    with open(

        output_file,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            report,

            file,

            indent=4

        )

    logger.info(

        "Sentence Transformer benchmark completed."

    )

    print(f"\nPerformance report saved to:\n{output_file}")


if __name__ == "__main__":

    main()