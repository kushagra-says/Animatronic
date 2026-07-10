import json
import statistics
import time

from gliner import GLiNER

from config import PERFORMANCE_LOGS_DIR
from logger import get_logger
from performance_monitor import PerformanceMonitor


logger = get_logger("gliner")


TEXT = """
My name is Kushagra.
I study AI & ML at Ramaiah Institute of Technology.
I am building an offline animatronic AI.
My favorite band is Tame Impala.
"""

LABELS = [
    "person",
    "university",
    "field of study",
    "project",
    "music artist",
]


def main():

    print("=" * 70)
    print("GLiNER TEST")
    print("=" * 70)

    # ----------------------------------------------------
    # Model Loading
    # ----------------------------------------------------

    monitor = PerformanceMonitor()

    monitor.start()

    load_start = time.perf_counter()

    model = GLiNER.from_pretrained(
        "urchade/gliner_medium-v2.1"
    )

    load_time = time.perf_counter() - load_start

    monitor.stop()

    logger.info("GLiNER loaded successfully.")

    # ----------------------------------------------------
    # Inference
    # ----------------------------------------------------

    runs = 5

    inference_times = []

    entities = None

    for _ in range(runs):

        start = time.perf_counter()

        entities = model.predict_entities(
            TEXT,
            LABELS
        )

        inference_times.append(
            time.perf_counter() - start
        )

    average_latency = statistics.mean(inference_times)

    average_confidence = statistics.mean(
        entity["score"] for entity in entities
    )

    report = monitor.report()

    report.update({

        "model": "GLiNER Medium v2.1",

        "load_time_sec": round(load_time, 3),

        "average_latency_ms": round(
            average_latency * 1000,
            2
        ),

        "average_confidence": round(
            average_confidence,
            3
        ),

        "runs": runs
    })

    # ----------------------------------------------------
    # Console Output
    # ----------------------------------------------------

    print(f"\nModel                : {report['model']}")
    print(f"Load Time            : {report['load_time_sec']} s")
    print(f"Average Latency      : {report['average_latency_ms']} ms")
    print(f"RAM Usage            : {report['ram_mb']} MB")
    print(f"CPU Usage            : {report['cpu_percent']} %")

    if report["gpu_percent"] is not None:

        print(f"GPU Usage            : {report['gpu_percent']} %")
        print(f"GPU Memory           : {report['gpu_memory_mb']:.2f} MB")

    else:

        print("GPU                  : Not Available")

    print(f"\nAverage Confidence   : {report['average_confidence']}")

    print("\nDetected Entities\n")

    for entity in entities:

        print(entity)

    # ----------------------------------------------------
    # Save Report
    # ----------------------------------------------------

    output_file = PERFORMANCE_LOGS_DIR / "gliner.json"

    with open(output_file, "w", encoding="utf-8") as file:

        json.dump(report, file, indent=4)

    logger.info("GLiNER benchmark completed.")

    print(f"\nPerformance report saved to:\n{output_file}")


if __name__ == "__main__":
    main()