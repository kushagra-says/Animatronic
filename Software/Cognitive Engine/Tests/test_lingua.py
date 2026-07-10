import json
import statistics
import time

from lingua import LanguageDetectorBuilder

from config import PERFORMANCE_LOGS_DIR
from logger import get_logger
from performance_monitor import PerformanceMonitor


logger = get_logger("lingua")


TEXTS = [

    "Hello, how are you?",

    "नमस्ते, आप कैसे हैं?",

    "こんにちは",

    "Bonjour tout le monde.",

    "Hola amigo."
]


def main():

    print("=" * 70)
    print("LINGUA TEST")
    print("=" * 70)

    monitor = PerformanceMonitor()

    monitor.start()

    load_start = time.perf_counter()

    detector = (
        LanguageDetectorBuilder
        .from_all_languages()
        .build()
    )

    load_time = time.perf_counter() - load_start

    monitor.stop()

    logger.info("Lingua loaded successfully.")

    runs = 5

    inference_times = []

    results = []

    for text in TEXTS:

        start = time.perf_counter()

        language = detector.detect_language_of(text)

        inference_times.append(
            time.perf_counter() - start
        )

        results.append({

            "text": text,

            "language": str(language)
        })

    average_latency = statistics.mean(inference_times)

    report = monitor.report()

    report.update({

        "model": "Lingua",

        "load_time_sec": round(load_time,3),

        "average_latency_ms": round(
            average_latency*1000,
            2
        ),

        "runs": len(TEXTS)
    })

    print(f"\nModel : {report['model']}")

    print(f"Load Time : {report['load_time_sec']} s")

    print(f"Average Latency : {report['average_latency_ms']} ms")

    print(f"RAM Usage : {report['ram_mb']} MB")

    print(f"CPU Usage : {report['cpu_percent']} %")

    if report["gpu_percent"] is not None:

        print(f"GPU Usage : {report['gpu_percent']} %")

        print(f"GPU Memory : {report['gpu_memory_mb']:.2f} MB")

    print("\nDetected Languages\n")

    for result in results:

        print(result)

    with open(

        PERFORMANCE_LOGS_DIR / "lingua.json",

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            report,

            file,

            indent=4

        )

    logger.info("Lingua benchmark completed.")


if __name__ == "__main__":
    main()