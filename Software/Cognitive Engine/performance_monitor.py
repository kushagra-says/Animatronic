"""
performance_monitor.py

Shared performance monitoring utilities.
"""

import time
import psutil

try:
    from pynvml import *
    nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class PerformanceMonitor:

    def __init__(self):

        self.process = psutil.Process()

        self.start_time = None
        self.end_time = None

        self.ram_before = 0
        self.ram_after = 0

        self.gpu_before = None
        self.gpu_after = None

        self.vram_before = None
        self.vram_after = None

    # --------------------------------------------------

    def _gpu_info(self):

        if not GPU_AVAILABLE:
            return None, None

        handle = nvmlDeviceGetHandleByIndex(0)

        util = nvmlDeviceGetUtilizationRates(handle)

        memory = nvmlDeviceGetMemoryInfo(handle)

        return util.gpu, memory.used / (1024 ** 2)

    # --------------------------------------------------

    def start(self):

        self.start_time = time.perf_counter()

        self.ram_before = self.process.memory_info().rss / (1024 ** 2)

        self.gpu_before, self.vram_before = self._gpu_info()

    # --------------------------------------------------

    def stop(self):

        self.end_time = time.perf_counter()

        self.ram_after = self.process.memory_info().rss / (1024 ** 2)

        self.gpu_after, self.vram_after = self._gpu_info()

    # --------------------------------------------------

    def report(self):

        cpu = psutil.cpu_percent(interval=0.1)

        return {

            "elapsed_time_sec":
                round(self.end_time - self.start_time, 3),

            "ram_before_mb":
                round(self.ram_before, 2),

            "ram_after_mb":
                round(self.ram_after, 2),

            "ram_delta_mb":
                round(self.ram_after - self.ram_before, 2),

            "cpu_percent":
                round(cpu, 2),

            "gpu_percent":
                self.gpu_after,

            "gpu_before_percent":
                self.gpu_before,

            "vram_before_mb":
                self.vram_before,

            "vram_after_mb":
                self.vram_after,

            "vram_delta_mb":
                None if self.vram_before is None else
                round(self.vram_after - self.vram_before, 2)
        }