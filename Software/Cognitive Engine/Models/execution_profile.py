from dataclasses import dataclass


@dataclass
class ExecutionProfile:

    name: str

    preferred_device: str

    supports_cpu: bool
    supports_gpu: bool

    average_load_time_sec: float
    average_latency_ms: float

    average_ram_mb: float
    average_vram_mb: float

    reliability: float