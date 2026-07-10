from dataclasses import dataclass


@dataclass
class ExecutionProfile:

    preferred_device: str

    supports_cpu: bool

    supports_gpu: bool

    estimated_ram_mb: float

    estimated_vram_mb: float

    average_latency_ms: float

LINGUA_PROFILE = ExecutionProfile(

    preferred_device="cpu",

    supports_cpu=True,

    supports_gpu=False,

    estimated_ram_mb=60,

    estimated_vram_mb=0,

    average_latency_ms=3
)

GLINER_PROFILE = ExecutionProfile(

    preferred_device="gpu",

    supports_cpu=True,

    supports_gpu=True,

    estimated_ram_mb=1600,

    estimated_vram_mb=700,

    average_latency_ms=150
)