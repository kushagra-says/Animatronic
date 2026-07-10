from .execution_profile import ExecutionProfile


LINGUA_PROFILE = ExecutionProfile(

    name="Lingua",

    preferred_device="cpu",

    supports_cpu=True,

    supports_gpu=False,

    average_load_time_sec=0.0,

    average_latency_ms=2.47,

    average_ram_mb=57.24,

    average_vram_mb=0,

    reliability=0.98
)

SENTENCE_TRANSFORMER_PROFILE = ExecutionProfile(

    name="Sentence Transformer",

    preferred_device="cpu",

    supports_cpu=True,

    supports_gpu=True,

    average_load_time_sec=5.64,

    average_latency_ms=24.58,

    average_ram_mb=23.08,

    average_vram_mb=0,

    reliability=0.97
)

GLINER_PROFILE = ExecutionProfile(

    name="GLiNER",

    preferred_device="gpu",

    supports_cpu=True,

    supports_gpu=True,

    average_load_time_sec=7.33,

    average_latency_ms=156.37,

    average_ram_mb=1617.70,

    average_vram_mb=0,

    reliability=0.92
)

PROFILES = {

    "lingua": LINGUA_PROFILE,

    "gliner": GLINER_PROFILE,

    "sentence_transformer": SENTENCE_TRANSFORMER_PROFILE
}

profile = PROFILES["gliner"]

print(profile.average_latency_ms)