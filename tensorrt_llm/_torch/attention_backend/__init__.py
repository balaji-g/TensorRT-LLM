from ..custom_ops import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, AttentionMetadata
from .trtllm import AttentionInputType, TrtllmAttention, TrtllmAttentionMetadata
from .vanilla import VanillaAttention, VanillaAttentionMetadata

__all__ = [
    "AttentionMetadata",
    "AttentionBackend",
    "AttentionInputType",
    "TrtllmAttention",
    "TrtllmAttentionMetadata",
    "VanillaAttention",
    "VanillaAttentionMetadata",
]

# Optional: TurboQuant KV-cache compression backend. Imported lazily so
# the rest of TRT-LLM keeps working when the `turboquant` wheel is not
# installed; only the user who selects `--attention-backend TURBOQUANT`
# pays the import cost (and gets the import-error message if the wheel
# is missing).
try:
    from .turboquant import TurboquantAttention, TurboquantAttentionMetadata  # noqa: F401
    IS_TURBOQUANT_AVAILABLE = True
    __all__ += ["TurboquantAttention", "TurboquantAttentionMetadata"]
except ImportError:
    IS_TURBOQUANT_AVAILABLE = False

if IS_FLASHINFER_AVAILABLE:
    from .flashinfer import FlashInferAttention, FlashInferAttentionMetadata
    from .star_flashinfer import StarAttention, StarAttentionMetadata
    __all__ += [
        "FlashInferAttention", "FlashInferAttentionMetadata", "StarAttention",
        "StarAttentionMetadata"
    ]
