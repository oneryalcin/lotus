from lotus.models.lm import LM
from lotus.models.reranker import Reranker
from lotus.models.rm import RM
from lotus.models.litellm_rm import LiteLLMRM
from lotus.models.colbertv2_rm import ColBERTv2RM
from lotus.models.model2vec_rm import Model2VecRM

# Lazy imports for optional dependencies
def __getattr__(name):
    if name == "CrossEncoderReranker":
        try:
            from lotus.models.cross_encoder_reranker import CrossEncoderReranker
            return CrossEncoderReranker
        except ImportError as e:
            raise ImportError(
                f"CrossEncoderReranker requires sentence-transformers. "
                f"Install with: pip install 'lotus-ai[sentence_transformers]'. "
                f"Original error: {e}"
            )
    elif name == "SentenceTransformersRM":
        try:
            from lotus.models.sentence_transformers_rm import SentenceTransformersRM
            return SentenceTransformersRM
        except ImportError as e:
            raise ImportError(
                f"SentenceTransformersRM requires sentence-transformers. "
                f"Install with: pip install 'lotus-ai[sentence_transformers]'. "
                f"Original error: {e}"
            )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CrossEncoderReranker",
    "LM",
    "RM",
    "Reranker",
    "LiteLLMRM",
    "SentenceTransformersRM",
    "ColBERTv2RM",
    "Model2VecRM",
]
