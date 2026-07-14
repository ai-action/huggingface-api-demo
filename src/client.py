import os

from huggingface_hub import InferenceClient


def get_client(token: str | None = None) -> InferenceClient:
    """Return an ``InferenceClient`` authenticated with the given token.

    The token can be passed directly or via the ``HF_TOKEN`` environment
    variable. If neither is provided, ``huggingface_hub`` will fall back to its
    own token lookup (e.g. the HF cache directory).

    Args:
        token: Optional Hugging Face access token.

    Returns:
        An authenticated ``InferenceClient`` instance.
    """
    return InferenceClient(token=token or os.getenv("HF_TOKEN"))
