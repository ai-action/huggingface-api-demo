from pathlib import Path

from huggingface_hub import InferenceClient
from PIL import Image

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def generate_image(
    client: InferenceClient,
    prompt: str,
    model: str = DEFAULT_MODEL,
    guidance_scale: float = 8,
    seed: int = 42,
    use_cache: bool = True,
    output_path: str | Path | None = None,
) -> Image.Image:
    """Generate an image using the Serverless Inference API.

    This mirrors the "Creating Images with Stable Diffusion" example from the
    Hugging Face cookbook. By default, ``InferenceClient`` caches responses,
    so repeating the same prompt returns the same image. Passing
    ``use_cache=False`` disables caching via the ``x-use-cache: 0`` header.

    Args:
        client: An authenticated ``InferenceClient``.
        prompt: The text prompt to generate an image from.
        model: Hugging Face model ID for a text-to-image model.
        guidance_scale: How closely the image should follow the prompt.
        seed: Random seed for reproducible generation.
        use_cache: Whether to use the inference cache. Set to ``False`` to
            force a new generation with the same payload.
        output_path: Optional path to save the generated image to.

    Returns:
        The generated PIL image.
    """
    if not use_cache:
        client.headers["x-use-cache"] = "0"

    image = client.text_to_image(
        prompt=prompt,
        model=model,
        guidance_scale=guidance_scale,
        seed=seed,
    )

    if output_path:
        image.save(output_path)

    return image
