import argparse
import uuid

from client import get_client
from image import generate_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the image generation demo."""
    parser = argparse.ArgumentParser(
        description="Generate an image using a Hugging Face text-to-image model",
    )
    parser.add_argument(
        "prompt",
        help="Text prompt for the image",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Hugging Face model ID",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=8,
        help="Guidance scale for image generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Width of the generated image in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Height of the generated image in pixels",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output image path (default: output_<uuid>.png)",
    )
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use inference response caching (default: True)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (or set HF_TOKEN)",
    )
    return parser.parse_args(argv)


def _default_output_path() -> str:
    """Return an output filename with a UUID."""
    return f"output_{uuid.uuid4().hex}.png"


def main(argv: list[str] | None = None) -> None:
    """Run the image generation demo."""
    args = parse_args(argv)
    output_path = args.output or _default_output_path()
    client = get_client(args.token)
    generate_image(
        client,
        prompt=args.prompt,
        model=args.model,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        width=args.width,
        height=args.height,
        use_cache=args.cache,
        output_path=output_path,
    )
    print(f"Saved image to {output_path} (prompt: {args.prompt!r})")


if __name__ == "__main__":  # pragma: no cover
    main()
