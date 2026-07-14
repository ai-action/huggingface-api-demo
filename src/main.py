import argparse

from client import get_client
from image import DEFAULT_MODEL, generate_image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the image generation demo."""
    parser = argparse.ArgumentParser(
        description="Generate an image with Stable Diffusion",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Astronaut riding a horse in space",
        help="Text prompt for the image",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=DEFAULT_MODEL,
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
        "--output",
        "-o",
        default="output.png",
        help="Output image path",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable inference response caching",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face access token (or set HF_TOKEN)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the Stable Diffusion image generation demo."""
    args = parse_args(argv)
    client = get_client(args.token)
    generate_image(
        client,
        prompt=args.prompt,
        model=args.model,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_cache=not args.no_cache,
        output_path=args.output,
    )
    print(f"Saved image to {args.output} (prompt: {args.prompt!r})")


if __name__ == "__main__":  # pragma: no cover
    main()
