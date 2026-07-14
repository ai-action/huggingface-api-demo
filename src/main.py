from client import get_client
from image import generate_image


def main() -> None:
    """Run the Stable Diffusion image generation demo."""
    prompt = "Astronaut riding a horse in space"
    client = get_client()
    generate_image(client, prompt=prompt, output_path="output.png")
    print(f"Saved image to output.png (prompt: {prompt!r})")


if __name__ == "__main__":
    main()
