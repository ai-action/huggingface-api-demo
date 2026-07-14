from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

from PIL import Image

from image import generate_image


def test_generate_image_caches_by_default(tmp_path: Path) -> None:
    client: Any = MagicMock()
    fake_image = Image.new("RGB", (64, 64))
    client.text_to_image.return_value = fake_image
    output = tmp_path / "out.png"

    result = generate_image(client, "a prompt", "model/id", output_path=output)

    client.text_to_image.assert_called_once_with(
        prompt="a prompt",
        model="model/id",
        guidance_scale=8,
        seed=42,
    )
    assert result == fake_image
    assert output.exists()


def test_generate_image_can_disable_cache() -> None:
    client: Any = MagicMock()
    client.headers = cast(dict[str, str], {})
    client.text_to_image.return_value = Image.new("RGB", (64, 64))

    generate_image(client, "a prompt", "model/id", use_cache=False)

    assert client.headers["x-use-cache"] == "0"
