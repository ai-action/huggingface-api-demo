import re
from typing import Any
from unittest.mock import MagicMock, patch

from main import main, parse_args


def test_parse_args_defaults() -> None:
    args = parse_args(
        [
            "Astronaut riding a horse in space",
            "--model",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
    )
    assert args.prompt == "Astronaut riding a horse in space"
    assert args.model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert args.guidance_scale == 8
    assert args.seed == 42
    assert args.width is None
    assert args.height is None
    assert args.output is None
    assert args.cache is True
    assert args.token is None


def test_parse_args_overrides() -> None:
    args = parse_args(
        [
            "a cat",
            "--model",
            "model/id",
            "--guidance-scale",
            "4",
            "--seed",
            "1",
            "--width",
            "512",
            "--height",
            "768",
            "--output",
            "cat.png",
            "--no-cache",
            "--token",
            "secret",
        ],
    )
    assert args.prompt == "a cat"
    assert args.model == "model/id"
    assert args.guidance_scale == 4
    assert args.seed == 1
    assert args.width == 512
    assert args.height == 768
    assert args.output == "cat.png"
    assert args.cache is False
    assert args.token == "secret"


@patch("main.get_client")
@patch("main.generate_image")
def test_main_runs_with_defaults(
    mock_generate_image: Any,
    mock_get_client: Any,
) -> None:
    client: Any = MagicMock()
    mock_get_client.return_value = client

    main(
        [
            "Astronaut riding a horse in space",
            "--model",
            "stabilityai/stable-diffusion-xl-base-1.0",
        ],
    )

    mock_get_client.assert_called_once_with(None)
    assert mock_generate_image.call_count == 1
    call_kwargs = mock_generate_image.call_args.kwargs
    assert call_kwargs["prompt"] == "Astronaut riding a horse in space"
    assert call_kwargs["model"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert call_kwargs["guidance_scale"] == 8
    assert call_kwargs["seed"] == 42
    assert call_kwargs["width"] is None
    assert call_kwargs["height"] is None
    assert call_kwargs["use_cache"] is True
    assert re.fullmatch(r"output_[0-9a-f]{32}\.png", call_kwargs["output_path"])


@patch("main.get_client")
@patch("main.generate_image")
def test_main_passes_no_cache(
    mock_generate_image: Any,
    mock_get_client: Any,
) -> None:
    client: Any = MagicMock()
    mock_get_client.return_value = client

    main(["a prompt", "--model", "model/id", "--no-cache", "--output", "out.png"])

    assert mock_generate_image.call_args.kwargs["use_cache"] is False
    assert mock_generate_image.call_args.kwargs["output_path"] == "out.png"
