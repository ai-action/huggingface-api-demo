from typing import Any
from unittest.mock import MagicMock, patch

from main import main, parse_args


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert args.prompt == "Astronaut riding a horse in space"
    assert args.model == "stabilityai/stable-diffusion-xl-base-1.0"
    assert args.guidance_scale == 8
    assert args.seed == 42
    assert args.output == "output.png"
    assert args.no_cache is False
    assert args.token is None


def test_parse_args_overrides() -> None:
    args = parse_args(
        [
            "--prompt",
            "a cat",
            "--model",
            "model/id",
            "--guidance-scale",
            "4",
            "--seed",
            "1",
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
    assert args.output == "cat.png"
    assert args.no_cache is True
    assert args.token == "secret"


@patch("main.get_client")
@patch("main.generate_image")
def test_main_runs_with_defaults(
    mock_generate_image: Any,
    mock_get_client: Any,
) -> None:
    client: Any = MagicMock()
    mock_get_client.return_value = client

    main([])

    mock_get_client.assert_called_once_with(None)
    mock_generate_image.assert_called_once_with(
        client,
        prompt="Astronaut riding a horse in space",
        model="stabilityai/stable-diffusion-xl-base-1.0",
        guidance_scale=8,
        seed=42,
        use_cache=True,
        output_path="output.png",
    )


@patch("main.get_client")
@patch("main.generate_image")
def test_main_passes_no_cache(
    mock_generate_image: Any,
    mock_get_client: Any,
) -> None:
    client: Any = MagicMock()
    mock_get_client.return_value = client

    main(["--no-cache", "--output", "out.png"])

    assert mock_generate_image.call_args.kwargs["use_cache"] is False
    assert mock_generate_image.call_args.kwargs["output_path"] == "out.png"
