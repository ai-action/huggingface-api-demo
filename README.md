# huggingface-api-demo

[![codecov](https://codecov.io/gh/ai-action/huggingface-api-demo/graph/badge.svg?token=A5tAkYLEAw)](https://codecov.io/gh/ai-action/huggingface-api-demo)
[![lint](https://github.com/ai-action/huggingface-api-demo/actions/workflows/lint.yml/badge.svg)](https://github.com/ai-action/huggingface-api-demo/actions/workflows/lint.yml)

🤗 Hugging Face [Serverless Inference API](https://huggingface.co/learn/cookbook/enterprise_hub_serverless_inference_api) demo.

This project demonstrates how to generate images with text-to-image models (e.g. Stable Diffusion) using the Serverless Inference API and the [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) Python library.

It follows the [Creating Images with Stable Diffusion](https://huggingface.co/learn/cookbook/enterprise_hub_serverless_inference_api#2-creating-images-with-stable-diffusion) section of the cookbook.

## Prerequisites

- [Python](https://www.python.org/) 3.10+
- [uv](https://docs.astral.sh/uv/)
- [Hugging Face access token](https://huggingface.co/docs/hub/security-tokens)

## Install

Install with uv:

```sh
uv sync
```

Install pre-commit into your git hooks:

```sh
uv run pre-commit install
```

## Usage

Set your [Hugging Face token](https://huggingface.co/settings/tokens) as an environment variable:

```sh
export HF_TOKEN=<your_huggingface_token>
```

Run the demo:

```sh
uv run src/main.py "Astronaut riding a horse in space" \
  --model stabilityai/stable-diffusion-xl-base-1.0
```

This will generate an `output_<uuid>.png` image.

Available options:

```sh
uv run src/main.py --help
```

### Disable caching

The `InferenceClient` caches responses by default. To force a new generation with the same prompt, pass `--no-cache`:

```sh
uv run src/main.py "Astronaut riding a horse in space" \
  --model stabilityai/stable-diffusion-xl-base-1.0 \
  --no-cache
```

## License

[MIT](https://github.com/ai-action/huggingface-api-demo/blob/main/LICENSE)
