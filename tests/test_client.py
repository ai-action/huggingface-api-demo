from typing import Any
from unittest.mock import MagicMock, patch

from client import get_client


@patch("client.InferenceClient")
@patch("client.os.getenv", return_value="env-token")
def test_get_client_uses_env_token(
    _mock_getenv: Any,
    mock_client: Any,
) -> None:
    get_client()
    mock_client.assert_called_once_with(token="env-token")


@patch("client.InferenceClient")
def test_get_client_uses_explicit_token(mock_client: Any) -> None:
    client: Any = MagicMock()
    mock_client.return_value = client
    result = get_client("explicit-token")
    mock_client.assert_called_once_with(token="explicit-token")
    assert result is client
