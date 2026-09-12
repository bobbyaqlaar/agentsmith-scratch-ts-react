"""
runtime/test/test_provider_dispatch_cloud.py — request/response shape tests
for the cloud-native provider adapters (vertex_ai, azure_openai, bedrock,
huawei_modelarts) in runtime/provider_dispatch.py.

All four are mocked: credential acquisition (OAuth2 token fetch, boto3
session/SigV4 signer) is patched out so these run without live cloud
credentials or network access. They assert request URL/headers/body shape
and response parsing only — they do not validate against a real cloud
account.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from runtime.provider_dispatch import (
    build_cloud_request,
    get_cloud_adapter,
    is_cloud_provider,
    parse_cloud_response,
)

MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Say hi"},
]


def test_is_cloud_provider():
    for name in ("vertex_ai", "azure_openai", "bedrock", "huawei_modelarts"):
        assert is_cloud_provider(name)
    for name in ("anthropic", "openai", "ollama"):
        assert not is_cloud_provider(name)


def test_unknown_cloud_provider_raises():
    with pytest.raises(ValueError):
        get_cloud_adapter("not_a_provider")


def test_vertex_ai_anthropic_publisher():
    fake_creds = MagicMock(token="fake-token", expiry=None)
    with (
        patch("google.auth.default", return_value=(fake_creds, "proj")),
        patch("google.auth.transport.requests.Request", return_value=MagicMock()),
    ):
        url, headers, body = build_cloud_request(
            "vertex_ai",
            "claude-sonnet-4-6",
            MESSAGES,
            cfg={
                "project": "my-proj",
                "region": "us-central1",
                "publisher": "anthropic",
            },
            max_tokens=64,
            temperature=0.2,
        )
    assert url == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj"
        "/locations/us-central1/publishers/anthropic/models/claude-sonnet-4-6:streamRawPredict"
    )
    assert headers["Authorization"] == "Bearer fake-token"
    assert body["system"] == "You are terse."
    assert body["messages"] == [{"role": "user", "content": "Say hi"}]

    text, in_tok, out_tok = parse_cloud_response(
        "vertex_ai",
        {
            "content": [{"text": "hi!"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )
    assert (text, in_tok, out_tok) == ("hi!", 5, 2)


def test_vertex_ai_default_publisher_is_gemini():
    """Default publisher (no `publisher` in cfg) is google/Gemini, not anthropic."""
    fake_creds = MagicMock(token="fake-token", expiry=None)
    with (
        patch("google.auth.default", return_value=(fake_creds, "proj")),
        patch("google.auth.transport.requests.Request", return_value=MagicMock()),
    ):
        url, _, body = build_cloud_request(
            "vertex_ai",
            "gemini-1.5-pro",
            MESSAGES,
            cfg={"project": "my-proj"},
            max_tokens=64,
            temperature=0.2,
        )
    assert "/publishers/google/models/" in url
    assert url.endswith(":generateContent")
    assert "generationConfig" in body


def test_vertex_ai_gemini_publisher():
    fake_creds = MagicMock(token="fake-token", expiry=None)
    with (
        patch("google.auth.default", return_value=(fake_creds, "proj")),
        patch("google.auth.transport.requests.Request", return_value=MagicMock()),
    ):
        url, _headers, body = build_cloud_request(
            "vertex_ai",
            "gemini-1.5-pro",
            MESSAGES,
            cfg={"project": "my-proj", "publisher": "google"},
            max_tokens=64,
            temperature=0.2,
        )
    assert url.endswith("/publishers/google/models/gemini-1.5-pro:generateContent")
    assert body["generationConfig"]["maxOutputTokens"] == 64

    text, in_tok, out_tok = parse_cloud_response(
        "vertex_ai",
        {
            "candidates": [{"content": {"parts": [{"text": "hi!"}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
        },
    )
    assert (text, in_tok, out_tok) == ("hi!", 5, 2)


def test_azure_openai():
    with patch.dict("os.environ", {"AZURE_OPENAI_API_KEY": "az-key"}):
        url, headers, body = build_cloud_request(
            "azure_openai",
            "gpt-4o",
            MESSAGES,
            cfg={"resource": "my-resource", "deployment": "gpt-4o-prod"},
            max_tokens=64,
            temperature=0.2,
        )
    assert url == (
        "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-prod"
        "/chat/completions?api-version=2024-06-01"
    )
    assert headers["api-key"] == "az-key"
    assert "Authorization" not in headers
    assert body["messages"] == MESSAGES

    text, in_tok, out_tok = parse_cloud_response(
        "azure_openai",
        {
            "choices": [{"message": {"content": "hi!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )
    assert (text, in_tok, out_tok) == ("hi!", 5, 2)


def test_bedrock():
    fake_request = MagicMock(
        headers={"Authorization": "AWS4-HMAC-SHA256 ...", "X-Amz-Date": "..."}
    )
    with (
        patch("boto3.Session") as mock_session,
        patch("botocore.awsrequest.AWSRequest", return_value=fake_request),
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_sigv4.return_value.add_auth = MagicMock()
        url, headers, body = build_cloud_request(
            "bedrock",
            "anthropic.claude-3-5-sonnet",
            MESSAGES,
            cfg={"region": "us-east-1"},
            max_tokens=64,
            temperature=0.2,
        )
    assert (
        url
        == "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/invoke"
    )
    assert "Authorization" in headers
    assert body["system"] == "You are terse."

    text, in_tok, out_tok = parse_cloud_response(
        "bedrock",
        {
            "content": [{"text": "hi!"}],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        },
    )
    assert (text, in_tok, out_tok) == ("hi!", 5, 2)


def test_vertex_ai_default_region_is_verified_working_region():
    """Default region (no `region` in cfg) is us-central1 — confirmed live to serve
    gemini-2.5-flash. A GCC default (me-central1/me-central2) was tried and reverted
    after a live test against a real Vertex AI project returned 404 "Publisher model
    ... not found" for both — see VertexAIAdapter's docstring."""
    fake_creds = MagicMock(token="fake-token", expiry=None)
    with (
        patch("google.auth.default", return_value=(fake_creds, "proj")),
        patch("google.auth.transport.requests.Request", return_value=MagicMock()),
    ):
        url, _, _ = build_cloud_request(
            "vertex_ai",
            "gemini-2.5-flash",
            MESSAGES,
            cfg={"project": "my-proj"},
            max_tokens=64,
            temperature=0.2,
        )
    assert "us-central1" in url
    assert "me-central1" not in url and "me-central2" not in url


def test_vertex_ai_project_supports_env_var_expansion():
    """`project` supports ${VAR} expansion so a real project id is never committed literally."""
    fake_creds = MagicMock(token="fake-token", expiry=None)
    with (
        patch("google.auth.default", return_value=(fake_creds, "proj")),
        patch("google.auth.transport.requests.Request", return_value=MagicMock()),
        patch.dict("os.environ", {"GCP_PROJECT_ID": "real-project-123"}),
    ):
        url, _, _ = build_cloud_request(
            "vertex_ai",
            "gemini-2.5-flash",
            MESSAGES,
            cfg={"project": "${GCP_PROJECT_ID}"},
            max_tokens=64,
            temperature=0.2,
        )
    assert "real-project-123" in url


def test_bedrock_default_region_is_us_east_1():
    """Default region (no `region` in cfg) is us-east-1, not an unverified GCC region —
    no AWS credentials were available to repeat the live test done for Vertex AI, so the
    GCC default was not kept without verification."""
    fake_request = MagicMock(headers={"Authorization": "AWS4-HMAC-SHA256 ..."})
    with (
        patch("boto3.Session") as mock_session,
        patch("botocore.awsrequest.AWSRequest", return_value=fake_request),
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_sigv4.return_value.add_auth = MagicMock()
        url, _, _ = build_cloud_request(
            "bedrock",
            "anthropic.claude-3-5-sonnet",
            MESSAGES,
            cfg={},
            max_tokens=64,
            temperature=0.2,
        )
    assert "us-east-1" in url
    assert "me-central-1" not in url


def test_vertex_ai_url_template_override():
    fake_creds = MagicMock(token="fake-token", expiry=None)
    with (
        patch("google.auth.default", return_value=(fake_creds, "proj")),
        patch("google.auth.transport.requests.Request", return_value=MagicMock()),
    ):
        url, _, _ = build_cloud_request(
            "vertex_ai",
            "claude-sonnet-4-6",
            MESSAGES,
            cfg={
                "project": "my-proj",
                "url_template": "https://private-vpc.internal/projects/{project}/models/{model_id}",
            },
            max_tokens=64,
            temperature=0.2,
        )
    assert (
        url == "https://private-vpc.internal/projects/my-proj/models/claude-sonnet-4-6"
    )


def test_bedrock_url_template_override():
    fake_request = MagicMock(headers={"Authorization": "AWS4-HMAC-SHA256 ..."})
    with (
        patch("boto3.Session") as mock_session,
        patch(
            "botocore.awsrequest.AWSRequest", return_value=fake_request
        ) as mock_aws_request,
        patch("botocore.auth.SigV4Auth") as mock_sigv4,
    ):
        mock_session.return_value.get_credentials.return_value = MagicMock()
        mock_sigv4.return_value.add_auth = MagicMock()
        url, _, _ = build_cloud_request(
            "bedrock",
            "anthropic.claude-3-5-sonnet",
            MESSAGES,
            cfg={
                "region": "us-east-1",
                "url_template": "https://vpce-xxxx.bedrock-runtime.{region}.vpce.amazonaws.com/model/{model_id}/invoke",
            },
            max_tokens=64,
            temperature=0.2,
        )
    assert url == (
        "https://vpce-xxxx.bedrock-runtime.us-east-1.vpce.amazonaws.com"
        "/model/anthropic.claude-3-5-sonnet/invoke"
    )
    # signing must happen against the overridden URL, not the public default
    assert mock_aws_request.call_args.kwargs["url"] == url


def test_huawei_modelarts_path_template_override():
    with patch.dict(
        "os.environ", {"HUAWEICLOUD_SDK_AK": "ak", "HUAWEICLOUD_SDK_SK": "sk"}
    ):
        url, headers, _ = build_cloud_request(
            "huawei_modelarts",
            "my-model",
            MESSAGES,
            cfg={
                "endpoint": "xxxxx.cn-north-4.modelarts-infer.com",
                "path_template": "/v2/infers/{model_id}/chat/completions",
            },
            max_tokens=64,
            temperature=0.2,
        )
    assert (
        url
        == "https://xxxxx.cn-north-4.modelarts-infer.com/v2/infers/my-model/chat/completions"
    )
    assert "SignedHeaders=content-type;host;x-sdk-date" in headers["Authorization"]


def test_huawei_modelarts():
    with patch.dict(
        "os.environ", {"HUAWEICLOUD_SDK_AK": "ak", "HUAWEICLOUD_SDK_SK": "sk"}
    ):
        url, headers, body = build_cloud_request(
            "huawei_modelarts",
            "my-model",
            MESSAGES,
            cfg={"endpoint": "xxxxx.cn-north-4.modelarts-infer.com"},
            max_tokens=64,
            temperature=0.2,
        )
    assert (
        url
        == "https://xxxxx.cn-north-4.modelarts-infer.com/v1/infers/my-model/chat/completions"
    )
    assert headers["Authorization"].startswith("SDK-HMAC-SHA256 Access=ak")
    assert headers["Host"] == "xxxxx.cn-north-4.modelarts-infer.com"
    assert body["messages"] == MESSAGES

    text, in_tok, out_tok = parse_cloud_response(
        "huawei_modelarts",
        {
            "choices": [{"message": {"content": "hi!"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        },
    )
    assert (text, in_tok, out_tok) == ("hi!", 5, 2)


# ── An empty completion must not crash a guardrail, on ANY route ──────────────
#
# `parse_response` was hardened for `"content": null` after a null completion
# travelled out of the gateway and killed the PII scrubber with a TypeError
# several frames from the cause. The four cloud adapters each carried their own
# byte-identical copy of the unhardened parse and were missed, so the same
# response still crashed on Azure, Huawei, Bedrock and Vertex. They now share
# one parser; these are the cases that would have caught the divergence.


@pytest.mark.parametrize("provider", ["azure_openai", "huawei_modelarts"])
def test_a_null_openai_style_completion_yields_empty_text(provider: str) -> None:
    text, in_tok, out_tok = parse_cloud_response(
        provider,
        {
            "choices": [{"message": {"content": None}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 0},
        },
    )
    assert (text, in_tok, out_tok) == ("", 7, 0)


@pytest.mark.parametrize("provider", ["azure_openai", "huawei_modelarts"])
def test_a_missing_content_key_yields_empty_text(provider: str) -> None:
    """A stop before any content was emitted omits the key entirely."""
    text, _, _ = parse_cloud_response(
        provider, {"choices": [{"message": {}}], "usage": {}}
    )
    assert text == ""


@pytest.mark.parametrize("provider", ["bedrock", "vertex_ai"])
def test_an_empty_anthropic_style_content_array_yields_empty_text(provider: str) -> None:
    """The Anthropic-shaped equivalent: `content: []`. Indexing it raised
    IndexError inside the guardrail instead of returning nothing to interpret."""
    text, in_tok, out_tok = parse_cloud_response(
        provider, {"content": [], "usage": {"input_tokens": 4, "output_tokens": 0}}
    )
    assert (text, in_tok, out_tok) == ("", 4, 0)


def test_every_cloud_adapter_delegates_to_a_shared_parser() -> None:
    """No adapter may reintroduce its own copy.

    Four identical hand-written parsers is how the hardening was applied once
    and missed four times; matched on the AST so a comment cannot satisfy it.
    """
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parent.parent / "provider_dispatch.py"
    tree = ast.parse(src.read_text())
    shared = {"parse_openai_completion", "parse_anthropic_completion"}

    for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
        for fn in [f for f in cls.body if isinstance(f, ast.FunctionDef)
                   and f.name == "parse_response"]:
            if cls.name == "CloudProviderAdapter":
                continue                      # Protocol stub: body is `...`
            called = {
                n.func.id for n in ast.walk(fn)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            }
            assert called & shared, (
                f"{cls.name}.parse_response does not call a shared parser; "
                f"an inline copy will miss the next hardening fix"
            )
