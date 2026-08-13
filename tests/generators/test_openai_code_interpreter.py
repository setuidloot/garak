from types import SimpleNamespace

import httpx
import openai

from garak import _config
from garak.attempt import Conversation, Message, Turn
from garak.generators.openai_code_interpreter import OpenAICodeInterpreter


class _FakeContainers:
    def __init__(self):
        self.created = []
        self.deleted = []

    def create(self, **kwargs):
        self.created.append(kwargs)
        return SimpleNamespace(id="container-test")

    def delete(self, container_id):
        self.deleted.append(container_id)


class _FakeResponses:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            id=f"response-{len(self.calls)}",
            output_text="target response",
            output=[SimpleNamespace(type="message")],
            status="completed",
            incomplete_details=None,
        )


class _FakeClient:
    def __init__(self):
        self.containers = _FakeContainers()
        self.responses = _FakeResponses()
        self.closed = False

    def close(self):
        self.closed = True


def test_code_interpreter_is_neutral_single_tool(monkeypatch):
    _config.load_base_config()
    fake_client = _FakeClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "garak.generators.openai_code_interpreter.openai.OpenAI",
        lambda **kwargs: fake_client,
    )
    generator = OpenAICodeInterpreter(name="test-model", config_root=_config)
    conversation = Conversation([Turn("user", Message("ordinary probe prompt"))])

    output = generator.generate(conversation)

    assert output[0].text == "target response", "generate should return target text"
    assert output[0].notes["openai_response_output_types"] == [
        "message"
    ], "response item types should be retained as telemetry"
    request = fake_client.responses.calls[0]
    assert request["input"] == [
        {"role": "user", "content": "ordinary probe prompt"}
    ], "the public generation path should preserve ordinary probe text"
    assert request["tools"] == [
        {"type": "code_interpreter", "container": "container-test"}
    ], "exactly one neutral Code Interpreter tool should be exposed"
    assert "instructions" not in request, "the generator should not coach tool use"
    generator.close()
    assert fake_client.containers.deleted == [
        "container-test"
    ], "closing the generator should delete its hosted container"


def test_code_interpreter_container_persists_across_calls(monkeypatch):
    _config.load_base_config()
    fake_client = _FakeClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "garak.generators.openai_code_interpreter.openai.OpenAI",
        lambda **kwargs: fake_client,
    )
    generator = OpenAICodeInterpreter(name="test-model", config_root=_config)
    conversation = Conversation([Turn("user", Message("first turn"))])

    generator._call_model(conversation)
    followup = Conversation(
        [
            Turn("user", Message("first turn")),
            Turn("assistant", Message("target response")),
            Turn("user", Message("second turn")),
        ]
    )
    generator._call_model(followup)

    assert (
        len(fake_client.containers.created) == 1
    ), "one conversation branch should reuse one hosted container"
    assert all(
        call["tools"][0]["container"] == "container-test"
        for call in fake_client.responses.calls
    ), "all turns in a branch should address the same container"
    assert (
        "previous_response_id" not in fake_client.responses.calls[0]
    ), "a root request should not claim a previous response"
    assert (
        fake_client.responses.calls[1]["previous_response_id"] == "response-1"
    ), "a follow-up should preserve hidden tool-call context"
    assert fake_client.responses.calls[1]["input"] == [
        {"role": "user", "content": "second turn"}
    ], "a continued response should send only the new turn"
    generator.close()


def test_code_interpreter_retries_transient_response_errors(monkeypatch):
    _config.load_base_config()
    fake_client = _FakeClient()
    successful_create = fake_client.responses.create
    call_count = 0

    def transient_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise openai.APITimeoutError(httpx.Request("POST", "https://example.test"))
        return successful_create(**kwargs)

    fake_client.responses.create = transient_create
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "garak.generators.openai_code_interpreter.openai.OpenAI",
        lambda **kwargs: fake_client,
    )
    generator = OpenAICodeInterpreter(name="test-target", config_root=_config)
    conversation = Conversation([Turn("user", Message("retry this request"))])

    output = generator.generate(conversation)

    assert call_count == 2, "a transient timeout should retry the Responses call"
    assert output[0].text == "target response", "a successful retry should be returned"
    assert (
        len(fake_client.containers.created) == 1
    ), "retrying a response should not create a second container"
    generator.close()


def test_code_interpreter_returns_none_for_failed_response(monkeypatch):
    _config.load_base_config()
    fake_client = _FakeClient()

    def failed_create(**kwargs):
        fake_client.responses.calls.append(kwargs)
        return SimpleNamespace(status="failed")

    fake_client.responses.create = failed_create
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "garak.generators.openai_code_interpreter.openai.OpenAI",
        lambda **kwargs: fake_client,
    )
    generator = OpenAICodeInterpreter(name="test-target", config_root=_config)
    conversation = Conversation([Turn("user", Message("ordinary probe prompt"))])

    output = generator.generate(conversation)

    assert output == [None], "a terminal failed response should produce no generation"
    generator.close()


def test_code_interpreter_cleanup_continues_after_delete_error(monkeypatch):
    _config.load_base_config()
    fake_client = _FakeClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "garak.generators.openai_code_interpreter.openai.OpenAI",
        lambda **kwargs: fake_client,
    )
    generator = OpenAICodeInterpreter(name="test-target", config_root=_config)
    conversation = Conversation([Turn("user", Message("ordinary probe prompt"))])
    generator.generate(conversation)

    def failed_delete(container_id):
        raise OSError(f"could not delete {container_id}")

    fake_client.containers.delete = failed_delete
    generator.close()

    assert fake_client.closed, "client cleanup should continue after deletion fails"
    assert generator.client is None, "close should discard the client reference"
