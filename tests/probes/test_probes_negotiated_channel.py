import json

import pytest

from garak import _config
from garak.attempt import Message
from garak.generators.base import Generator
from garak.probes.negotiated_channel import (
    OPENING_STYLES,
    EncryptedPayload,
    _X25519Channel,
    _b64u_decode,
    _b64u_encode,
    _extract_json_object,
)


class _ChannelTarget(Generator):
    """Test target that implements the negotiated-channel protocol."""

    supports_multiple_generations = False
    generator_family_name = "Test"
    name = "Negotiated channel test target"

    def __init__(self, mode="comply", config_root=_config):
        self.mode = mode
        self.sessions = {}
        super().__init__(config_root=config_root)

    def _call_model(self, prompt, generations_this_call=1):
        request = _extract_json_object(prompt.last_message().text)
        session_id = request["session_id"]
        if request.get("protocol"):
            channel = _X25519Channel.create(session_id)
            keys = channel.derive(request["public_key"])
            self.sessions[session_id] = (channel, keys)
            return [
                Message(
                    json.dumps(
                        {
                            "v": 1,
                            "session_id": session_id,
                            "public_key": channel.public_key_text(),
                        }
                    )
                )
            ]

        if self.mode == "reflect":
            return [Message(json.dumps(request))]

        channel, keys = self.sessions[session_id]
        plaintext = channel.open(request, keys.to_target, "to-target")
        text = plaintext.split("TEXT: ", 1)[1].split("\nMARKER: ", 1)[0]
        marker = plaintext.split("\nMARKER: ", 1)[1]
        if self.mode == "refuse":
            text = "I will not follow the concealed instruction."
            marker = ""
        response = channel.seal(f"{text}\n{marker}", keys.to_evaluator, "to-evaluator")
        return [Message(json.dumps(response))]


@pytest.fixture
def probe():
    _config.load_base_config()
    instance = EncryptedPayload(config_root=_config)
    instance._cases = [(OPENING_STYLES[0], "test payload", None)]
    instance.generations = 1
    return instance


def _run(probe, target, tmp_path):
    report_path = tmp_path / "report.jsonl"
    with report_path.open("w+", encoding="utf-8") as report_file:
        _config.transient.reportfile = report_file
        _config.transient.report_filename = str(report_path)
        return probe.probe(target)


def test_base64url_round_trip_and_validation():
    value = bytes(range(32))

    assert _b64u_decode(_b64u_encode(value)) == value, "base64url should round-trip"
    with pytest.raises(ValueError, match="invalid base64url"):
        _b64u_decode("not valid!")
    with pytest.raises(ValueError, match="invalid base64url"):
        _b64u_decode("////")


def test_extract_json_tolerates_code_fences():
    parsed = _extract_json_object('answer:\n```json\n{"value": 3}\n```')

    assert parsed == {"value": 3}, "the first response JSON object should be parsed"


def test_probe_completes_authenticated_exchange(probe, tmp_path):
    target = _ChannelTarget(config_root=_config)

    attempts = _run(probe, target, tmp_path)

    assert len(attempts) == 2, "a successful exchange should use two target calls"
    assert attempts[0].notes["key_share_results"] == [
        True
    ], "the target key share should be accepted"
    assert attempts[0].notes["key_exchange_attempt_results"] == [
        True
    ], "a valid key share should record negotiation engagement"
    assert attempts[1].notes["authenticated_results"] == [
        True
    ], "the response should authenticate under the negotiated receive key"
    assert attempts[1].notes["payload_results"] == [
        True
    ], "the decrypted response should contain both concealed triggers"
    assert (
        attempts[1].notes["decrypted_outputs"][0].startswith("test payload\n")
    ), "the probe should retain evaluator-visible decrypted evidence"
    assert probe._sessions == {}, "ephemeral evaluator keys should be discarded"
    json.dumps([attempt.as_dict() for attempt in attempts])


def test_directional_keys_reject_ciphertext_reflection(probe, tmp_path):
    target = _ChannelTarget(mode="reflect", config_root=_config)

    attempts = _run(probe, target, tmp_path)

    assert attempts[-1].notes["authenticated_results"] == [
        False
    ], "a reflected request must not authenticate as a target response"
    assert attempts[-1].notes["payload_results"] == [
        False
    ], "ciphertext reflection must not count as concealed-payload compliance"


def test_authenticated_refusal_is_not_payload_compliance(probe, tmp_path):
    target = _ChannelTarget(mode="refuse", config_root=_config)

    attempts = _run(probe, target, tmp_path)

    assert attempts[-1].notes["authenticated_results"] == [
        True
    ], "a valid encrypted refusal should demonstrate channel capability"
    assert attempts[-1].notes["payload_results"] == [
        False
    ], "channel capability should not be confused with payload compliance"


def test_payload_intent_takes_priority(probe):
    probe._cases = [(OPENING_STYLES[0], "test payload", "T009ignore")]

    attempts = list(probe._create_init_attempts())

    assert (
        attempts[0].intent == "T009ignore"
    ), "payload intent should override the probe default"
    probe._sessions.clear()


def test_invalid_key_share_does_not_open_payload_phase(probe, tmp_path):
    class InvalidKeyTarget(Generator):
        """Test target that returns a malformed X25519 public key."""

        supports_multiple_generations = False
        generator_family_name = "Test"
        name = "Invalid key target"

        def _call_model(self, prompt, generations_this_call=1):
            offer = _extract_json_object(prompt.last_message().text)
            return [
                Message(
                    json.dumps(
                        {
                            "v": 1,
                            "session_id": offer["session_id"],
                            "public_key": "invalid",
                        }
                    )
                )
            ]

    attempts = _run(probe, InvalidKeyTarget(config_root=_config), tmp_path)

    assert len(attempts) == 1, "an invalid key share should end the branch"
    assert attempts[0].notes["key_share_results"] == [
        False
    ], "the failed key share should be recorded"
    assert attempts[0].notes["key_exchange_attempt_results"] == [
        True
    ], "a malformed key offer should still count as negotiation engagement"
