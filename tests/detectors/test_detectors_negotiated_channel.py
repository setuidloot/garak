from garak.attempt import Attempt, Message
from garak.detectors.negotiated_channel import (
    AuthenticatedChannel,
    ConcealedPayload,
    KeyExchange,
)


def _attempt(phase, outputs, **notes):
    attempt = Attempt(prompt=Message("test", lang="en"), notes={"phase": phase})
    attempt.outputs = [Message(output) for output in outputs]
    attempt.notes.update(notes)
    return attempt


def test_authenticated_channel_scores_verified_results():
    detector = AuthenticatedChannel()
    attempt = _attempt(
        "sealed_payload",
        ["first", "second"],
        authenticated_results=[True, False],
    )

    assert detector.detect(attempt) == [
        1.0,
        0.0,
    ], "authenticated results should map directly to detector scores"


def test_key_exchange_scores_negotiation_attempts():
    attempt = _attempt(
        "key_exchange",
        ["first", "second"],
        key_exchange_attempt_results=[True, False],
    )

    assert KeyExchange().detect(attempt) == [
        1.0,
        0.0,
    ], "negotiation attempts should be distinct from completed channel use"


def test_concealed_payload_scores_verified_results():
    detector = ConcealedPayload()
    attempt = _attempt("sealed_payload", ["response"], payload_results=[True])

    assert detector.detect(attempt) == [
        1.0
    ], "a verified concealed payload should be a hit"


def test_detectors_skip_key_exchange_phase():
    attempt = _attempt("key_exchange", ["response"])

    assert AuthenticatedChannel().detect(attempt) == [
        None
    ], "key exchange output should not be scored as an authenticated channel"
    assert ConcealedPayload().detect(attempt) == [
        None
    ], "key exchange output should not be scored as payload compliance"


def test_key_exchange_detector_skips_sealed_payload_phase():
    attempt = _attempt("sealed_payload", ["response"])

    assert KeyExchange().detect(attempt) == [
        None
    ], "sealed responses should not be scored as key exchange attempts"


def test_detectors_fail_closed_on_misaligned_results():
    attempt = _attempt("sealed_payload", ["first", "second"], payload_results=[True])

    assert ConcealedPayload().detect(attempt) == [
        None,
        None,
    ], "misaligned probe verification should not create detector hits"
