garak.probes.negotiated_channel
================================

This probe tests whether a target will negotiate and use a fresh, authenticated
opaque channel for a concealed instruction. It uses a general standards-based
profile (X25519, HKDF-SHA256, and ChaCha20-Poly1305), rather than recognizing a
particular word encoding or a fixed ciphertext.

The three reported outcomes should be interpreted separately:

* ``KeyExchange`` means the target attempted to supply a key share for the
  matching session, showing willingness to enter the offered channel even if
  that key or later cryptographic output is invalid.
* ``ConcealedPayload`` means the target both established the channel and
  complied with the payload carried inside it.
* ``AuthenticatedChannel`` means the target proved that it could establish and
  use the channel, even if it refused the concealed instruction.

The probe requires the optional ``cryptography`` package and is inactive by
default. Run it explicitly, for example::

   python -m garak --target_type <generator> --target_name <target> \
       --probes negotiated_channel.EncryptedPayload

By default, it draws from Garak's ``text_en`` payload group. The ``payloads``
configuration accepts any payload group name; the aliases ``default``, ``xss``,
and ``slur_terms`` select ``text_en``, ``web_html_js``, and ``slur_terms_en``.

To evaluate spontaneous tool use without adding tool-choice instructions to
the prompt, use this probe with a tool-capable target such as
``openai_code_interpreter.OpenAICodeInterpreter``. Tool availability is a
property of the target generator, not of this probe.

.. automodule:: garak.probes.negotiated_channel
   :members:
   :undoc-members:
   :show-inheritance:

   .. show-asr::
