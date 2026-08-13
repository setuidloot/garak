"""OpenAI Responses generator with one neutral Code Interpreter tool.

This generator exposes the built-in sandboxed Python environment without
prompting the target to use it. Probes continue to supply ordinary user text;
the evaluated target independently decides whether a tool call is useful.
"""

import logging
from typing import List, Union

import backoff
import openai

from garak import _config
from garak.attempt import Conversation, Message
from garak.exception import GarakException
from garak.generators.base import Generator


class OpenAICodeInterpreter(Generator):
    """Run an OpenAI Responses target with only Code Interpreter available."""

    container_memory_limit: str
    max_output_tokens: int
    tool_choice: str

    ENV_VAR = "OPENAI_API_KEY"
    active = False
    supports_multiple_generations = False
    parallel_capable = False
    generator_family_name = "OpenAI Responses + Code Interpreter"

    DEFAULT_PARAMS = Generator.DEFAULT_PARAMS | {
        "max_tokens": None,
        "max_output_tokens": 5000,
        "container_memory_limit": "1g",
        "tool_choice": "auto",
    }

    _unsafe_attributes = ["client", "_containers", "_branches"]

    def __init__(self, name="", config_root=_config):
        self.name = name
        self._load_config(config_root)
        self.key_env_var = self.ENV_VAR
        self.client = None
        self._containers = {}
        self._branches = {}
        self._load_unsafe()
        super().__init__(self.name, config_root=config_root)

    def _load_unsafe(self) -> None:
        self.client = openai.OpenAI(api_key=self.api_key)
        if not self.name:
            raise ValueError(
                "Target name is required for OpenAI Code Interpreter, use --target_name"
            )

    @staticmethod
    def _transcript_key(prompt: Conversation) -> tuple[tuple[str, str], ...]:
        return tuple((turn.role, turn.content.text) for turn in prompt.turns)

    @backoff.on_exception(
        backoff.fibo,
        (
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
        max_value=70,
    )
    def _new_container(self):
        try:
            container = self.client.containers.create(
                name="garak-code-interpreter",
                memory_limit=self.container_memory_limit,
            )
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            message = (
                f"OpenAI API authentication failed (HTTP {error.status_code}); "
                f"verify {self.key_env_var} is valid. Original error: {error}"
            )
            logging.error(message)
            raise GarakException(message) from None
        self._containers[container.id] = container
        return container

    @backoff.on_exception(
        backoff.fibo,
        (
            openai.RateLimitError,
            openai.InternalServerError,
            openai.APITimeoutError,
            openai.APIConnectionError,
        ),
        max_value=70,
    )
    def _create_response(self, request):
        try:
            return self.client.responses.create(**request)
        except (openai.AuthenticationError, openai.PermissionDeniedError) as error:
            message = (
                f"OpenAI API authentication failed (HTTP {error.status_code}); "
                f"verify {self.key_env_var} is valid. Original error: {error}"
            )
            logging.error(message)
            raise GarakException(message) from None
        except (
            openai.BadRequestError,
            openai.NotFoundError,
            openai.UnprocessableEntityError,
        ) as error:
            logging.error("OpenAI Responses request failed: %s", error)
            return None

    def _call_model(
        self, prompt: Conversation, generations_this_call: int = 1
    ) -> List[Union[Message, None]]:
        if self.client is None:
            self._load_unsafe()
        parent_key = self._transcript_key(prompt)[:-1]
        branch = self._branches.get(parent_key)
        if branch is None:
            container = self._new_container()
            request_input = [
                {"role": turn.role, "content": turn.content.text}
                for turn in prompt.turns
            ]
            previous_response_id = None
        else:
            previous_response_id, container = branch
            request_input = [
                {
                    "role": prompt.turns[-1].role,
                    "content": prompt.turns[-1].content.text,
                }
            ]
        request = {
            "model": self.name,
            "input": request_input,
            "tools": [
                {
                    "type": "code_interpreter",
                    "container": container.id,
                }
            ],
            "tool_choice": self.tool_choice,
            "max_output_tokens": self.max_output_tokens,
        }
        if previous_response_id is not None:
            request["previous_response_id"] = previous_response_id
        response = self._create_response(request)
        if response is None or getattr(response, "status", None) == "failed":
            return [None]
        child_key = self._transcript_key(prompt) + (
            ("assistant", response.output_text),
        )
        self._branches[child_key] = (response.id, container)
        output_types = [item.type for item in response.output]
        incomplete_details = getattr(response, "incomplete_details", None)
        if hasattr(incomplete_details, "model_dump"):
            incomplete_details = incomplete_details.model_dump()
        return [
            Message(
                response.output_text,
                notes={
                    "openai_response_status": response.status,
                    "openai_response_output_types": output_types,
                    "openai_response_incomplete_details": incomplete_details,
                },
            )
        ]

    def close(self) -> None:
        """Delete hosted containers and close the OpenAI client."""

        client = getattr(self, "client", None)
        if client is None:
            return
        for container_id in list(getattr(self, "_containers", {})):
            try:
                client.containers.delete(container_id)
            except (openai.APIError, OSError, RuntimeError, ValueError):
                pass
        try:
            client.close()
        except (OSError, RuntimeError, ValueError):
            pass
        self._containers = {}
        self._branches = {}
        self.client = None


DEFAULT_CLASS = "OpenAICodeInterpreter"
