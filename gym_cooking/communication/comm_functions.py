import json
from pathlib import Path

from openai import OpenAI


class CommunicationFunctions:
    PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
    SPEAK_PROMPT_PATH = PROMPTS_DIR / "speak_prompt"
    LISTEN_PROMPT_PATH = PROMPTS_DIR / "listen_prompt"

    def __init__(self, arglist):
        self.arglist = arglist

        env_path = Path(__file__).resolve().parent / ".llm-env"
        api_key = None
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() == "key":
                    api_key = value.strip()
                    break

        self.api_key = api_key
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

        self.speak_prompt_template = None
        if self.SPEAK_PROMPT_PATH.exists():
            self.speak_prompt_template = self.SPEAK_PROMPT_PATH.read_text()

        self.listen_prompt_template = None
        if self.LISTEN_PROMPT_PATH.exists():
            self.listen_prompt_template = self.LISTEN_PROMPT_PATH.read_text()

    def speak(self, name, obs, existence_beliefs, task_alloc_dist):
        task_allocation = task_alloc_dist.get_max()
        if self.speak_prompt_template is None:
            raise FileNotFoundError(
                f"Prompt file not found at {self.SPEAK_PROMPT_PATH}"
            )
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (missing API key).")
        prompt = self.speak_prompt_template.format(
            agent_name=name, task_allocation=task_allocation
        )
        response = self.client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
        return response.output_text

    def listen(self, name, obs, existence_beliefs, task_alloc_dist) -> int:
        messages = {k: v for k, v in obs.comms if k != name}
        task_allocs = list(task_alloc_dist.keys)

        if self.listen_prompt_template is None:
            raise FileNotFoundError(
                f"Prompt file not found at {self.LISTEN_PROMPT_PATH}"
            )
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (missing API key).")

        prompt = self.listen_prompt_template.format(
            agent_name=name,
            message_dict=messages,
            task_allocations=task_allocs,
        )

        response = self.client.responses.create(
            model="gpt-4.1",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "task_allocation_selection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_index": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": max(0, len(task_allocs) - 1),
                                "description": "0-based index of the selected task allocation",
                            }
                        },
                        "required": ["selected_index"],
                        "additionalProperties": False,
                    },
                }
            },
        )

        # With structured outputs, output_text should be valid JSON matching the schema.
        try:
            data = json.loads(response.output_text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Model returned non-JSON output: {response.output_text!r}"
            ) from e

        if "selected_index" not in data:
            raise ValueError(f"Missing 'selected_index' in model output: {data!r}")

        selected_index = data["selected_index"]

        if not isinstance(selected_index, int):
            raise TypeError(
                f"'selected_index' must be an int, got {type(selected_index).__name__}"
            )

        if not (0 <= selected_index < len(task_allocs)):
            raise ValueError(
                f"'selected_index' out of range: {selected_index}; "
                f"expected 0 <= index < {len(task_allocs)}"
            )

        task_alloc = task_allocs[selected_index]
        return task_alloc
