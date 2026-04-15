import json
from pathlib import Path

import tiktoken
from openai import OpenAI


class CommunicationFunctions:
    PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
    SPEAK_PROMPT_PATH = PROMPTS_DIR / "speak_prompt"
    LISTEN_PROMPT_PATH = PROMPTS_DIR / "listen_prompt"
    LOGITS_PROMPT_PATH = PROMPTS_DIR / "logits_prompt"

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

    def speak(self, name, obs, task_allocation):
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

    def listen(self, name, comms, task_alloc_dist):
        messages = {k: v for k, v in comms.items()}
        task_allocs = [
            t for t in task_alloc_dist.probs.keys() if task_alloc_dist.get(t) != 0
        ]

        comm_info = {}

        for k, v in messages.items():
            if self.listen_prompt_template is None:
                raise FileNotFoundError(
                    f"Prompt file not found at {self.LISTEN_PROMPT_PATH}"
                )
            if self.client is None:
                raise RuntimeError(
                    "OpenAI client is not initialized (missing API key)."
                )

            prompt = self.listen_prompt_template.format(
                agent_name=name,
                message=f'{k} says: "{v}"',
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
                                },
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0.0,
                                    "maximum": 1.0,
                                    "description": "Confidence score between 0 and 1",
                                },
                            },
                            "required": ["selected_index", "confidence"],
                            "additionalProperties": False,
                        },
                    }
                },
            )

            # Parse response
            try:
                data = json.loads(response.output_text)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Model returned non-JSON output: {response.output_text!r}"
                ) from e

            if "selected_index" not in data:
                raise ValueError(f"Missing 'selected_index' in model output: {data!r}")
            if "confidence" not in data:
                raise ValueError(f"Missing 'confidence' in model output: {data!r}")

            selected_index = data["selected_index"]
            confidence = data["confidence"]

            if not isinstance(selected_index, int):
                raise TypeError(
                    f"'selected_index' must be an int, got {type(selected_index).__name__}"
                )

            if not isinstance(confidence, (int, float)):
                raise TypeError(
                    f"'confidence' must be a number, got {type(confidence).__name__}"
                )

            if not (0.0 <= confidence <= 1.0):
                raise ValueError(
                    f"'confidence' must be between 0 and 1, got {confidence}"
                )

            if not (0 <= selected_index < len(task_allocs)):
                raise ValueError(
                    f"'selected_index' out of range: {selected_index}; "
                    f"expected 0 <= index < {len(task_allocs)}"
                )

            task_alloc = tuple(task_allocs[selected_index])
            comm_info[k] = (task_alloc, confidence, v)

        return comm_info

    def get_logits(self, agent_name, comm, task_allocation):
        if self.speak_prompt_template is None:
            raise FileNotFoundError(
                f"Prompt file not found at {self.LOGITS_PROMPT_PATH}"
            )
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (missing API key).")

        prompt = self.speak_prompt_template.format(
            agent_name=agent_name,
            task_allocation=task_allocation,
        )

        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        comm_token_ids = enc.encode(comm)

        total_logprob = 0.0
        prefix = ""

        for token_id in comm_token_ids:
            gold_token = enc.decode([token_id])

            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefix},
                ],
                temperature=0,
                max_completion_tokens=1,
                logprobs=True,
                top_logprobs=20,
            )

            token_info = resp.choices[0].logprobs.content[0]
            candidates = {c.token: c.logprob for c in token_info.top_logprobs}

            if gold_token in candidates:
                total_logprob += candidates[gold_token]
            elif token_info.token == gold_token:
                total_logprob += token_info.logprob
            else:
                total_logprob += -9999.0  # Very unlikely prob from docs

            prefix += gold_token

        return total_logprob
