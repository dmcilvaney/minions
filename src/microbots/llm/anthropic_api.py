import json
import os
from dataclasses import asdict
from logging import getLogger

from dotenv import load_dotenv
from anthropic import Anthropic
from microbots.llm.llm import LLMAskResponse, LLMInterface, LLMAskResponseModel

logger = getLogger(__name__)

load_dotenv()

endpoint = os.getenv("ANTHROPIC_END_POINT")
deployment_name = os.getenv("ANTHROPIC_DEPLOYMENT_NAME")
api_key = os.getenv("ANTHROPIC_API_KEY")


class AnthropicApi(LLMInterface):

    def __init__(self, system_prompt, deployment_name=deployment_name, max_retries=3):
        self.ai_client = Anthropic(
            api_key=api_key,
            base_url=endpoint
        )
        self.deployment_name = deployment_name
        super().__init__(system_prompt=system_prompt, max_retries=max_retries)

    def ask(self, message) -> LLMAskResponse:
        self.messages.append({"role": "user", "content": message})

        # Use beta.messages.parse() for structured outputs with Pydantic
        response = self.ai_client.beta.messages.parse(
            model=self.deployment_name,
            system=self.system_prompt,
            messages=self.messages,
            max_tokens=4096,
            betas=["structured-outputs-2025-11-13"],
            output_format=LLMAskResponseModel,
        )

        # Parse structured response from Anthropic
        parsed = response.parsed_output
        if parsed is None:
            raise ValueError("Structured output parsing failed - received None from Anthropic API")

        askResponse = LLMAskResponse(
            task_done=parsed.task_done,
            command=parsed.command,
            thoughts=parsed.thoughts or "",  # Convert None to empty string
        )

        self.messages.append({"role": "assistant", "content": json.dumps(asdict(askResponse))})

        return askResponse

    def clear_history(self):
        self.messages = []
        return True
