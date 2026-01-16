import json
import os
from dataclasses import asdict

from dotenv import load_dotenv
from openai import OpenAI
from microbots.llm.llm import LLMAskResponse, LLMInterface, LLMAskResponseModel

load_dotenv()

endpoint = os.getenv("OPEN_AI_END_POINT")
deployment_name = os.getenv("OPEN_AI_DEPLOYMENT_NAME")
api_key = os.getenv("OPEN_AI_KEY")  # use the api_key


class OpenAIApi(LLMInterface):

    def __init__(self, system_prompt, deployment_name=deployment_name, max_retries=3):
        self.deployment_name = deployment_name
        self.ai_client = OpenAI(base_url=f"{endpoint}", api_key=api_key)
        super().__init__(system_prompt=system_prompt, max_retries=max_retries)

    def ask(self, message) -> LLMAskResponse:
        self.messages.append({"role": "user", "content": message})

        # Use structured output via responses.parse()
        response = self.ai_client.responses.parse(
            model=self.deployment_name,
            input=self.messages,
            text_format=LLMAskResponseModel,
        )

        # Get the parsed Pydantic object
        parsed = response.output_parsed
        if parsed is None:
            raise ValueError("Structured output parsing failed - received None from OpenAI API")

        # Convert to LLMAskResponse dataclass for backward compatibility
        askResponse = LLMAskResponse(
            task_done=parsed.task_done,
            command=parsed.command,
            thoughts=parsed.thoughts or "",  # Convert None to empty string
        )

        # Add assistant message with structured response
        self.messages.append({"role": "assistant", "content": json.dumps(asdict(askResponse))})

        return askResponse

    def clear_history(self):
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]
        return True
