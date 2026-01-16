from dataclasses import dataclass
from abc import ABC, abstractmethod
from logging import getLogger
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = getLogger(__name__)


llm_output_format_str = """
{
    "task_done": <bool>,  // Indicates if the task is completed
    "thoughts": <str>,     // The reasoning behind the decision
    "command": <str>     // The command to be executed
}
"""

@dataclass
class LLMAskResponse:
    task_done: bool = False
    thoughts: str = ""
    command: str = ""


class LLMAskResponseModel(BaseModel):
    """Pydantic model for structured output validation"""
    model_config = ConfigDict(extra='forbid', strict=True)  # Prevents additional fields, strict type checking

    task_done: bool = Field(
        description="Set to true only when the entire task is complete and no more commands need to be executed. Set to false if you need to run more commands."
    )
    thoughts: str | None = Field(
        default=None,
        description="Your reasoning, observations, or analysis. Explain what you learned from command outputs, what you're planning to do next, or why you're taking a particular action."
    )
    command: str = Field(
        description="The shell command to execute (when task_done=false). Must be empty string when task_done=true. Use clear, specific commands. Available: cd, ls, cat, grep, find, sed, echo, python, pytest, git, etc."
    )

    @field_validator('command')
    @classmethod
    def validate_command(cls, v: str, info) -> str:
        """Validate command field based on task_done state"""
        task_done = info.data.get('task_done', False)

        if task_done and v.strip():
            raise ValueError("Command must be empty when task_done is true")

        if not task_done and not v.strip():
            raise ValueError("Command must be non-empty when task_done is false")

        return v

class LLMInterface(ABC):
    def __init__(self, system_prompt: str, max_retries: int = 3):
        self.system_prompt = system_prompt
        self.max_retries = max_retries
        self.retries = 0
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    @abstractmethod
    def ask(self, message: str) -> LLMAskResponse:
        pass

    @abstractmethod
    def clear_history(self) -> bool:
        pass

    def summarize_context(self, last_n_messages: int = 10, summary: str="") -> dict:
        """
        It is a helper function for the LLM to summarize its own context.
        Leave the last N messages and add the summary between system prompt and the last N messages.

        summary can be empty. If empty, empty summary will be added.
        """
        logger.debug("Messages : %s", self.messages)
        # Keep the system prompt
        msg0 = self.messages[0]["content"]
        # Pop the last message which asked for summarization
        self.messages.pop()
        # Get the last N conversations (user + assistant)
        # If there are not enough messages, take all except system prompt
        if (len(self.messages) > (last_n_messages*2 - 1)):
            logger.debug("Summarizing last %d messages", last_n_messages)
            recent_messages = self.messages[-(last_n_messages*2 - 1):]
        else:
            logger.debug("Not enough messages to summarize, taking all except system prompt")
            recent_messages = self.messages[1:]
        logger.debug("Recent messages that will not be summarized: %s", recent_messages)

        # Update system prompt if it already has a summary
        # summary will be between __summary__ and __end_summary__
        if "__summary__" in msg0:
            system_prompt = msg0.split("__summary__")[0]
            old_summary = msg0.split("__end_summary__")[0].split("__summary__")[1]
            logger.debug("Old summary found: %s", old_summary)
            combined_summary = old_summary + "\n" + summary
        else:
            system_prompt = msg0
            combined_summary = summary

        new_system_prompt = f"{system_prompt}\n__summary__\n{combined_summary}\n__end_summary__"

        # Append without previous user message
        self.messages = [{"role": "system", "content": new_system_prompt}] + recent_messages[:-1]
        logger.debug("Context summarized. New system prompt: %s", new_system_prompt)

        logger.debug("Last message before summarization: %s", recent_messages[-1])
        return recent_messages[-1]  # return the last user message that given before summarization
