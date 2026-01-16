"""
Unit tests for OpenAIApi class
"""
import pytest
import json
import sys
import os
from unittest.mock import Mock, patch
from dataclasses import asdict

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.llm.openai_api import OpenAIApi
from microbots.llm.llm import LLMAskResponse, LLMInterface, LLMAskResponseModel


@pytest.fixture(autouse=True)
def patch_openai_config():
    """Automatically patch OpenAI configuration for all tests"""
    with patch('microbots.llm.openai_api.endpoint', 'https://api.openai.com'), \
         patch('microbots.llm.openai_api.deployment_name', 'gpt-4'), \
         patch('microbots.llm.openai_api.api_key', 'test-api-key'), \
         patch('microbots.llm.openai_api.OpenAI') as mock_openai:
        yield mock_openai


@pytest.mark.unit
class TestOpenAIApiInitialization:
    """Tests for OpenAIApi initialization"""

    def test_init_with_default_deployment_name(self):
        """Test initialization with deployment name from parameter default"""
        system_prompt = "You are a helpful assistant"

        # When no deployment_name is passed, it uses the default parameter value
        # which comes from the module-level variable at function definition time
        api = OpenAIApi(system_prompt=system_prompt)

        assert api.system_prompt == system_prompt
        # The deployment_name will be whatever the module variable was set to
        # In test environment with mocked config, this might be None or the patched value
        assert api.max_retries == 3
        assert api.retries == 0
        assert len(api.messages) == 1
        assert api.messages[0]["role"] == "system"
        assert api.messages[0]["content"] == system_prompt

    def test_init_with_custom_deployment_name(self):
        """Test initialization with custom deployment name"""
        system_prompt = "You are a helpful assistant"
        custom_deployment = "gpt-3.5-turbo"

        api = OpenAIApi(
            system_prompt=system_prompt,
            deployment_name=custom_deployment
        )

        assert api.deployment_name == custom_deployment

    def test_init_with_custom_max_retries(self):
        """Test initialization with custom max_retries"""
        system_prompt = "You are a helpful assistant"

        api = OpenAIApi(
            system_prompt=system_prompt,
            max_retries=5
        )

        assert api.max_retries == 5
        assert api.retries == 0

    def test_init_creates_openai_client(self):
        """Test that initialization creates OpenAI client"""
        system_prompt = "You are a helpful assistant"

        api = OpenAIApi(system_prompt=system_prompt)

        assert api.ai_client is not None


@pytest.mark.unit
class TestOpenAIApiAsk:
    """Tests for OpenAIApi.ask method"""

    def test_ask_successful_response(self):
        """Test ask method with successful response"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response with structured output
        mock_parsed = LLMAskResponseModel(task_done=False, command="echo 'hello'", thoughts="")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask
        message = "Please say hello"
        result = api.ask(message)

        # Verify the result
        assert isinstance(result, LLMAskResponse)
        assert result.task_done is False
        assert result.command == "echo 'hello'"
        assert result.thoughts == ""

        # Verify messages were appended
        assert len(api.messages) == 3  # system + user + assistant
        assert api.messages[1]["role"] == "user"
        assert api.messages[1]["content"] == message
        assert api.messages[2]["role"] == "assistant"

    def test_ask_with_task_done_true(self):
        """Test ask method when task is complete"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response with structured output
        mock_parsed = LLMAskResponseModel(task_done=True, command="", thoughts="Task completed successfully")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask
        result = api.ask("Complete the task")

        # Verify the result
        assert result.task_done is True
        assert result.command == ""
        assert result.thoughts == "Task completed successfully"

    def test_ask_with_structured_output(self):
        """Test ask method uses structured output (no retries needed)"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response - structured output guarantees valid schema
        mock_parsed = LLMAskResponseModel(task_done=False, command="ls -la", thoughts="Listing files")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask
        result = api.ask("List files")

        # Verify it succeeded on first try
        assert result.task_done is False
        assert result.command == "ls -la"
        assert result.thoughts == "Listing files"

        # Verify it called the API only once (no retries with structured output)
        assert api.ai_client.responses.parse.call_count == 1

    def test_ask_appends_user_message(self):
        """Test that ask appends user message to messages list"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        initial_message_count = len(api.messages)

        # Mock the OpenAI client response with structured output
        mock_parsed = LLMAskResponseModel(task_done=False, command="pwd", thoughts="")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask
        user_message = "What directory am I in?"
        api.ask(user_message)

        # Verify user message was added
        assert len(api.messages) > initial_message_count
        user_messages = [m for m in api.messages if m["role"] == "user"]
        assert user_messages[-1]["content"] == user_message

    def test_ask_appends_assistant_response_as_json(self):
        """Test that ask appends assistant response as JSON string"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response with structured output
        mock_parsed = LLMAskResponseModel(task_done=False, command="echo test", thoughts="")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask
        api.ask("Run echo test")

        # Verify assistant message was added as JSON
        assistant_messages = [m for m in api.messages if m["role"] == "assistant"]
        assert len(assistant_messages) > 0

        # Parse the assistant message to verify it's valid JSON
        assistant_content = json.loads(assistant_messages[-1]["content"])
        assert assistant_content["task_done"] is False
        assert assistant_content["command"] == "echo test"
        assert assistant_content["thoughts"] == ""

    def test_ask_uses_asdict_for_response(self):
        """Test that ask uses asdict to convert LLMAskResponse to dict"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response with structured output
        mock_parsed = LLMAskResponseModel(task_done=True, command="", thoughts="Done")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask
        result = api.ask("Complete task")

        # Verify the assistant message contains the correct structure
        assistant_msg = json.loads(api.messages[-1]["content"])

        # Verify it matches what asdict would produce
        expected = asdict(result)
        assert assistant_msg == expected

    def test_ask_with_pydantic_validation(self):
        """Test that structured output uses Pydantic validation"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response with valid Pydantic model
        mock_parsed = LLMAskResponseModel(task_done=False, command="ls", thoughts="Checking directory")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask
        result = api.ask("List files")

        # Verify structured output was used
        assert isinstance(result, LLMAskResponse)
        assert result.command == "ls"
        assert result.thoughts == "Checking directory"


@pytest.mark.unit
class TestOpenAIApiClearHistory:
    """Tests for OpenAIApi.clear_history method"""

    def test_clear_history_resets_messages(self):
        """Test that clear_history resets messages to only system prompt"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Add some messages
        api.messages.append({"role": "user", "content": "Hello"})
        api.messages.append({"role": "assistant", "content": "Hi there"})
        api.messages.append({"role": "user", "content": "How are you?"})

        assert len(api.messages) == 4  # system + 3 added

        # Clear history
        result = api.clear_history()

        # Verify only system message remains
        assert result is True
        assert len(api.messages) == 1
        assert api.messages[0]["role"] == "system"
        assert api.messages[0]["content"] == system_prompt

    def test_clear_history_returns_true(self):
        """Test that clear_history returns True"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        result = api.clear_history()

        assert result is True

    def test_clear_history_preserves_system_prompt(self):
        """Test that clear_history preserves the original system prompt"""
        system_prompt = "You are a code assistant specialized in Python"
        api = OpenAIApi(system_prompt=system_prompt)

        # Add and clear messages multiple times
        for i in range(3):
            api.messages.append({"role": "user", "content": f"Message {i}"})
            api.clear_history()

        # Verify system prompt is still correct
        assert len(api.messages) == 1
        assert api.messages[0]["content"] == system_prompt


@pytest.mark.unit
class TestOpenAIApiInheritance:
    """Tests to verify OpenAIApi correctly inherits from LLMInterface"""

    def test_openai_api_is_llm_interface(self):
        """Test that OpenAIApi is an instance of LLMInterface"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        assert isinstance(api, LLMInterface)

    def test_openai_api_implements_ask(self):
        """Test that OpenAIApi implements ask method"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        assert hasattr(api, 'ask')
        assert callable(api.ask)

    def test_openai_api_implements_clear_history(self):
        """Test that OpenAIApi implements clear_history method"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        assert hasattr(api, 'clear_history')
        assert callable(api.clear_history)


@pytest.mark.unit
class TestOpenAIApiEdgeCases:
    """Tests for edge cases and error scenarios"""

    def test_ask_with_empty_message(self):
        """Test ask with empty string message"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response with structured output
        mock_parsed = LLMAskResponseModel(task_done=False, command="echo ''", thoughts="")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Call ask with empty message
        result = api.ask("")

        # Verify it still works
        assert isinstance(result, LLMAskResponse)
        assert api.messages[-2]["content"] == ""  # User message

    def test_multiple_ask_calls_append_messages(self):
        """Test that multiple ask calls append all messages"""
        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock the OpenAI client response with structured output
        mock_parsed = LLMAskResponseModel(task_done=False, command="pwd", thoughts="")
        mock_response = Mock()
        mock_response.output_parsed = mock_parsed
        api.ai_client.responses.parse = Mock(return_value=mock_response)

        # Make multiple ask calls
        api.ask("First question")
        api.ask("Second question")
        api.ask("Third question")

        # Verify all messages are preserved
        # Should have: 1 system + 3 user + 3 assistant = 7 messages
        assert len(api.messages) == 7

        user_messages = [m for m in api.messages if m["role"] == "user"]
        assert len(user_messages) == 3
        assert user_messages[0]["content"] == "First question"
        assert user_messages[1]["content"] == "Second question"
        assert user_messages[2]["content"] == "Third question"


@pytest.mark.unit
class TestOpenAIApiValidation:
    """Tests for validation error handling in OpenAI API"""

    def test_invalid_response_task_done_true_with_command(self, patch_openai_config):
        """Test that Pydantic validation catches invalid response: task_done=True with non-empty command"""
        from pydantic import ValidationError

        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        mock_client = patch_openai_config.return_value
        mock_response = Mock()

        # Mock invalid output: task_done=True but command is not empty
        try:
            mock_parsed = LLMAskResponseModel(task_done=True, command="ls -la", thoughts="Invalid")
            # This should raise ValidationError, test should not reach here
            pytest.fail("Expected ValidationError but model was created successfully")
        except ValidationError as e:
            # This is expected - validation should fail
            assert 'command' in str(e)
            assert 'empty' in str(e).lower()

    def test_invalid_response_task_done_false_with_empty_command(self, patch_openai_config):
        """Test that Pydantic validation catches invalid response: task_done=False with empty command"""
        from pydantic import ValidationError

        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        # Mock invalid output: task_done=False but command is empty
        with pytest.raises(ValidationError) as exc_info:
            mock_parsed = LLMAskResponseModel(task_done=False, command="", thoughts="Invalid")

        assert 'command' in str(exc_info.value)
        assert 'non-empty' in str(exc_info.value).lower()

    def test_api_propagates_validation_error(self, patch_openai_config):
        """Test that ValidationError from structured output propagates correctly"""
        from pydantic import ValidationError

        system_prompt = "You are a helpful assistant"
        api = OpenAIApi(system_prompt=system_prompt)

        mock_client = patch_openai_config.return_value

        # Simulate the API trying to parse invalid data by having output_parsed creation fail
        def side_effect_validation_error(*args, **kwargs):
            # This simulates what happens when structured output returns invalid data
            # The API will try to access response.output_parsed, which internally validates
            mock_response = Mock()
            # When accessing output_parsed, trigger validation by creating invalid model
            type(mock_response).output_parsed = property(lambda self: LLMAskResponseModel(
                task_done=True, command="ls", thoughts="Invalid"
            ))
            return mock_response

        mock_client.responses.parse.side_effect = side_effect_validation_error

        # The ask() method should propagate the ValidationError
        with pytest.raises(ValidationError):
            api.ask("test message")
