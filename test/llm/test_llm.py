"""
Unit tests for LLM interface and response validation
"""
import pytest
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from microbots.llm.llm import LLMInterface, LLMAskResponse, llm_output_format_str


class ConcreteLLM(LLMInterface):
    """Concrete implementation of LLMInterface for testing"""

    def __init__(self, max_retries=3):
        self.max_retries = max_retries
        self.retries = 0
        self.messages = []

    def ask(self, message: str) -> LLMAskResponse:
        """Simple implementation for testing"""
        return LLMAskResponse(task_done=False, command="test", thoughts="")

    def clear_history(self) -> bool:
        """Simple implementation for testing"""
        self.messages = []
        return True

@pytest.mark.unit
class TestLlmAskResponse:
    """Tests for LLMAskResponse dataclass"""

    def test_default_values(self):
        """Test that default values are set correctly"""
        response = LLMAskResponse()
        assert response.task_done is False
        assert response.command == ""
        assert response.thoughts == ""

    def test_custom_values(self):
        """Test creating response with custom values"""
        response = LLMAskResponse(
            task_done=True,
            command="echo 'hello'",
            thoughts="Task completed successfully"
        )
        assert response.task_done is True
        assert response.command == "echo 'hello'"
        assert response.thoughts == "Task completed successfully"

    def test_partial_initialization(self):
        """Test partial initialization with some defaults"""
        response = LLMAskResponse(command="ls -la")
        assert response.task_done is False
        assert response.command == "ls -la"
        assert response.thoughts == ""

@pytest.mark.unit
class TestValidateLlmResponse:
    """Tests for Pydantic validation behavior (replaces old _validate_llm_response tests)"""

    def test_valid_response_task_not_done(self):
        """Test validation of a valid response with task_done=False"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": False,
            "command": "echo 'hello world'",
            "thoughts": None
        })

        # Should parse successfully
        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.task_done is False
        assert parsed.command == "echo 'hello world'"
        assert parsed.thoughts is None

    def test_valid_response_task_done(self):
        """Test validation of a valid response with task_done=True"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": "Task completed successfully"
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.task_done is True
        assert parsed.command == ""
        assert parsed.thoughts == "Task completed successfully"

    def test_invalid_json(self):
        """Test validation with invalid JSON"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = "This is not valid JSON { invalid }"

        with pytest.raises((ValidationError, ValueError, json.JSONDecodeError)):
            LLMAskResponseModel.model_validate_json(response)

    def test_missing_required_fields(self):
        """Test validation with missing required fields"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            # Missing "command" and "thoughts"
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'command' for error in errors)

    def test_task_done_not_boolean(self):
        """Test validation when task_done is not a boolean"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": "yes",  # Should be boolean
            "command": "echo test",
            "thoughts": None
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        assert any(error['loc'][0] == 'task_done' for error in exc_info.value.errors())

    def test_empty_command_when_task_not_done(self):
        """Test validation when command is empty but task_done is False"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": "",  # Empty command
            "thoughts": None
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        assert 'command' in str(exc_info.value)
        assert 'non-empty' in str(exc_info.value).lower()

    def test_whitespace_only_command_when_task_not_done(self):
        """Test validation when command is whitespace only but task_done is False"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": "   ",  # Whitespace only
            "thoughts": None
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_null_command_when_task_not_done(self):
        """Test validation when command is null but task_done is False"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": None,  # Null command
            "thoughts": None
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_non_empty_command_when_task_done(self):
        """Test validation when command is not empty but task_done is True"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": True,
            "command": "echo 'should not have this'",  # Should be empty
            "thoughts": "Done"
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        assert 'command' in str(exc_info.value)
        assert 'empty' in str(exc_info.value).lower()

    def test_max_retries_exceeded(self):
        """Test that Pydantic raises ValidationError (no retry logic in Pydantic)"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": "",  # Invalid
            "thoughts": None
        })

        # Pydantic doesn't have retry logic - it just raises ValidationError
        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_retry_increments(self):
        """Test that Pydantic consistently raises errors (no retry state)"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({"task_done": "invalid"})

        # Each validation attempt raises error independently
        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_valid_response_with_result_string(self):
        """Test validation with result as a string"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": "Analysis complete: Found 5 errors"
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.thoughts == "Analysis complete: Found 5 errors"

    def test_valid_response_with_null_result(self):
        """Test validation with result as null"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": False,
            "command": "ls -la",
            "thoughts": None
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.thoughts is None

    def test_command_with_special_characters(self):
        """Test validation with command containing special characters"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": False,
            "command": "echo 'Hello \"World\"' | grep -i 'world'",
            "thoughts": None
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.command == "echo 'Hello \"World\"' | grep -i 'world'"

    def test_extra_fields_ignored(self):
        """Test that extra fields in response are forbidden"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": "echo test",
            "thoughts": None,
            "extra_field": "should be forbidden",
            "another_extra": 123
        })

        # With extra='forbid', this should raise ValidationError
        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_task_done_false_boolean(self):
        """Test validation with task_done explicitly set to False"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": False,
            "command": "pwd",
            "thoughts": None
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.task_done is False

    def test_task_done_true_boolean(self):
        """Test validation with task_done explicitly set to True"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": "All tasks completed"
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.task_done is True

    def test_command_with_newlines(self):
        """Test validation with multi-line command"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": False,
            "command": "for i in 1 2 3; do\n  echo $i\ndone",
            "thoughts": None
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert "\n" in parsed.command

    def test_error_message_appended_to_messages(self):
        """Test that Pydantic provides detailed error messages"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": "not a boolean",
            "command": "test",
            "thoughts": None
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        # Pydantic provides detailed error info
        assert len(exc_info.value.errors()) > 0

    def test_multiple_validation_failures(self):
        """Test multiple fields with validation errors"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        # Missing command field AND task_done is wrong type
        response = json.dumps({
            "task_done": "not boolean",
            "thoughts": None
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        # Should have errors for multiple fields
        errors = exc_info.value.errors()
        assert len(errors) >= 1


@pytest.mark.unit
class TestLlmOutputFormatStr:
    """Test the output format string constant"""

    def test_format_string_contains_required_fields(self):
        """Test that the format string contains all required field names"""
        assert "task_done" in llm_output_format_str
        assert "command" in llm_output_format_str
        assert "thoughts" in llm_output_format_str

    def test_format_string_contains_types(self):
        """Test that the format string shows the types"""
        assert "bool" in llm_output_format_str
        assert "str" in llm_output_format_str

@pytest.mark.unit
class TestConcreteLLMImplementation:
    """Test the concrete LLM implementation used for testing"""

    def test_ask_returns_LLMAskResponse(self):
        """Test that ask method returns correct type"""
        llm = ConcreteLLM()
        response = llm.ask("test message")

        assert isinstance(response, LLMAskResponse)

    def test_clear_history(self):
        """Test that clear_history clears messages"""
        llm = ConcreteLLM()
        llm.messages = [{"role": "user", "content": "test"}]

        result = llm.clear_history()

        assert result is True
        assert len(llm.messages) == 0

    def test_max_retries_initialization(self):
        """Test that max_retries is set correctly"""
        llm = ConcreteLLM(max_retries=5)
        assert llm.max_retries == 5
        assert llm.retries == 0

@pytest.mark.unit
class TestValidateLlmResponseAdditionalCases:
    """Additional Pydantic validation test cases"""

    def test_command_is_integer_not_string(self):
        """Test validation when command is an integer instead of string"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": 123,  # Integer, not string
            "thoughts": None
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        assert any(error['loc'][0] == 'command' for error in exc_info.value.errors())

    def test_only_task_done_field_present(self):
        """Test validation with only task_done field"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": True
            # Missing command and thoughts
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_only_command_field_present(self):
        """Test validation with only command field"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "command": "echo test"
            # Missing task_done and thoughts
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_task_done_as_string_true(self):
        """Test validation when task_done is string 'true' instead of boolean"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": "true",  # String instead of boolean
            "command": "",
            "thoughts": None
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_task_done_as_integer(self):
        """Test validation when task_done is integer (2) instead of boolean"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": 2,  # Integer that's not 0 or 1
            "command": "test",
            "thoughts": None
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_command_with_only_spaces_when_task_not_done(self):
        """Test that command with only spaces is invalid when task_done=False"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": "     ",  # Only spaces
            "thoughts": None
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_command_with_tabs_when_task_not_done(self):
        """Test that command with tabs/whitespace is invalid when task_done=False"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": False,
            "command": "\t\t\t",  # Only tabs
            "thoughts": None
        })

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_command_with_leading_trailing_spaces_valid(self):
        """Test that command with leading/trailing spaces but content is valid"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": False,
            "command": "  echo test  ",  # Has actual content
            "thoughts": None
        })

        # This should be valid because strip() shows it has content
        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.command == "  echo test  "

    def test_task_done_true_with_whitespace_command(self):
        """Test that task_done=True with whitespace-only command is valid"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": True,
            "command": "   ",  # Whitespace treated as empty
            "thoughts": "Done"
        })

        # This is valid because strip() == "", which is allowed when task_done=True
        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.task_done is True

    def test_json_with_comments_fails(self):
        """Test that JSON with comments fails to parse"""
        from microbots.llm.llm import LLMAskResponseModel

        response = """{
            "task_done": false,  // This is a comment
            "command": "test",
            "thoughts": null
        }"""

        with pytest.raises((json.JSONDecodeError, ValueError)):
            LLMAskResponseModel.model_validate_json(response)

    def test_empty_string_response(self):
        """Test validation with empty string response"""
        from microbots.llm.llm import LLMAskResponseModel

        response = ""

        with pytest.raises((json.JSONDecodeError, ValueError)):
            LLMAskResponseModel.model_validate_json(response)

    def test_array_response(self):
        """Test validation with JSON array instead of object"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps([{"task_done": False, "command": "test"}])

        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_result_with_empty_string(self):
        """Test that result can be an empty string"""
        from microbots.llm.llm import LLMAskResponseModel

        response = json.dumps({
            "task_done": True,
            "command": "",
            "thoughts": ""  # Empty string result
        })

        parsed = LLMAskResponseModel.model_validate_json(response)
        assert parsed.thoughts == ""

    def test_task_done_true_with_none_command_field(self):
        """Test validation when task_done is True but command field None"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": True,
            "command": None,
            "thoughts": "Task completed"
        })

        # None is not a valid string, should fail for backward compatibility
        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(response)

    def test_task_done_true_with_not_none_command_field(self):
        """Test validation when task_done is True but command field is not empty"""
        from microbots.llm.llm import LLMAskResponseModel
        from pydantic import ValidationError

        response = json.dumps({
            "task_done": True,
            "command": "not empty",
            "thoughts": "Task completed"
        })

        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel.model_validate_json(response)

        assert "empty" in str(exc_info.value).lower()


@pytest.mark.unit
class TestSummarizeContext:
    """Tests for LLMInterface.summarize_context method"""

    @pytest.fixture
    def llm(self):
        """Create a concrete LLM instance for testing"""
        return ConcreteLLM(max_retries=3)

    def test_basic_summarize_with_enough_messages(self, llm):
        """Test summarization when there are more messages than last_n_messages*2"""
        # Setup: system prompt + 25 messages (more than default 10*2=20)
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        for i in range(25):
            role = "user" if i % 2 == 0 else "assistant"
            llm.messages.append({"role": role, "content": f"Message {i}"})

        # Add a final message that will be popped (the summarization request)
        llm.messages.append({"role": "user", "content": "Please summarize"})

        last_msg = llm.summarize_context(last_n_messages=10, summary="This is a summary")

        # Should have: 1 system prompt + 18 recent messages (10*2-1 minus last one)
        assert len(llm.messages) == 19
        assert llm.messages[0]["role"] == "system"
        assert "__summary__" in llm.messages[0]["content"]
        assert "This is a summary" in llm.messages[0]["content"]
        assert "__end_summary__" in llm.messages[0]["content"]
        # Verify the last user message is returned
        assert last_msg["content"] == "Message 24"

    def test_summarize_with_fewer_messages_than_threshold(self, llm):
        """Test summarization when there are fewer messages than last_n_messages*2"""
        # Setup: system prompt + 5 messages (less than 10*2=20)
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm good"},
            {"role": "user", "content": "Please summarize"}  # Will be popped
        ]

        last_msg = llm.summarize_context(last_n_messages=10, summary="Short conversation summary")

        # Should have: 1 system prompt + 3 messages (all except system, popped, and last user msg)
        assert len(llm.messages) == 4
        assert llm.messages[0]["role"] == "system"
        assert "__summary__" in llm.messages[0]["content"]
        assert "Short conversation summary" in llm.messages[0]["content"]
        # Verify the last user message is returned
        assert last_msg["content"] == "I'm good"

    def test_summarize_with_empty_summary(self, llm):
        """Test summarization with empty summary string"""
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        last_msg = llm.summarize_context(last_n_messages=10, summary="")

        assert len(llm.messages) == 2  # system + Hello (recent_messages[:-1])
        assert "__summary__" in llm.messages[0]["content"]
        assert "__end_summary__" in llm.messages[0]["content"]
        # Empty summary should still have the markers
        assert "\n__summary__\n\n__end_summary__" in llm.messages[0]["content"]
        # Verify the last user message is returned
        assert last_msg["content"] == "Hi"

    def test_summarize_updates_existing_summary(self, llm):
        """Test that existing summary is updated when system prompt already has one"""
        llm.messages = [
            {"role": "system", "content": "You are a helpful assistant.\n__summary__\nOld summary content\n__end_summary__"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        last_msg = llm.summarize_context(last_n_messages=10, summary="New summary content")

        assert "__summary__" in llm.messages[0]["content"]
        assert "Old summary content" in llm.messages[0]["content"]
        assert "New summary content" in llm.messages[0]["content"]
        # Check that the old and new summaries are combined
        content = llm.messages[0]["content"]
        summary_section = content.split("__summary__")[1].split("__end_summary__")[0]
        assert "Old summary content" in summary_section
        assert "New summary content" in summary_section
        # Verify the last user message is returned
        assert last_msg["content"] == "Hi"

    def test_summarize_preserves_system_prompt_before_summary(self, llm):
        """Test that original system prompt is preserved before the summary"""
        original_prompt = "You are a coding assistant with expertise in Python."
        llm.messages = [
            {"role": "system", "content": original_prompt},
            {"role": "user", "content": "Help me"},
            {"role": "assistant", "content": "Sure"},
            {"role": "user", "content": "Summarize"}
        ]

        llm.summarize_context(last_n_messages=10, summary="Summary here")

        assert llm.messages[0]["content"].startswith(original_prompt)

    def test_summarize_pops_last_message(self, llm):
        """Test that the last message (summarization request) is popped"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "This should be popped"}
        ]

        last_msg = llm.summarize_context(last_n_messages=10, summary="Summary")

        # The last user message should be gone from messages list
        for msg in llm.messages:
            assert msg["content"] != "This should be popped"
        # But it should be returned as the last message before summarization
        assert last_msg["content"] == "Response"

    def test_summarize_with_custom_last_n_messages(self, llm):
        """Test summarization with custom last_n_messages value"""
        # Setup: system prompt + 15 messages
        llm.messages = [
            {"role": "system", "content": "System prompt"}
        ]
        for i in range(15):
            role = "user" if i % 2 == 0 else "assistant"
            llm.messages.append({"role": role, "content": f"Message {i}"})
        llm.messages.append({"role": "user", "content": "Summarize"})

        last_msg = llm.summarize_context(last_n_messages=5, summary="Custom summary")

        # Should have: 1 system prompt + 8 recent messages (5*2-1 minus last one)
        assert len(llm.messages) == 9
        # Verify the most recent messages are kept (minus last one which is returned)
        assert llm.messages[-1]["content"] == "Message 13"
        # Verify the last message is returned
        assert last_msg["content"] == "Message 14"

    def test_summarize_with_only_system_prompt_and_one_message(self, llm):
        """Test summarization with minimal messages - edge case that raises IndexError"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "To be popped"}
        ]

        # This edge case raises IndexError because after popping, messages[1:] is empty
        # and recent_messages[-1] will fail on an empty list
        with pytest.raises(IndexError):
            llm.summarize_context(last_n_messages=10, summary="Minimal summary")

    def test_summarize_keeps_correct_recent_messages(self, llm):
        """Test that exactly the right recent messages are kept"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Old message 1"},
            {"role": "assistant", "content": "Old response 1"},
            {"role": "user", "content": "Old message 2"},
            {"role": "assistant", "content": "Old response 2"},
            {"role": "user", "content": "Recent message 1"},
            {"role": "assistant", "content": "Recent response 1"},
            {"role": "user", "content": "Summarize request"}  # Will be popped
        ]

        last_msg = llm.summarize_context(last_n_messages=2, summary="Summary")

        # Should keep last 3 messages (2*2-1 = 3, then [:-1] removes last one)
        assert len(llm.messages) == 3  # 1 system + 2 recent (Old response 2, Recent message 1)
        message_contents = [m["content"] for m in llm.messages[1:]]
        assert "Old response 2" in message_contents
        assert "Recent message 1" in message_contents
        assert "Old message 1" not in message_contents
        assert "Old response 1" not in message_contents
        assert "Old message 2" not in message_contents
        # Recent response 1 is now returned as the last message
        assert last_msg["content"] == "Recent response 1"

    def test_summarize_default_empty_summary(self, llm):
        """Test that default summary parameter is empty string"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Summarize"}
        ]

        # Call without summary parameter
        last_msg = llm.summarize_context(last_n_messages=10)

        assert "__summary__" in llm.messages[0]["content"]
        assert "__end_summary__" in llm.messages[0]["content"]
        # Verify return value
        assert last_msg["content"] == "Response"

    def test_summarize_with_multiline_summary(self, llm):
        """Test summarization with multi-line summary content"""
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Summarize"}
        ]

        multiline_summary = """Line 1 of summary
Line 2 of summary
Line 3 with details"""

        last_msg = llm.summarize_context(last_n_messages=10, summary=multiline_summary)

        assert multiline_summary in llm.messages[0]["content"]
        # Verify return value
        assert last_msg["content"] == "Hi"

    def test_summarize_combines_old_and_new_summary_with_newline(self, llm):
        """Test that old and new summaries are combined with newline separator"""
        llm.messages = [
            {"role": "system", "content": "Prompt\n__summary__\nFirst summary\n__end_summary__"},
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "Summarize"}
        ]

        last_msg = llm.summarize_context(last_n_messages=10, summary="Second summary")

        content = llm.messages[0]["content"]
        summary_section = content.split("__summary__")[1].split("__end_summary__")[0]
        # Check newline separator between summaries
        assert "\nFirst summary\n\nSecond summary\n" in content or "First summary\n\nSecond summary" in summary_section
        # Verify return value
        assert last_msg["content"] == "Hello"

    def test_summarize_exact_boundary_messages(self, llm):
        """Test summarization when messages count equals exactly last_n_messages*2"""
        # Setup: system prompt + exactly 6 messages (3*2)
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Msg 3"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        last_msg = llm.summarize_context(last_n_messages=3, summary="Boundary test")

        # After popping, we have 7 messages. last_n_messages*2 - 1 = 5
        # len(messages) > 5, so we take the last 5
        # Then recent_messages[:-1] removes the last one
        assert len(llm.messages) == 5  # 1 system + 4 recent
        # Verify return value (last message in recent_messages)
        assert last_msg["content"] == "Resp 3"

    def test_summarize_one_more_than_boundary(self, llm):
        """Test summarization when messages count is one more than last_n_messages*2"""
        # Setup: system prompt + 7 messages (one more than 3*2=6)
        llm.messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Should be excluded"},
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Resp 1"},
            {"role": "user", "content": "Msg 2"},
            {"role": "assistant", "content": "Resp 2"},
            {"role": "user", "content": "Msg 3"},
            {"role": "assistant", "content": "Resp 3"},
            {"role": "user", "content": "Summarize"}  # Will be popped
        ]

        last_msg = llm.summarize_context(last_n_messages=3, summary="Test")

        # After popping, last_n_messages*2-1 = 5, we take last 5 then [:-1] = 4
        # Should exclude "Should be excluded" and "Msg 1"
        assert len(llm.messages) == 5  # 1 system + 4 recent
        message_contents = [m["content"] for m in llm.messages]
        assert "Should be excluded" not in message_contents
        assert "Msg 1" not in message_contents
        # Verify return value
        assert last_msg["content"] == "Resp 3"
