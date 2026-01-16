"""Tests for LLMAskResponseModel Pydantic validation"""
import pytest
from pydantic import ValidationError
from microbots.llm.llm import LLMAskResponseModel


class TestLLMAskResponseModelValidation:
    """Test Pydantic field validators on LLMAskResponseModel"""

    def test_valid_task_not_done_with_command(self):
        """Valid: task_done=False with non-empty command"""
        model = LLMAskResponseModel(
            task_done=False,
            command="ls -la",
            thoughts="Listing files"
        )
        assert model.task_done is False
        assert model.command == "ls -la"

    def test_valid_task_done_with_empty_command(self):
        """Valid: task_done=True with empty command"""
        model = LLMAskResponseModel(
            task_done=True,
            command="",
            thoughts="Task completed"
        )
        assert model.task_done is True
        assert model.command == ""

    def test_invalid_task_not_done_with_empty_command(self):
        """Invalid: task_done=False with empty command should raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel(
                task_done=False,
                command="",
                thoughts="This should fail"
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('command',)
        assert 'non-empty' in errors[0]['msg'].lower()

    def test_invalid_task_not_done_with_whitespace_command(self):
        """Invalid: task_done=False with whitespace-only command should raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel(
                task_done=False,
                command="   ",
                thoughts="This should fail"
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('command',)
        assert 'non-empty' in errors[0]['msg'].lower()

    def test_invalid_task_done_with_command(self):
        """Invalid: task_done=True with non-empty command should raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel(
                task_done=True,
                command="ls -la",
                thoughts="This should fail"
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]['loc'] == ('command',)
        assert 'empty' in errors[0]['msg'].lower()

    def test_invalid_task_done_with_whitespace_command(self):
        """Valid: task_done=True with whitespace command is allowed (treated as empty)"""
        # Whitespace-only commands are stripped and treated as empty, which is valid for task_done=True
        model = LLMAskResponseModel(
            task_done=True,
            command="  \n ",
            thoughts="Whitespace treated as empty"
        )
        assert model.task_done is True
        assert model.command == "  \n "  # Original value preserved, but validation treats it as empty

    def test_extra_fields_forbidden(self):
        """Invalid: extra fields should be rejected due to extra='forbid'"""
        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel(
                task_done=False,
                command="ls",
                thoughts="Test",
                extra_field="should fail" # pyright: ignore[reportCallIssue] - We intentionally pass an extra field
            )

        errors = exc_info.value.errors()
        assert any('extra_field' in str(error) for error in errors)

    def test_missing_required_fields(self):
        """Invalid: missing required fields should raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel(
                task_done=False,
                # missing command and thoughts
            ) # pyright: ignore[reportCallIssue] - We intentionally omit required fields

        errors = exc_info.value.errors()
        assert len(errors) >= 1
        field_names = [error['loc'][0] for error in errors]
        assert 'command' in field_names

    def test_wrong_field_types(self):
        """Invalid: wrong field types should raise ValidationError"""
        with pytest.raises(ValidationError) as exc_info:
            LLMAskResponseModel(
                task_done="not a boolean", # pyright: ignore[reportArgumentType] - We intentionally pass wrong type
                command="ls",
                thoughts="Test"
            )

        errors = exc_info.value.errors()
        assert any(error['loc'][0] == 'task_done' for error in errors)

    def test_json_validation(self):
        """Test validation from JSON string"""
        # Valid JSON
        valid_json = '{"task_done": false, "command": "ls", "thoughts": "test"}'
        model = LLMAskResponseModel.model_validate_json(valid_json)
        assert model.command == "ls"

        # Invalid JSON - task_done=True with command
        invalid_json = '{"task_done": true, "command": "ls", "thoughts": "test"}'
        with pytest.raises(ValidationError):
            LLMAskResponseModel.model_validate_json(invalid_json)
