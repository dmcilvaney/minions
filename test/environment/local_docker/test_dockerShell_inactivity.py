"""
Unit tests for dockerShell inactivity timeout feature
"""
import pytest
import os
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../src")))


class TestDockerShellInactivity:
    """Test the inactivity timeout feature in dockerShell"""

    @pytest.mark.asyncio
    async def test_check_inactivity_shuts_down_after_timeout(self):
        """Test that check_inactivity exits when timeout is exceeded"""
        # Import inside test to allow environment variable patching
        with patch.dict(os.environ, {"INACTIVITY_TIMEOUT_MINUTES": "0.01"}):  # 0.6 seconds
            # Mock the modules that have external dependencies
            mock_shell = MagicMock()
            sys.modules['ShellCommunicator'] = MagicMock()
            sys.modules['ShellCommunicator'].ShellCommunicator = MagicMock(return_value=mock_shell)

            # Import after mocking
            import importlib
            import src.microbots.environment.local_docker.image_builder.dockerShell as dockerShell
            importlib.reload(dockerShell)

            # Set last activity to past time to simulate inactivity
            dockerShell.last_activity_time = datetime.now() - timedelta(minutes=1)

            # Mock os.kill to verify it's called instead of actually sending signal
            with patch('os.kill') as mock_kill:
                # Run check_inactivity - it will check quickly due to short timeout
                task = asyncio.create_task(dockerShell.check_inactivity())

                # Wait for check to happen (check interval is capped at minimum 1s, so first check at 1s)
                await asyncio.sleep(2.0)
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Verify os.kill was called with SIGINT
                import signal
                mock_kill.assert_called_with(os.getpid(), signal.SIGINT)

    @pytest.mark.asyncio
    async def test_check_inactivity_does_not_shutdown_with_recent_activity(self):
        """Test that check_inactivity does NOT exit when there's recent activity"""
        with patch.dict(os.environ, {"INACTIVITY_TIMEOUT_MINUTES": "0.05"}):  # 3 seconds
            # Mock the modules that have external dependencies
            mock_shell = MagicMock()
            sys.modules['ShellCommunicator'] = MagicMock()
            sys.modules['ShellCommunicator'].ShellCommunicator = MagicMock(return_value=mock_shell)

            # Import after mocking
            import importlib
            import src.microbots.environment.local_docker.image_builder.dockerShell as dockerShell
            importlib.reload(dockerShell)

            # Set last activity to NOW (recent activity)
            dockerShell.last_activity_time = datetime.now()

            # Mock os._exit to verify it's NOT called
            with patch('os._exit') as mock_exit:
                # Run check_inactivity for a short time
                task = asyncio.create_task(dockerShell.check_inactivity())

                # Wait briefly
                await asyncio.sleep(2)
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Verify os._exit was NOT called (recent activity)
                mock_exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_activity_updates_prevent_shutdown(self):
        """Test that updating last_activity_time prevents shutdown"""
        with patch.dict(os.environ, {"INACTIVITY_TIMEOUT_MINUTES": "0.02"}):  # 1.2 seconds
            # Mock the modules that have external dependencies
            mock_shell = MagicMock()
            sys.modules['ShellCommunicator'] = MagicMock()
            sys.modules['ShellCommunicator'].ShellCommunicator = MagicMock(return_value=mock_shell)

            # Import after mocking
            import importlib
            import src.microbots.environment.local_docker.image_builder.dockerShell as dockerShell
            importlib.reload(dockerShell)

            # Start with old activity
            dockerShell.last_activity_time = datetime.now() - timedelta(minutes=1)

            # Mock os._exit
            with patch('os._exit') as mock_exit:
                task = asyncio.create_task(dockerShell.check_inactivity())

                # Immediately update activity to recent
                await asyncio.sleep(0.1)
                dockerShell.last_activity_time = datetime.now()

                # Wait a bit longer
                await asyncio.sleep(2)
                task.cancel()

                try:
                    await task
                except asyncio.CancelledError:
                    pass

                # Should NOT exit because we updated activity
                mock_exit.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_configuration_from_environment(self):
        """Test that INACTIVITY_TIMEOUT_MINUTES is correctly read from environment"""
        test_timeout = "5.5"

        with patch.dict(os.environ, {"INACTIVITY_TIMEOUT_MINUTES": test_timeout}):
            # Mock the modules that have external dependencies
            mock_shell = MagicMock()
            sys.modules['ShellCommunicator'] = MagicMock()
            sys.modules['ShellCommunicator'].ShellCommunicator = MagicMock(return_value=mock_shell)

            # Import after mocking to pick up new env var
            import importlib
            import src.microbots.environment.local_docker.image_builder.dockerShell as dockerShell
            importlib.reload(dockerShell)

            # Verify the timeout was set correctly
            assert dockerShell.INACTIVITY_TIMEOUT_MINUTES == 5.5

    @pytest.mark.asyncio
    async def test_default_timeout_is_60_minutes(self):
        """Test that default timeout is 30 minutes when env var not set"""
        # Ensure env var is not set
        env_clean = os.environ.copy()
        env_clean.pop("INACTIVITY_TIMEOUT_MINUTES", None)

        with patch.dict(os.environ, env_clean, clear=True):
            # Mock the modules that have external dependencies
            mock_shell = MagicMock()
            sys.modules['ShellCommunicator'] = MagicMock()
            sys.modules['ShellCommunicator'].ShellCommunicator = MagicMock(return_value=mock_shell)

            # Import after mocking
            import importlib
            import src.microbots.environment.local_docker.image_builder.dockerShell as dockerShell
            importlib.reload(dockerShell)

            # Verify default is 30 minutes
            assert dockerShell.INACTIVITY_TIMEOUT_MINUTES == 60.0
