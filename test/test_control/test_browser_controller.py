import os
import sys
import asyncio
import importlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# ---------------------------------------------------------------------------
# Mock out browser_use before importing the module under test, so the import
# succeeds even when the real browser_use package is not installed.
# ---------------------------------------------------------------------------
import unittest.mock as _mock

_browser_use_mock = _mock.MagicMock()
sys.modules.setdefault("browser_use", _browser_use_mock)

import browser_orchestrator  # noqa: E402  — module under test
from browser_orchestrator import run_command  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_browser():
    """Return a mock Browser instance."""
    return MagicMock()


@pytest.fixture
def mock_history():
    """Return a mock agent history result."""
    history = MagicMock()
    history.final_result.return_value = "task completed"
    return history


# ---------------------------------------------------------------------------
# Tests for run_command
# ---------------------------------------------------------------------------

class TestRunCommand:
    """Tests for the async run_command function."""

    @pytest.mark.asyncio
    async def test_run_command_returns_history(self, mock_browser, mock_history):
        """run_command should return the history object produced by agent.run()."""
        with patch("browser_orchestrator.ChatBrowserUse") as MockLLM, \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            result = await run_command("do something", mock_browser)

            assert result is mock_history

    @pytest.mark.asyncio
    async def test_run_command_creates_agent_with_correct_args(self, mock_browser, mock_history):
        """run_command should instantiate Agent with task, llm, and browser."""
        with patch("browser_orchestrator.ChatBrowserUse") as MockLLM, \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_llm_instance = MockLLM.return_value
            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            await run_command("open google", mock_browser)

            MockAgent.assert_called_once_with(
                task="open google",
                llm=mock_llm_instance,
                browser=mock_browser,
            )

    @pytest.mark.asyncio
    async def test_run_command_calls_agent_run(self, mock_browser, mock_history):
        """run_command should call agent.run() exactly once."""
        with patch("browser_orchestrator.ChatBrowserUse"), \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            await run_command("search something", mock_browser)

            mock_agent_instance.run.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_run_command_instantiates_chat_browser_use(self, mock_browser, mock_history):
        """run_command should create a ChatBrowserUse LLM instance."""
        with patch("browser_orchestrator.ChatBrowserUse") as MockLLM, \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            await run_command("click button", mock_browser)

            MockLLM.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_command_passes_browser_to_agent(self, mock_browser, mock_history):
        """run_command should pass the provided Browser object to Agent."""
        with patch("browser_orchestrator.ChatBrowserUse"), \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            await run_command("navigate to page", mock_browser)

            _, kwargs = MockAgent.call_args
            assert kwargs["browser"] is mock_browser

    @pytest.mark.asyncio
    async def test_run_command_passes_task_string_to_agent(self, mock_browser, mock_history):
        """run_command should pass the exact command string as task to Agent."""
        with patch("browser_orchestrator.ChatBrowserUse"), \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            command = "fill in the login form"
            await run_command(command, mock_browser)

            _, kwargs = MockAgent.call_args
            assert kwargs["task"] == command

    @pytest.mark.asyncio
    async def test_run_command_propagates_agent_exception(self, mock_browser):
        """run_command should propagate exceptions raised by agent.run()."""
        with patch("browser_orchestrator.ChatBrowserUse"), \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.side_effect = RuntimeError("agent failed")
            MockAgent.return_value = mock_agent_instance

            with pytest.raises(RuntimeError, match="agent failed"):
                await run_command("some task", mock_browser)

    @pytest.mark.asyncio
    async def test_run_command_with_empty_string_command(self, mock_browser, mock_history):
        """run_command should work with an empty command string."""
        with patch("browser_orchestrator.ChatBrowserUse"), \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            result = await run_command("", mock_browser)

            assert result is mock_history

    @pytest.mark.asyncio
    async def test_run_command_multiple_calls_create_separate_agents(self, mock_browser, mock_history):
        """Each call to run_command should create a new Agent instance."""
        with patch("browser_orchestrator.ChatBrowserUse"), \
             patch("browser_orchestrator.Agent") as MockAgent:

            mock_agent_instance = AsyncMock()
            mock_agent_instance.run.return_value = mock_history
            MockAgent.return_value = mock_agent_instance

            await run_command("task one", mock_browser)
            await run_command("task two", mock_browser)

            assert MockAgent.call_count == 2


# ---------------------------------------------------------------------------
# Tests for environment / module-level setup
# ---------------------------------------------------------------------------

class TestEnvironmentSetup:

    def test_env_path_resolves_to_dotenv_file(self):
        """ENV_PATH should point to a .env file two levels above the module."""
        assert browser_orchestrator.ENV_PATH.name == ".env"

    def test_browser_use_api_key_read_from_env(self):
        """BROWSER_USE_API_KEY should be loaded from the environment."""
        with patch.dict(os.environ, {"BROWSER_USE_API_KEY": "test-key-123"}):
            importlib.reload(browser_orchestrator)
            assert browser_orchestrator.BROWSER_USE_API_KEY == "test-key-123"
            
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only policy")
    def test_windows_event_loop_policy_set(self):
        """On Windows, the event loop policy should be WindowsProactorEventLoopPolicy."""
        policy = asyncio.get_event_loop_policy()
        assert isinstance(policy, asyncio.WindowsProactorEventLoopPolicy)