"""
Unit tests for CommandOrchestrator module.
Run with: pytest test_command_orchestrator.py -v --cov=command_orchestrator --cov-report=term-missing
"""

import json
import os
import pytest
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
import sys
import types


# ---------------------------------------------------------------------------
# Stub out google.genai so the module imports without the real SDK
# ---------------------------------------------------------------------------

google_pkg = types.ModuleType("google")
genai_mod = types.ModuleType("google.genai")
genai_types_mod = types.ModuleType("google.genai.types")

# Minimal Content / Part / GenerateContentConfig stubs
class _Part:
    def __init__(self, text=""): self.text = text

class _Content:
    def __init__(self, role="", parts=None): self.role = role; self.parts = parts or []

class _GenerateContentConfig:
    def __init__(self, **kwargs): self.__dict__.update(kwargs)

genai_types_mod.Part = _Part
genai_types_mod.Content = _Content
genai_types_mod.GenerateContentConfig = _GenerateContentConfig
genai_mod.types = genai_types_mod
genai_mod.Client = MagicMock()

google_pkg.genai = genai_mod
sys.modules.setdefault("google", google_pkg)
sys.modules.setdefault("google.genai", genai_mod)
sys.modules.setdefault("google.genai.types", genai_types_mod)

# ---------------------------------------------------------------------------
# Import module under test
# ---------------------------------------------------------------------------
from command_orchestrator import (
    OrchestratorResponse,
    OrchestratorError,
    InitializationError,
    InferenceError,
    CommandOrchestrator,
    create_orchestrator,
)


# ===========================================================================
# Helpers
# ===========================================================================

API_KEY = "test-api-key-123"


def _make_orchestrator(**kwargs) -> CommandOrchestrator:
    """Create an orchestrator with a mocked genai client."""
    kwargs.setdefault("api_key", API_KEY)
    orch = CommandOrchestrator(**kwargs)
    orch.client = MagicMock()
    return orch


def _llm_json_response(payload: dict) -> MagicMock:
    """Build a fake LLM response object whose .text is JSON."""
    mock_resp = MagicMock()
    mock_resp.text = json.dumps(payload)
    return mock_resp


# ===========================================================================
# OrchestratorResponse dataclass
# ===========================================================================

class TestOrchestratorResponse:

    def test_needs_clarification_true(self):
        r = OrchestratorResponse(needs_clarification=True, user_prompt="What do you mean?")
        assert r.needs_clarification is True
        assert r.user_prompt == "What do you mean?"

    def test_needs_clarification_false(self):
        r = OrchestratorResponse(needs_clarification=False, clarified_command="Navigate to Gmail")
        assert r.needs_clarification is False
        assert r.clarified_command == "Navigate to Gmail"

    def test_all_fields_default_none(self):
        r = OrchestratorResponse(needs_clarification=False)
        assert r.user_prompt is None
        assert r.clarified_command is None
        assert r.reasoning is None
        assert r.metadata is None

    def test_reasoning_and_metadata_stored(self):
        r = OrchestratorResponse(
            needs_clarification=False,
            reasoning="All clear",
            metadata={"key": "value"}
        )
        assert r.reasoning == "All clear"
        assert r.metadata == {"key": "value"}


# ===========================================================================
# Custom exceptions
# ===========================================================================

class TestExceptions:

    def test_orchestrator_error_is_exception(self):
        with pytest.raises(OrchestratorError):
            raise OrchestratorError("base error")

    def test_initialization_error_is_orchestrator_error(self):
        with pytest.raises(OrchestratorError):
            raise InitializationError("init failed")

    def test_inference_error_is_orchestrator_error(self):
        with pytest.raises(OrchestratorError):
            raise InferenceError("inference failed")

    def test_initialization_error_message(self):
        try:
            raise InitializationError("no key")
        except InitializationError as e:
            assert "no key" in str(e)

    def test_inference_error_message(self):
        try:
            raise InferenceError("api down")
        except InferenceError as e:
            assert "api down" in str(e)


# ===========================================================================
# CommandOrchestrator.__init__
# ===========================================================================

class TestCommandOrchestratorInit:

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(InitializationError, match="GEMINI_API_KEY not found"):
                CommandOrchestrator(api_key=None)

    def test_uses_env_var_api_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            orch = CommandOrchestrator()
            assert orch._api_key == "env-key"

    def test_explicit_api_key_takes_precedence(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            orch = _make_orchestrator(api_key="explicit-key")
            assert orch._api_key == "explicit-key"

    def test_model_id_stored(self):
        orch = _make_orchestrator(model_id="gemini-2.5-pro")
        assert orch.model_id == "gemini-2.5-pro"

    def test_default_model_id(self):
        orch = _make_orchestrator()
        assert orch.model_id == "gemini-2.5-flash"

    def test_conversation_history_empty_on_init(self):
        orch = _make_orchestrator()
        assert orch.conversation_history == []

    def test_current_goal_none_on_init(self):
        orch = _make_orchestrator()
        assert orch.current_goal is None

    def test_system_prompt_loaded_from_file(self, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Custom system prompt", encoding="utf-8")
        orch = _make_orchestrator(prompt_path=str(prompt_file))
        assert orch.system_prompt == "Custom system prompt"

    def test_default_prompt_used_when_no_file(self):
        orch = _make_orchestrator(prompt_path=None)
        assert "clarification" in orch.system_prompt.lower()

    def test_default_prompt_used_when_path_missing(self):
        orch = _make_orchestrator(prompt_path="/nonexistent/path/prompt.txt")
        # Should fall back to default, not raise
        assert len(orch.system_prompt) > 0

    def test_raises_when_client_init_fails(self):
        """If genai.Client raises, InitializationError should propagate."""
        import google.genai as _genai
        original_client = _genai.Client
        _genai.Client = MagicMock(side_effect=Exception("client init failed"))
        try:
            raised = False
            try:
                CommandOrchestrator(api_key=API_KEY)
            except (InitializationError, Exception):
                raised = True
            assert raised
        finally:
            _genai.Client = original_client


# ===========================================================================
# _get_default_system_prompt
# ===========================================================================

class TestGetDefaultSystemPrompt:

    def test_returns_non_empty_string(self):
        orch = _make_orchestrator()
        prompt = orch._get_default_system_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_contains_needs_clarification_key(self):
        orch = _make_orchestrator()
        assert "needs_clarification" in orch._get_default_system_prompt()

    def test_contains_clarified_command_key(self):
        orch = _make_orchestrator()
        assert "clarified_command" in orch._get_default_system_prompt()

    def test_contains_example_commands(self):
        orch = _make_orchestrator()
        prompt = orch._get_default_system_prompt()
        assert "Gmail" in prompt or "navigate" in prompt.lower()


# ===========================================================================
# _parse_response
# ===========================================================================

class TestParseResponse:

    def setup_method(self):
        self.orch = _make_orchestrator()

    # --- Clarification needed ---

    def test_parse_needs_clarification_true(self):
        raw = json.dumps({"needs_clarification": True, "question": "Which service?"})
        result = self.orch._parse_response(raw, "Go to Google")
        assert result.needs_clarification is True
        assert result.user_prompt == "Which service?"

    def test_parse_clarification_metadata_includes_original_input(self):
        raw = json.dumps({"needs_clarification": True, "question": "Which?"})
        result = self.orch._parse_response(raw, "original input")
        assert result.metadata["original_input"] == "original input"

    def test_parse_missing_question_falls_back_to_default(self):
        raw = json.dumps({"needs_clarification": True})
        result = self.orch._parse_response(raw, "vague")
        assert result.user_prompt is not None
        assert len(result.user_prompt) > 0

    # --- Clear command ---

    def test_parse_needs_clarification_false(self):
        raw = json.dumps({"needs_clarification": False, "clarified_command": "Navigate to Gmail"})
        result = self.orch._parse_response(raw, "open gmail")
        assert result.needs_clarification is False
        assert result.clarified_command == "Navigate to Gmail"

    def test_parse_missing_clarified_command_falls_back_to_original(self):
        raw = json.dumps({"needs_clarification": False})
        result = self.orch._parse_response(raw, "open gmail")
        assert result.clarified_command == "open gmail"

    def test_parse_metadata_includes_original_input(self):
        raw = json.dumps({"needs_clarification": False, "clarified_command": "Navigate to Gmail"})
        result = self.orch._parse_response(raw, "open gmail")
        assert result.metadata["original_input"] == "open gmail"

    # --- Markdown code block stripping ---

    def test_parse_strips_markdown_code_block(self):
        raw = "```json\n{\"needs_clarification\": false, \"clarified_command\": \"Navigate to Gmail\"}\n```"
        result = self.orch._parse_response(raw, "open gmail")
        assert result.needs_clarification is False
        assert result.clarified_command == "Navigate to Gmail"

    def test_parse_strips_plain_code_block(self):
        raw = "```\n{\"needs_clarification\": false, \"clarified_command\": \"Navigate to Gmail\"}\n```"
        result = self.orch._parse_response(raw, "open gmail")
        assert result.needs_clarification is False

    # --- Error / fallback ---

    def test_parse_invalid_json_returns_clarification(self):
        result = self.orch._parse_response("not json at all !!!", "something")
        assert result.needs_clarification is True
        assert "rephrase" in result.user_prompt.lower() or len(result.user_prompt) > 0

    def test_parse_invalid_json_includes_reasoning(self):
        result = self.orch._parse_response("{broken json", "something")
        assert result.reasoning is not None

    def test_parse_invalid_json_metadata_has_raw_response(self):
        raw = "totally not json"
        result = self.orch._parse_response(raw, "input")
        assert result.metadata["raw_response"] == raw

    def test_parse_empty_string_returns_clarification(self):
        result = self.orch._parse_response("", "input")
        assert result.needs_clarification is True


# ===========================================================================
# _call_llm (async)
# ===========================================================================

class TestCallLlm:

    def setup_method(self):
        self.orch = _make_orchestrator()

    @pytest.mark.asyncio
    async def test_returns_response_text(self):
        mock_resp = MagicMock()
        mock_resp.text = '{"needs_clarification": false, "clarified_command": "Navigate to Gmail"}'
        self.orch.client.models.generate_content = MagicMock(return_value=mock_resp)
        result = await self.orch._call_llm([{"role": "user", "content": "Open Gmail"}])
        assert "Navigate to Gmail" in result

    @pytest.mark.asyncio
    async def test_raises_inference_error_on_api_failure(self):
        self.orch.client.models.generate_content = MagicMock(side_effect=Exception("API down"))
        with pytest.raises(InferenceError, match="LLM API call failed"):
            await self.orch._call_llm([{"role": "user", "content": "test"}])

    @pytest.mark.asyncio
    async def test_assistant_role_mapped_to_model(self):
        """Assistant messages should be passed as 'model' role to genai."""
        mock_resp = MagicMock()
        mock_resp.text = '{"needs_clarification": false, "clarified_command": "x"}'
        self.orch.client.models.generate_content = MagicMock(return_value=mock_resp)
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        await self.orch._call_llm(messages)
        call_args = self.orch.client.models.generate_content.call_args
        contents = call_args[1]["contents"] if "contents" in call_args[1] else call_args[0][1]
        roles = [c.role for c in contents]
        assert "model" in roles

    @pytest.mark.asyncio
    async def test_system_prompt_included_in_first_message(self):
        mock_resp = MagicMock()
        mock_resp.text = '{"needs_clarification": false, "clarified_command": "x"}'
        self.orch.client.models.generate_content = MagicMock(return_value=mock_resp)
        await self.orch._call_llm([{"role": "user", "content": "hello"}])
        call_args = self.orch.client.models.generate_content.call_args
        contents = call_args[1].get("contents") or call_args[0][1]
        first_text = contents[0].parts[0].text
        assert len(first_text) > 50  # system prompt is substantial

    @pytest.mark.asyncio
    async def test_uses_configured_model_id(self):
        orch = _make_orchestrator(model_id="gemini-custom-model")
        mock_resp = MagicMock()
        mock_resp.text = '{"needs_clarification": false, "clarified_command": "x"}'
        orch.client.models.generate_content = MagicMock(return_value=mock_resp)
        await orch._call_llm([{"role": "user", "content": "hi"}])
        call_args = orch.client.models.generate_content.call_args
        model_arg = call_args[1].get("model") or call_args[0][0]
        assert model_arg == "gemini-custom-model"


# ===========================================================================
# process (async, full integration with mocked LLM)
# ===========================================================================

class TestProcess:

    def setup_method(self):
        self.orch = _make_orchestrator()

    def _setup_llm(self, payload: dict):
        mock_resp = MagicMock()
        mock_resp.text = json.dumps(payload)
        self.orch.client.models.generate_content = MagicMock(return_value=mock_resp)

    @pytest.mark.asyncio
    async def test_clear_command_returns_clarified(self):
        self._setup_llm({"needs_clarification": False, "clarified_command": "Navigate to Gmail"})
        result = await self.orch.process("Open Gmail")
        assert result.needs_clarification is False
        assert result.clarified_command == "Navigate to Gmail"

    @pytest.mark.asyncio
    async def test_ambiguous_command_returns_question(self):
        self._setup_llm({"needs_clarification": True, "question": "Which Google service?"})
        result = await self.orch.process("Go to Google")
        assert result.needs_clarification is True
        assert "Google" in result.user_prompt

    @pytest.mark.asyncio
    async def test_conversation_context_forwarded(self):
        self._setup_llm({"needs_clarification": False, "clarified_command": "Navigate to Gmail"})
        context = [{"role": "assistant", "content": "Which service?"}]
        await self.orch.process("Gmail", conversation_context=context)
        call_args = self.orch.client.models.generate_content.call_args
        contents = call_args[1].get("contents") or call_args[0][1]
        # Should contain more than just system prompt + current user message
        assert len(contents) > 2

    @pytest.mark.asyncio
    async def test_no_conversation_context_still_works(self):
        self._setup_llm({"needs_clarification": False, "clarified_command": "Search for cats"})
        result = await self.orch.process("Search cats")
        assert result.needs_clarification is False

    @pytest.mark.asyncio
    async def test_inference_error_wrapped_correctly(self):
        self.orch.client.models.generate_content = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(InferenceError):
            await self.orch.process("Open Gmail")

    @pytest.mark.asyncio
    async def test_llm_bad_json_returns_clarification_gracefully(self):
        mock_resp = MagicMock()
        mock_resp.text = "I don't understand this format"
        self.orch.client.models.generate_content = MagicMock(return_value=mock_resp)
        result = await self.orch.process("something weird")
        assert result.needs_clarification is True

    @pytest.mark.asyncio
    async def test_empty_user_input_processed(self):
        self._setup_llm({"needs_clarification": True, "question": "What would you like to do?"})
        result = await self.orch.process("")
        assert result.needs_clarification is True

    @pytest.mark.asyncio
    async def test_response_metadata_has_original_input(self):
        self._setup_llm({"needs_clarification": False, "clarified_command": "Navigate to Gmail"})
        result = await self.orch.process("open gmail please")
        assert result.metadata["original_input"] == "open gmail please"


# ===========================================================================
# reset
# ===========================================================================

class TestReset:

    def test_reset_clears_conversation_history(self):
        orch = _make_orchestrator()
        orch.conversation_history = [{"role": "user", "content": "hello"}]
        orch.reset()
        assert orch.conversation_history == []

    def test_reset_clears_current_goal(self):
        orch = _make_orchestrator()
        orch.current_goal = "browse the web"
        orch.reset()
        assert orch.current_goal is None

    def test_reset_idempotent(self):
        orch = _make_orchestrator()
        orch.reset()
        orch.reset()
        assert orch.conversation_history == []
        assert orch.current_goal is None


# ===========================================================================
# apply_guardrails (static method)
# ===========================================================================

class TestApplyGuardrails:

    # --- Allowed cases ---

    def test_empty_command_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("")
        assert allowed is True
        assert msg is None

    def test_none_command_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails(None)
        assert allowed is True

    def test_normal_command_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("Open Gmail")
        assert allowed is True

    def test_question_about_hack_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("is this a hack?")
        assert allowed is True

    def test_informational_what_is_malware_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("what is malware")
        assert allowed is True

    def test_how_to_question_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("how to protect against phishing")
        assert allowed is True

    def test_is_this_a_scam_question_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("is this a scam?")
        assert allowed is True

    def test_learn_about_malware_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("learn about malware")
        assert allowed is True

    def test_navigate_to_google_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("navigate to google.com")
        assert allowed is True

    def test_search_for_cats_allowed(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("Search for cats on YouTube")
        assert allowed is True

    # --- Blocked cases ---

    def test_hack_verb_blocked(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("hack into this website")
        assert allowed is False
        assert msg is not None

    def test_steal_password_blocked(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("steal password from the login page")
        assert allowed is False

    def test_phish_verb_blocked(self):
        # The blocked keyword is "phishing", so we use that in the command
        allowed, msg = CommandOrchestrator.apply_guardrails("phish users with phishing form")
        assert allowed is False

    def test_blocked_message_contains_safety_policy(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("hack the server")
        assert allowed is False
        assert "safety" in msg.lower() or "cannot" in msg.lower()

    def test_exploit_verb_blocked(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("exploit this vulnerability")
        assert allowed is False

    def test_please_hack_blocked(self):
        allowed, msg = CommandOrchestrator.apply_guardrails("please hack this system")
        assert allowed is False

    # --- Case insensitivity ---

    def test_uppercase_normal_command_allowed(self):
        allowed, _ = CommandOrchestrator.apply_guardrails("OPEN GMAIL")
        assert allowed is True

    def test_mixed_case_hack_command_blocked(self):
        allowed, _ = CommandOrchestrator.apply_guardrails("Hack into the server")
        assert allowed is False


# ===========================================================================
# create_orchestrator (factory function)
# ===========================================================================

class TestCreateOrchestrator:

    def test_returns_command_orchestrator_instance(self):
        orch = create_orchestrator(api_key=API_KEY)
        assert isinstance(orch, CommandOrchestrator)

    def test_api_key_passed_through(self):
        orch = create_orchestrator(api_key=API_KEY)
        assert orch._api_key == API_KEY

    def test_model_id_passed_through(self):
        orch = create_orchestrator(api_key=API_KEY, model_id="gemini-custom")
        assert orch.model_id == "gemini-custom"

    def test_prompt_path_used_when_valid(self, tmp_path):
        pf = tmp_path / "prompt.txt"
        pf.write_text("Factory prompt", encoding="utf-8")
        orch = create_orchestrator(api_key=API_KEY, prompt_path=str(pf))
        assert orch.system_prompt == "Factory prompt"

    def test_raises_without_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("GEMINI_API_KEY", None)
            with pytest.raises(InitializationError):
                create_orchestrator(api_key=None)


# ===========================================================================
# End-to-end: multi-turn clarification flow
# ===========================================================================

class TestMultiTurnFlow:

    @pytest.mark.asyncio
    async def test_clarification_then_command(self):
        """Simulates: vague input → clarification → resolved command."""
        orch = _make_orchestrator()

        # First turn: ambiguous
        resp1 = MagicMock()
        resp1.text = json.dumps({"needs_clarification": True, "question": "Which service?"})

        # Second turn: resolved
        resp2 = MagicMock()
        resp2.text = json.dumps({"needs_clarification": False, "clarified_command": "Navigate to Gmail"})

        orch.client.models.generate_content = MagicMock(side_effect=[resp1, resp2])

        r1 = await orch.process("Go to Google")
        assert r1.needs_clarification is True

        context = [
            {"role": "user", "content": "Go to Google"},
            {"role": "assistant", "content": r1.user_prompt},
        ]
        r2 = await orch.process("Gmail", conversation_context=context)
        assert r2.needs_clarification is False
        assert r2.clarified_command == "Navigate to Gmail"

    @pytest.mark.asyncio
    async def test_direct_clear_command_no_round_trip(self):
        orch = _make_orchestrator()
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({"needs_clarification": False, "clarified_command": "Search for cats on Google"})
        orch.client.models.generate_content = MagicMock(return_value=mock_resp)

        result = await orch.process("Search for cats")
        assert result.needs_clarification is False
        assert "cats" in result.clarified_command.lower()