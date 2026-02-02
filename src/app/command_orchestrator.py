"""
CommandOrchestrator Module (M10)

Uses external LLM API for natural language reasoning to generate browser commands.
Integrates with User Profile Manager (M8), Error Feedback Module (M6), and Security Layer (M9).
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from google import genai
from google.genai import types


# ============================================================================
# Exported Constants
# ============================================================================

MAX_CONTEXT_TOKENS: int = 4000

COMMAND_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "required": ["action", "target"],
    "properties": {
        "action": {
            "type": "string",
            "enum": ["click", "type", "navigate", "scroll", "wait", "extract"]
        },
        "target": {
            "type": "string",
            "description": "CSS selector or URL"
        },
        "value": {
            "type": "string",
            "description": "Optional value for type or navigate actions"
        },
        "confirmation_required": {
            "type": "boolean",
            "description": "Whether this action requires user confirmation"
        }
    }
}


# ============================================================================
# Supporting Data Structures
# ============================================================================

class ActionType(Enum):
    """Enumeration of supported browser actions."""
    CLICK = "click"
    TYPE = "type"
    NAVIGATE = "navigate"
    SCROLL = "scroll"
    WAIT = "wait"
    EXTRACT = "extract"


class Mode(Enum):
    BASELINE = "BASELINE"
    BLOCKED = "BLOCKED"
    RECOVERING = "RECOVERING"


@dataclass
class Command:
    """Structured command object for browser control."""
    action: str
    target: str
    value: Optional[str] = None
    confirmation_required: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert command to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Message:
    """Represents a single message in conversation history."""
    role: str  # 'user' or 'assistant'
    content: str
    tokens: int = 0
    
    def to_dict(self) -> Dict[str, str]:
        """Convert message to dictionary for API payload."""
        return {"role": self.role, "content": self.content}


@dataclass
class Config:
    """API configuration parameters."""
    model_id: str
    endpoint_url: str
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: int = 30


# ============================================================================
# Custom Exceptions
# ============================================================================

class InitializationError(Exception):
    """Raised when orchestrator cannot be initialized properly."""
    pass


class InferenceError(Exception):
    """Raised when LLM API fails or returns unparsable response."""
    pass


# ============================================================================
# 10.2.2 CommandOrchestrator Class
# ============================================================================

class CommandOrchestrator:

    """
    Orchestrates natural language to browser command translation using a stateful 
    three-prompt architecture (Primary, Error, Recovery) using LLM API.
    Maintains conversation context, validates outputs, and applies safety guardrails.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gemini-2.0-flash",
        prompt_path: str = "prompts/browser_assistant.md",
        user_profile_manager=None,
        error_feedback_module=None,
        security_layer=None
    ):
        """
        Initializes the CommandOrchestrator using the new Google Gen AI SDK.
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise InitializationError("GEMINI_API_KEY not found in parameters or environment.")
        
        # Initialize the Gen AI Client
        self.client = genai.Client(api_key=self._api_key)
        self.model_id = model_id
        self.prompt_path = prompt_path
        # Load the persistent primary prompt and specific role overlays from prompts directory
        prompt_dir = Path(__file__).parent / "prompts"
        # primary_prompt.md is the base system role required on every call
        self.primary_prompt = self._load_system_prompt(str(prompt_dir / "primary_prompt.md"))
        # error and recovery overlays (UI-facing / recovery role)
        self.error_evaluation_prompt = self._load_system_prompt(str(prompt_dir / "error_evaluation_prompt.md"))
        self.recovery_prompt = self._load_system_prompt(str(prompt_dir / "recovery_prompt.md"))

        # system_prompt remains the base persistent prompt (primary)
        self.system_prompt = self.primary_prompt

        # Conversation history uses local Message dataclass for compact bookkeeping
        self.conversation_history: List[Message] = []
        
        # Integration placeholders
        self.user_profile_manager = user_profile_manager
        self.error_feedback_module = error_feedback_module
        self.security_layer = security_layer

        # Minimal orchestrator state per design
        # `primary_goal` is captured from the first user transcript and remains immutable
        self.primary_goal: Optional[str] = None
        self.mode: Mode = Mode.BASELINE
        # Active browser error metadata when a substep fails
        self.active_browser_error: Optional[Dict[str, Any]] = None

        # Track if we are awaiting user confirmation for the primary goal
        self.awaiting_primary_goal_confirmation: bool = False
        # Store the last proposed command and user text for confirmation
        self._pending_primary_command: Optional[Command] = None
        self._pending_primary_user_text: Optional[str] = None

        # API config used by the low-level caller (defaults can be overridden)
        self.api_config = Config(
            model_id=self.model_id,
            endpoint_url=os.environ.get("LLM_ENDPOINT", "https://api.example.com/v1/generate"),
            temperature=0.2,
            max_tokens=512,
            timeout=30,
        )

        # Small role overlays (kept minimal/token-efficient)
        self._role_overlays = {
            "BASELINE": "Convert the user's transcript into ONE high-level browser command. Do NOT reason about errors or recovery. Output must conform to the JSON command schema.",
            "ERROR": self.error_evaluation_prompt,
            "RECOVERY": self.recovery_prompt,
        }

        # Attempt to load overlay prompt files from the same directory as prompt_path.
        # Do this after establishing default overlays so file contents override defaults.
        try:
            prompt_dir = Path(self.prompt_path).parent
            self._load_overlay_prompts_from_dir(str(prompt_dir))
        except Exception:
            pass

        # If no explicit baseline overlay file exists, treat the persistent system prompt as the baseline overlay.
        if not self._role_overlays.get("BASELINE"):
            self._role_overlays["BASELINE"] = self.system_prompt

    def _load_system_prompt(self, path: str) -> str:
        """Loads prompt from a file. Falls back to a basic string if file is missing."""
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Warning: Prompt file {path} not found. Using default.")
            return "You are a helpful browser assistant."

    def _construct_system_instruction(self, role: str = "BASELINE") -> List[Dict[str, str]]:
        """
        Aggregate the prompt stack for every LLM call as a list of system messages.

        Stack order (injected every call):
          1) primary_prompt.md (persistent base system role)
          2) compact shared context (primary goal, execution contract, optional error metadata)
          3) role-specific overlay (ERROR or RECOVERY) if applicable

        Returns a list of dicts suitable for the `messages` payload.
        """
        base = self.primary_prompt or self.system_prompt
        shared_ctx = self._build_shared_context()

        # Choose overlay content for role if available
        overlay = self._role_overlays.get(role, "")

        system_blocks = [
            {"role": "system", "content": base},
            {"role": "system", "content": shared_ctx},
        ]
        if overlay:
            system_blocks.append({"role": "system", "content": overlay})

        return system_blocks
    
    def _load_overlay_prompts_from_dir(self, prompt_dir: str) -> None:
        """Load role overlay markdown files from a directory into self._role_overlays.

        Recognized filenames: baseline.md, error.md, recovery.md (case-insensitive).
        Any other .md files will be loaded using their stem name as the role key (uppercased).
        Existing in-memory overlays are preserved unless a file provides an override.
        """
        try:
            dir_path = Path(prompt_dir)
            if not dir_path.exists() or not dir_path.is_dir():
                return

            loaded: Dict[str, str] = {}
            # flexible matching: accept files whose stem contains the role keyword
            for f in dir_path.glob("*.md"):
                try:
                    stem = f.stem.lower()
                    text = f.read_text(encoding="utf-8")
                except Exception:
                    continue

                if "baseline" in stem:
                    loaded["BASELINE"] = text
                elif "error" in stem:
                    loaded["ERROR"] = text
                elif "recovery" in stem:
                    loaded["RECOVERY"] = text
                else:
                    # fallback: map stem -> uppercased role name
                    loaded[stem.upper()] = text

            if loaded:
                # prefer file contents over existing in-memory overlays
                self._role_overlays.update(loaded)

        except Exception:
            # best-effort loader: do not raise on failures
            return

    def reload_prompt_files(self, prompt_path: Optional[str] = None) -> None:
        """Reload the system prompt and overlay prompt files."""
        path = prompt_path or self.prompt_path
        # reload base system prompt
        try:
            self.system_prompt = self._load_system_prompt(path)
        except Exception:
            pass

        # attempt to load overlay markdowns from same directory as the system prompt
        try:
            prompt_dir = Path(path).parent
            self._load_overlay_prompts_from_dir(str(prompt_dir))
        except Exception:
            pass


    def processTranscript(self, text: str):
        """
        Converts natural language to a plain-text browser command or sequence of commands.
        Handles confirmation/decline of the primary goal before setting it.
        Returns:
            - If awaiting confirmation: None (wait for accept/decline)
            - If error: Command or clarifying question
            - If baseline: Command (if confirmed), or Command (pending confirmation)
        """
        # If an error is active, route to recovery
        if self.active_browser_error:
            return self.processRecoveryInput(text)

        # If awaiting confirmation, ignore new user input until accept/decline is called
        if self.awaiting_primary_goal_confirmation:
            return None

        # Always track user input
        self._update_history(Message(role="user", content=text, tokens=max(1, len(text.split()))))

        # If no primary goal, propose a command and require confirmation
        if self.primary_goal is None:
            payload = self._construct_payload(text=text, role="BASELINE")
            try:
                raw = self._call_llm_api(payload)
                guarded = self._apply_guardrails(raw)
                self._update_history(Message(role="assistant", content=guarded, tokens=max(1, len(guarded.split()))))
                self.awaiting_primary_goal_confirmation = True
                self._pending_primary_command = guarded
                self._pending_primary_user_text = text
                return guarded
            except Exception as e:
                if self.error_feedback_module:
                    self.error_feedback_module.report_error("CommandOrchestrator", str(e))
                raise InferenceError(f"GenAI inference failed: {str(e)}")

        # If primary goal is set, proceed as normal (BASELINE)
        payload = self._construct_payload(text=text, role="BASELINE")
        try:
            raw = self._call_llm_api(payload)
            guarded = self._apply_guardrails(raw)
            self._update_history(Message(role="assistant", content=guarded, tokens=max(1, len(guarded.split()))))
            return guarded
        except Exception as e:
            if self.error_feedback_module:
                self.error_feedback_module.report_error("CommandOrchestrator", str(e))
            raise InferenceError(f"GenAI inference failed: {str(e)}")
    
    def validateSchema(self, output: str) -> bool:
        """
        Validate LLM output against COMMAND_SCHEMA.
        
        Args:
            output: Raw JSON string from LLM
            
        Returns:
            True if output contains all required keys, False otherwise
        """
        try:
            data = json.loads(output)
            
            # Check required fields
            required_fields = COMMAND_SCHEMA["required"]
            for field in required_fields:
                if field not in data:
                    return False
            
            # Validate action type
            valid_actions = COMMAND_SCHEMA["properties"]["action"]["enum"]
            if data.get("action") not in valid_actions:
                return False
            
            # Validate target is a string
            if not isinstance(data.get("target"), str):
                return False
            
            return True
            
        except (json.JSONDecodeError, KeyError, TypeError):
            return False
    
    def resetContext(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
    
    # ========================================================================
    # 10.3.5 Local Functions
    # ========================================================================
    
    def _construct_payload(self, text: str, role: str = "BASELINE") -> Dict[str, Any]:
        """
        Package layered context into an API payload.

        Layers (injected each call):
          1) persistent base system prompt
          2) compact shared context (regenerated each call)
          3) small role overlay
          4) very short rolling conversation history

        Args:
            text: Current user input
            role: Role overlay type (BASELINE, ERROR, RECOVERY)

        Returns:
            JSON payload for LLM API
        """

        # Construct system instruction stack (base + shared context + role overlay)
        system_blocks = self._construct_system_instruction(role)

        # Layer 4: very short rolling conversation history (trim aggressively)
        short_history: List[Dict[str, str]] = []
        try:
            tail = self.conversation_history[-4:]
            for c in tail:
                # support both types.Content-like objects (with parts) and our Message dataclass
                content = ""
                role_name = getattr(c, "role", "user")
                if hasattr(c, "parts"):
                    parts = getattr(c, "parts", [])
                    text_parts = []
                    for p in parts:
                        txt = getattr(p, "text", None)
                        if txt:
                            text_parts.append(txt)
                    content = "\n".join(text_parts)
                elif hasattr(c, "content"):
                    content = getattr(c, "content")

                if content:
                    short_history.append({"role": role_name, "content": content})
        except Exception:
            short_history = []

        messages = []
        messages.extend(system_blocks)
        messages.append({"role": "user", "content": text})
        messages.extend(short_history)

        payload = {
            "model": self.api_config.model_id,
            "messages": messages,
            "temperature": self.api_config.temperature,
            "max_tokens": self.api_config.max_tokens,
            "response_schema": COMMAND_SCHEMA if role in ("BASELINE", "RECOVERY") else None,
        }

        return payload

    def _build_shared_context(self) -> str:
        """Build compact shared context injected every call.

        Contains the immutable primary goal (if set) and the execution contract.
        This is regenerated each call and kept minimal for token efficiency.
        """
        parts: List[str] = []
        if self.primary_goal:
            parts.append(f"Primary user goal: {self.primary_goal}")
        else:
            parts.append("Primary user goal: <not set>")
        # Encourage token efficiency
        parts.append("Token-efficiency: prefer concise outputs and avoid verbose internal deliberation.")
        contract = (
            "Execution contract: The browser controller executes all substeps autonomously. "
            "The model MUST NOT plan substeps or restart the primary goal. "
            "Substep reasoning is allowed ONLY during recovery after a browser error."
        )
        parts.append(contract)

        if self.active_browser_error:
            try:
                meta = json.dumps(self.active_browser_error)
                parts.append(f"Active browser error metadata: {meta}")
            except Exception:
                parts.append("Active browser error metadata: <unserializable>")

        return "\n".join(parts)



    def accept_command_as_primary(self, cmd: Optional[Command] = None, user_text: Optional[str] = None) -> None:
        """
        Mark the given command (or the stored pending command) as the immutable primary goal.
        This should be called after the assistant's BASELINE command is presented to the user
        and the user confirms it represents their intended primary goal.
        """
        if self.primary_goal is not None:
            raise InitializationError("Primary goal already set for this session and is immutable.")
        if not self.awaiting_primary_goal_confirmation:
            raise InitializationError("No pending primary command to accept.")
        # Use the stored pending command and user text if not provided
        cmd = cmd or self._pending_primary_command
        user_text = user_text or self._pending_primary_user_text
        if cmd is None or user_text is None:
            raise InitializationError("No pending command or user text to accept as primary goal.")
        # Set the primary goal as the user text (or command as string)
        self.primary_goal = user_text
        self.awaiting_primary_goal_confirmation = False
        self._pending_primary_command = None
        self._pending_primary_user_text = None

    def _reset_pending_goal_state(self) -> None:
        """
        Reset all state related to pending or proposed primary goal, including confirmation and mode.
        """
        self.awaiting_primary_goal_confirmation = False
        self._pending_primary_command = None
        self._pending_primary_user_text = None
        self.mode = Mode.BASELINE

    def decline_proposed_command(self) -> None:
        """
        Called when the user declines the assistant's proposed command as their primary goal.
        This leaves `primary_goal` unset so the user can provide a corrected instruction.
        """
        self._reset_pending_goal_state()

    def cancel_primary_goal(self) -> None:
        """
        Explicitly cancel the current primary goal (developer-controlled). Clears state
        so a new primary goal can be set later.
        """
        self.primary_goal = None
        self._reset_pending_goal_state()


    def handle_browser_feedback(self, error_data: Dict[str, Any]) -> str:
        """
        Sets active error metadata and returns the plain human-readable text generated by the ERROR overlay. This bypasses
        JSON schema validation for the ERROR role.
        """
        self.mode = Mode.BLOCKED
        self.active_browser_error = error_data

        payload = self._construct_payload(text="", role="ERROR")
        raw = self._call_llm_api(payload)
        return raw

    def processRecoveryInput(self, text: str):
        """Convert user's recovery input into a minimal plain-text command scoped to the failed substep.

        Returns either a plain-text command or a clarifying question (str) when the model requests a single piece of missing information.
        Tracks all steps in the messages array.
        """
        if not self.active_browser_error:
            raise InferenceError("No active browser error to recover from")

        self._update_history(Message(role="recovery", content=text, tokens=max(1, len(text.split()))))
        payload = self._construct_payload(text=text, role="RECOVERY")
        try:
            raw = self._call_llm_api(payload)
            guarded = self._apply_guardrails(raw)
            self._update_history(Message(role="assistant", content=guarded, tokens=max(1, len(guarded.split()))))
            self.active_browser_error = None
            self.mode = Mode.BASELINE
            return guarded
        except Exception as e:
            if self.error_feedback_module:
                self.error_feedback_module.report_error("CommandOrchestrator", str(e))
            raise InferenceError(f"GenAI inference failed: {str(e)}")
        
    
    def _update_history(self, msg: Message) -> None:
        """
        Manage context window by removing oldest messages if needed.
        
        Args:
            msg: New message to add to history
        """
        self.conversation_history.append(msg)

        # Calculate total tokens (simple sum over Message.tokens)
        try:
            total_tokens = sum(int(getattr(m, "tokens", 0) or 0) for m in self.conversation_history)
        except Exception:
            total_tokens = 0

        # Remove oldest messages if we exceed MAX_CONTEXT_TOKENS
        # TODO: ensure required context does not get trimmed
        while total_tokens > MAX_CONTEXT_TOKENS and len(self.conversation_history) > 1:
            removed = self.conversation_history.pop(0)
            total_tokens -= int(getattr(removed, "tokens", 0) or 0)
    
    def _call_llm_api(self, payload: Dict[str, Any]) -> str:
        """
        Make actual API call to LLM service.
        
        Args:
            payload: API request payload
            
        Returns:
            Raw response text from LLM
            
        Raises:
            InferenceError: If API call fails
        """
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.api_config.endpoint_url,
                json=payload,
                headers=headers,
                timeout=self.api_config.timeout
            )
            
            response.raise_for_status()
            
            data = response.json()
            
            # Extract content based on provider
            # Gemini format
            if "candidates" in data:
                return data["candidates"][0]["content"]["parts"][0]["text"]
            # OpenAI format  
            elif "choices" in data:
                return data["choices"][0]["message"]["content"]
            # Anthropic format
            elif "content" in data:
                return data["content"][0]["text"]
            else:
                raise InferenceError(f"Unexpected API response format: {data}")
                            
        except requests.Timeout:
            raise InferenceError("API request timed out")
        except requests.RequestException as e:
            raise InferenceError(f"API request failed: {str(e)}")
        except Exception as e:
            raise InferenceError(f"Unexpected error calling API: {str(e)}")
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into command dictionary.
        
        Args:
            response: Raw text response from LLM
            
        Returns:
            Dictionary representing command
            
        Raises:
            InferenceError: If response cannot be parsed
        """
        try:
            # Try to extract JSON from response
            # LLMs sometimes wrap JSON in markdown code blocks
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            
            # Parse JSON
            command_dict = json.loads(response)
            
            return command_dict
            
        except json.JSONDecodeError as e:
            raise InferenceError(f"Failed to parse LLM response as JSON: {str(e)}")

    def _apply_guardrails(self, obj):
        """
        Apply safety guardrails to either a Command object or a plain-text command string.
        For Command: sets confirmation_required if dangerous.
        For str: prepends a warning if dangerous.
        """
        dangerous_keywords = [
            "delete", "clear", "remove", "reset", "logout",
            "purchase", "buy", "payment", "transfer",
            "bank", "paypal", "stripe"
        ]
        if isinstance(obj, Command):
            # Check if target or value contains dangerous keywords
            target_lower = obj.target.lower()
            value_lower = (obj.value or "").lower()
            for keyword in dangerous_keywords:
                if keyword in target_lower or keyword in value_lower:
                    obj.confirmation_required = True
                    break
            # Financial URLs require confirmation
            financial_domains = ["bank", "paypal", "stripe", "payment"]
            if obj.action == "navigate":
                for domain in financial_domains:
                    if domain in obj.target.lower():
                        obj.confirmation_required = True
                        break
            return obj
        elif isinstance(obj, str):
            lower = obj.lower()
            flagged = False
            for keyword in dangerous_keywords:
                if keyword in lower:
                    flagged = True
                    break
            if flagged:
                warning = "[CONFIRMATION REQUIRED] This command may be dangerous. Please review before executing.\n"
                return warning + obj
            return obj
        else:
            return obj
