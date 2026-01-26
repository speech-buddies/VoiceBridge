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
    Orchestrates natural language to browser command translation using LLM API.
    
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
        
        self.system_prompt = self._load_system_prompt(self.prompt_path)

        # Conversation history uses local Message dataclass for compact bookkeeping
        self.conversation_history: List[Message] = []
        
        # Integration placeholders
        self.user_profile_manager = user_profile_manager
        self.error_feedback_module = error_feedback_module
        self.security_layer = security_layer

        # Minimal orchestrator state per design
        self.primary_goal: Optional[str] = None
        self.mode: Mode = Mode.BASELINE
        self.active_browser_error: Optional[Dict[str, Any]] = None

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
            "ERROR": "Analyze the browser error relative to the primary goal. Produce a short user-facing explanation and TWO options: (a) manual resolution instructions, (b) assisted automation listing exact info required. DO NOT output browser commands or JSON.",
            "RECOVERY": "Take the user's recovery input and produce the minimal browser command required to unblock the failed substep. Output must be valid JSON matching the command schema and scope-limited to the failed substep.",
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


    def processTranscript(self, text: str) -> Command:
        """
        Converts natural language to a Command using Structured Outputs.
        """
        # If no primary goal is set yet, keep it external: we don't auto-set here.
        # Routing by mode
        if self.mode == Mode.BLOCKED:
            # In BLOCKED mode, treat incoming transcript as recovery input
            return self.processRecoveryInput(text)

        # BASELINE happy path: produce a single high-level browser command
        # Build payload using layered context
        payload = self._construct_payload(text=text, role="BASELINE")

        try:
            raw = self._call_llm_api(payload)
            # parse LLM output
            command_dict = self._parse_llm_response(raw)

            # Support optional 'messages' returned by the model; remove before building Command
            messages_from_model = None
            if isinstance(command_dict, dict) and "messages" in command_dict:
                messages_from_model = command_dict.pop("messages")

            # Build Command object from remaining keys
            command = Command(**command_dict)
            command = self._apply_guardrails(command)

            # If the model provided compact assistant messages, append them to history
            try:
                if messages_from_model and isinstance(messages_from_model, list):
                    for m in messages_from_model:
                        role = m.get("role", "assistant")
                        content = m.get("content", "")
                        if content:
                            self._update_history(Message(role=role, content=content, tokens=max(1, len(content.split()))))
            except Exception:
                pass
           
            # command_dict = self._parse_llm_response(raw)

            # # Command dict should only contain command fields
            # command = Command(**command_dict)
            # command = self._apply_guardrails(command)

            # # Record minimal history entry (avoid large tokens) using Message dataclass
            # try:
            #     user_msg = Message(role="user", content=text, tokens=max(1, len(text.split())))
            #     assistant_msg = Message(role="assistant", content=json.dumps(command.to_dict()), tokens=max(1, len(json.dumps(command.to_dict()).split())))
            #     self._update_history(user_msg)
            #     self._update_history(assistant_msg)
            # except Exception:
            #     pass

            return command

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
        # Layer 1: persistent base system prompt
        base = self.system_prompt

        # Layer 2: compact shared context always regenerated
        shared_ctx = self._build_shared_context()

        # Layer 3: small role overlay
        overlay = self._role_overlays.get(role, "")

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

        messages = [
            {"role": "system", "content": base},
            {"role": "system", "content": shared_ctx},
            {"role": "system", "content": overlay},
            {"role": "user", "content": text},
        ]
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

    def set_primary_goal(self, goal: str) -> None:
        """Set or overwrite the primary goal for the session."""
        # allowed to set or overwrite via explicit API
        self.primary_goal = goal

    def report_browser_error(self, error_meta: Dict[str, Any]) -> str:
        """Called by Browser Controller to notify an error; returns error explanation string."""
        self.mode = Mode.BLOCKED
        self.active_browser_error = error_meta

        payload = self._construct_payload(text="", role="ERROR")
        raw = self._call_llm_api(payload)
        return raw

    def processRecoveryInput(self, text: str) -> Command:
        """Convert user's recovery input into a minimal command scoped to the failed substep."""
        if not self.active_browser_error:
            raise InferenceError("No active browser error to recover from")

        payload = self._construct_payload(text=text, role="RECOVERY")
        raw = self._call_llm_api(payload)
        cmd_dict = self._parse_llm_response(raw)

        # Support optional 'messages' field in RECOVERY JSON. Extract before Command construction.
        messages_from_model = None
        if isinstance(cmd_dict, dict) and "messages" in cmd_dict:
            messages_from_model = cmd_dict.pop("messages")

        cmd = Command(**cmd_dict)
        cmd = self._apply_guardrails(cmd)

        # If the model supplied messages for history bookkeeping, append them.
        try:
            if messages_from_model and isinstance(messages_from_model, list):
                for m in messages_from_model:
                    role = m.get("role", "assistant")
                    content = m.get("content", "")
                    if content:
                        self._update_history(Message(role=role, content=content, tokens=max(1, len(content.split()))))
        except Exception:
            pass

        # Clear error and resume baseline
        self.active_browser_error = None
        self.mode = Mode.BASELINE

        return cmd
    
    def _apply_guardrails(self, cmd: Command) -> Command:
        """
        Check generated command against safety policies.
        
        Args:
            cmd: Command to validate
            
        Returns:
            Modified command with safety flags applied
        """
        # TODO: update guardrails
        # Dangerous actions that require confirmation
        dangerous_keywords = [
            "delete", "clear", "remove", "reset", "logout",
            "purchase", "buy", "payment", "transfer"
        ]
        
        # Check if target or value contains dangerous keywords
        target_lower = cmd.target.lower()
        value_lower = (cmd.value or "").lower()
        
        for keyword in dangerous_keywords:
            if keyword in target_lower or keyword in value_lower:
                cmd.confirmation_required = True
                break
        
        # Financial URLs require confirmation
        financial_domains = ["bank", "paypal", "stripe", "payment"]
        if cmd.action == "navigate":
            for domain in financial_domains:
                if domain in cmd.target.lower():
                    cmd.confirmation_required = True
                    break
        
        return cmd
    
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
