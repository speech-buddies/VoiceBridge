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
        # The new SDK handles history as a list of Content objects
        self.conversation_history: List[types.Content] = []
        
        # Integration placeholders
        self.user_profile_manager = user_profile_manager
        self.error_feedback_module = error_feedback_module
        self.security_layer = security_layer

    def _load_system_prompt(self, path: str) -> str:
        """Loads prompt from a file. Falls back to a basic string if file is missing."""
        try:
            return Path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"Warning: Prompt file {path} not found. Using default.")
            return "You are a helpful browser assistant."
    
    
    def processTranscript(self, text: str) -> Command:
        """
        Converts natural language to a Command using Structured Outputs.
        """
        # Append user message to history
        self.conversation_history.append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )

        try:
            # Generate content with native JSON schema enforcement
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=self.conversation_history,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                    response_schema=COMMAND_SCHEMA,  # Enforces JSON structure
                    temperature=0.2,
                ),
            )

            command_dict = json.loads(response.text)
            command = Command(**command_dict)
            
            # Apply safety guardrails (logic remains the same)
            command = self._apply_guardrails(command)

            # Record assistant response in history
            self.conversation_history.append(
                types.Content(role="model", parts=[types.Part(text=response.text)])
            )

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
    
    def _construct_payload(self, text: str) -> Dict[str, Any]:
        """
        Package system prompt and conversation history into API format.
        
        Args:
            text: Current user input
            
        Returns:
            JSON payload for LLM API
        """
        # TODO: ensure required context is available
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history
        messages.extend([msg.to_dict() for msg in self.conversation_history])
        
        payload = {
            "model": self.api_config.model_id,
            "messages": messages,
            "temperature": self.api_config.temperature,
            "max_tokens": self.api_config.max_tokens
        }
        
        return payload
    
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
        
        # Calculate total tokens
        total_tokens = sum(m.tokens for m in self.conversation_history)
        
        # Remove oldest messages if we exceed MAX_CONTEXT_TOKENS
        # TODO: ensure required context does not get trimmed
        while total_tokens > MAX_CONTEXT_TOKENS and len(self.conversation_history) > 1:
            removed = self.conversation_history.pop(0)
            total_tokens -= removed.tokens
    
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
