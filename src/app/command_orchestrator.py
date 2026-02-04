"""
CommandOrchestrator Module - Simplified Version

Handles command clarification for natural language browser control:
- Determines if user intent is clear
- Asks clarifying questions when needed
- Returns clarified natural language commands
- The actual browser automation is handled by an LLM-powered browser controller
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class OrchestratorResponse:
    """
    Response from the orchestrator that clearly indicates what should happen next.
    """
    # Indicates if we need user input before proceeding
    needs_clarification: bool
    
    # If needs_clarification=True, this is the question to ask the user
    user_prompt: Optional[str] = None
    
    # If needs_clarification=False, this is the clarified natural language command
    # to pass to the browser controller
    clarified_command: Optional[str] = None
    
    # Additional context for debugging
    reasoning: Optional[str] = None
    
    # Metadata from the orchestrator's processing
    metadata: Optional[Dict[str, Any]] = None


# ============================================================================
# Exceptions
# ============================================================================

class OrchestratorError(Exception):
    """Base exception for orchestrator errors."""
    pass


class InitializationError(OrchestratorError):
    """Raised when orchestrator cannot be initialized."""
    pass


class InferenceError(OrchestratorError):
    """Raised when LLM API fails."""
    pass


# ============================================================================
# CommandOrchestrator - Simplified
# ============================================================================

class CommandOrchestrator:
    """
    Orchestrates natural language to browser commands using an LLM.
    
    Key responsibilities:
    1. Parse user transcript into browser commands
    2. Ask clarifying questions when needed
    3. Maintain conversation context for multi-turn interactions
    4. Apply safety guardrails
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "gemini-2.5-flash",
        prompt_path: Optional[str] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            api_key: LLM API key (or use GEMINI_API_KEY env var)
            model_id: Model to use
            prompt_path: Path to system prompt file
        """
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise InitializationError("GEMINI_API_KEY not found")
        
        # Initialize the Gen AI Client
        try:
            from google import genai
            self.client = genai.Client(api_key=self._api_key)
        except ImportError:
            raise InitializationError("google-genai package not installed. Run: pip install google-genai")
        
        self.model_id = model_id
        
        # Load system prompt
        if prompt_path and Path(prompt_path).exists():
            self.system_prompt = Path(prompt_path).read_text(encoding="utf-8")
        else:
            # Default prompt
            self.system_prompt = self._get_default_system_prompt()
        
        # Conversation history for multi-turn interactions
        # Format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        self.conversation_history: List[Dict[str, str]] = []
        
        # Current state
        self.current_goal: Optional[str] = None
    
    def _get_default_system_prompt(self) -> str:
        """Default system prompt if no file provided."""
        return """You are a command clarification assistant for a browser automation system.

Your job is to determine if the user's intent is clear enough to pass to an LLM-powered browser controller.

The browser controller can handle natural language commands like:
- "Navigate to Gmail"
- "Click the login button"
- "Search for cats on Google"
- "Fill out the contact form with my name and email"

You should ask for clarification when:
1. The command is ambiguous (e.g., "Go to Google" - which Google service?)
2. The command is too vague (e.g., "Do something" - what specifically?)
3. Critical information is missing (e.g., "Send an email" - to whom? what content?)

When the user's intent is CLEAR, respond with:
{
    "needs_clarification": false,
    "clarified_command": "The clear natural language command to pass to the browser controller"
}

When you need clarification, respond with:
{
    "needs_clarification": true,
    "question": "Your specific question to the user"
}

Examples:

User: "Open Gmail"
{
    "needs_clarification": false,
    "clarified_command": "Navigate to Gmail"
}

User: "Go to Google"
{
    "needs_clarification": true,
    "question": "Which Google service? Google.com, Gmail, Google Drive, or something else?"
}

User (after being asked which Google service): "Gmail"
Context: Previous conversation shows they said "Go to Google"
{
    "needs_clarification": false,
    "clarified_command": "Navigate to Gmail"
}

User: "Search for cats"
{
    "needs_clarification": false,
    "clarified_command": "Search for cats on Google"
}

User: "Send an email"
{
    "needs_clarification": true,
    "question": "Who would you like to send an email to, and what should it say?"
}

Be concise and helpful. Only ask for clarification when truly necessary. The browser controller is intelligent and can handle natural language well."""
    
    async def process(
        self,
        user_input: str,
        conversation_context: Optional[List[Dict[str, str]]] = None
    ) -> OrchestratorResponse:
        """
        Process user input and return either a clarified command or a clarification request.
        
        This method determines if the user's intent is clear. If clear, it returns a 
        clarified natural language command to pass to the browser controller. If unclear,
        it asks a clarifying question.
        
        Args:
            user_input: The user's transcript or response
            conversation_context: Optional conversation history from the server
                                Format: [{"role": "user|assistant", "content": "..."}]
        
        Returns:
            OrchestratorResponse indicating what to do next:
            - If needs_clarification=True: Ask user the question in user_prompt
            - If needs_clarification=False: Pass clarified_command to browser controller
            
        Examples:
            # Clear command
            response = await orchestrator.process("Open Gmail")
            # response.needs_clarification = False
            # response.clarified_command = "Navigate to Gmail"
            
            # Needs clarification
            response = await orchestrator.process("Go to Google")
            # response.needs_clarification = True
            # response.user_prompt = "Which Google service? Google.com, Gmail, ..."
        """
        # Build conversation context
        messages = []
        
        # Add conversation history if provided
        if conversation_context:
            messages.extend(conversation_context)
        
        # Add current user input
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Call LLM
        try:
            response_text = await self._call_llm(messages)
            
            # Parse response
            return self._parse_response(response_text, user_input)
            
        except Exception as e:
            raise InferenceError(f"Failed to process input: {str(e)}")
    
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM API with the conversation history.
        
        Args:
            messages: Conversation messages
            
        Returns:
            LLM response text
        """
        try:
            from google.genai import types
            
            # Construct the full conversation with system prompt
            full_messages = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=self.system_prompt)]
                )
            ]
            
            # Add conversation messages
            for msg in messages:
                role = "user" if msg["role"] == "user" else "model"
                full_messages.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=msg["content"])]
                    )
                )
            
            # Generate response
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=full_messages,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=512,
                )
            )
            
            return response.text
            
        except Exception as e:
            raise InferenceError(f"LLM API call failed: {str(e)}")
    
    def _parse_response(self, response_text: str, original_input: str) -> OrchestratorResponse:
        """
        Parse LLM response into an OrchestratorResponse.
        
        Args:
            response_text: Raw LLM response
            original_input: Original user input
            
        Returns:
            Parsed OrchestratorResponse
        """
        try:
            # Clean up response (remove markdown code blocks if present)
            cleaned = response_text.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:].strip()
            
            # Parse JSON
            data = json.loads(cleaned)
            
            # Check if it's a clarification request
            if data.get("needs_clarification"):
                return OrchestratorResponse(
                    needs_clarification=True,
                    user_prompt=data.get("question", "Could you clarify what you'd like me to do?"),
                    metadata={"original_input": original_input}
                )
            
            # Otherwise, it's a clarified command
            clarified_command = data.get("clarified_command", original_input)
            
            return OrchestratorResponse(
                needs_clarification=False,
                clarified_command=clarified_command,
                metadata={"original_input": original_input}
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # If we can't parse the response, treat it as needing clarification
            return OrchestratorResponse(
                needs_clarification=True,
                user_prompt="I'm not sure I understand. Could you rephrase that?",
                reasoning=f"Failed to parse LLM response: {str(e)}",
                metadata={"original_input": original_input, "raw_response": response_text}
            )
    
    def reset(self):
        """Reset orchestrator state."""
        self.conversation_history = []
        self.current_goal = None


# ============================================================================
# Convenience Functions
# ============================================================================

def create_orchestrator(
    api_key: Optional[str] = None,
    model_id: str = "gemini-3.5-flash",
    prompt_path: Optional[str] = None
) -> CommandOrchestrator:
    """
    Factory function to create a CommandOrchestrator.
    
    Args:
        api_key: LLM API key
        model_id: Model to use
        prompt_path: Path to system prompt
        
    Returns:
        Initialized CommandOrchestrator
    """
    return CommandOrchestrator(
        api_key=api_key,
        model_id=model_id,
        prompt_path=prompt_path
    )