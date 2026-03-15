"""
FastAPI Server Entry Point - Improved Version

This version uses a simplified CommandOrchestrator that focuses solely on
command clarification, not command generation. The orchestrator determines
if the user's intent is clear and asks clarifying questions when needed.

The actual browser automation is handled by an LLM-powered browser controller
that can understand natural language commands directly.

Key improvements:
1. Cleaner parse_and_execute_command function
2. Better state management for conversation flow
3. Clear separation: Orchestrator for clarification, Browser controller for execution
"""

import asyncio
from utils import logger 

import sys
import threading
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from state_manager import StateManager, AppState
from control.browser_orchestrator import run_command
from control.session_manager import get_browser, start_session, stop_session

from app.command_orchestrator import CommandOrchestrator, OrchestratorError
from app.speech_to_text_engine import SpeechToTextEngine
from app.user_profile_manager import UserProfileManager, DEFAULT_PROFILE_ID, ProfileNotFoundError
from app.shortcut_manager import ShortcutManager
from data import training_data_recorder
from data.feedback_store import FeedbackStore

# Windows compatibility
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Audio capture lazy import
def _get_audio_capture():
    try:
        from input.audio_capture import RealtimeAudioCapture, AudioConfig
        return RealtimeAudioCapture, AudioConfig, None
    except ImportError as e:
        return None, None, str(e)


logger = logger.get_logger("VoiceBridgeServer")

_capture_instance = None
_capture_lock = threading.Lock()
_event_loop = None


# ============================================================================
# Pydantic Models
# ============================================================================

class StatusResponse(BaseModel):
    # Response model for server status
    state: str
    timestamp: str
    has_audio: bool
    has_transcript: bool
    error: Optional[str]
    transcript: Optional[str] = None
    user_transcript: Optional[str] = None  # Latest user input
    clarified_command: Optional[str] = None  # LLM-clarified version
    last_command: Optional[str] = None
    user_prompt: Optional[str] = None       # Message shown to user (clarification OR confirmation summary)
    awaiting_confirmation: bool = False     # True only when a complete command awaits yes/no
    pending_command: Optional[str] = None   # The command text held until confirmed
    last_action: Optional[str] = None       # "confirmed" | "cancelled" | None — explicit outcome signal
    cache_stats: Optional[dict] = None

class FeedbackRequest(BaseModel):
    command_id: str
    feedback_type: str
    value: str
    source: str
    command_text: str = None

class PreferencesUpdate(BaseModel):
    guardrails_enabled:       Optional[bool] = None
    custom_training_enabled:  Optional[bool] = None
    custom_shortcuts_enabled: Optional[bool] = None

class ShortcutCreate(BaseModel):
    phrase:  str
    command: str

# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manages WebSocket connections for real-time state updates"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting: {e}")
                disconnected.append(connection)
        
        for conn in disconnected:
            self.disconnect(conn)


# ============================================================================
# Audio Capture Callback
# ============================================================================

def _on_audio_ready(audio_float32: np.ndarray, metadata: dict):
    """
    Callback when audio capture detects silence.
    Routes to voice pipeline or saves to file.
    """
    global server, _event_loop
    if server and server.is_voice_mode_active and _event_loop:
        # Feed into voice pipeline
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        asyncio.run_coroutine_threadsafe(
            server.on_silence_detected(audio_bytes), _event_loop
        )


# ============================================================================
# VoiceBridge Server 
# ============================================================================

class VoiceBridgeServer:
    """
    Main server for voice automation pipeline.

    """
    
    def __init__(self):
        self.state_manager = StateManager()
        self.connection_manager = ConnectionManager()
        self.is_voice_mode_active = False
        self._browser_session_started = False
        self._session_lock = asyncio.Lock()
        self.shortcut_manager = ShortcutManager()

        self._command_processing_lock = asyncio.Lock()
        
    
        # Initialize modules
        self.speech_to_text = SpeechToTextEngine()
        
        # Initialize command orchestrator
        try:
            self.command_orchestrator = CommandOrchestrator()
            logger.info("CommandOrchestrator initialized")
        except OrchestratorError as e:
            logger.error(f"Failed to initialize CommandOrchestrator: {e}")
            self.command_orchestrator = None
        
        # Register state change callbacks
        self.state_manager.register_callback(None, self._broadcast_state_change)
        self.state_manager.register_callback(None, self._on_state_change)
        
        # Tracking variables
        self._last_transcript: Optional[str] = None
        self._last_command: Optional[str] = None
        self._user_transcript: Optional[str] = None  # Latest user input (command or response)
        self._clarified_command: Optional[str] = None  # LLM-clarified version
        
        # Conversation context for multi-turn interactions
        self._conversation_context: List[dict] = []
        self._current_user_prompt: Optional[str] = None

        # Canonical first-turn transcript for the current session.
        # Captured before conversation context is updated and used as the cache key.
        # Cleared after execution, cancellation, error, or reset.
        self._initial_intent: Optional[str] = None

        # Confirmation gate: True only while waiting for the user to say yes/no
        # _pending_command holds the clarified command text until confirmed
        self._awaiting_confirmation: bool = False
        self._pending_command: Optional[str] = None
        self._pending_audio_turns: list[bytes] = []
        self._pending_transcript_turns: list[str] = []
        self._last_action: Optional[str] = None  # "confirmed" | "cancelled" | None

        self.feedback_store = FeedbackStore()

        # Voice customisation
        self.profile_manager = UserProfileManager()
        self._shortcuts: dict = {}   # phrase -> command
    
    def _broadcast_state_change(self, state_data, old_state):
        """Broadcast state changes to WebSocket clients"""
        message = {
            'type': 'state_change',
            'old_state': old_state.value,
            'new_state': state_data.state.value,
            'timestamp': state_data.timestamp.isoformat(),
            'has_transcript': state_data.transcript is not None,
            'transcript': state_data.transcript,
            'error': state_data.error,
            'user_prompt': self._current_user_prompt
        }
        
        logger.info(f"State: {old_state.value} -> {state_data.state.value}")
        if self._current_user_prompt:
            logger.info(f"User prompt: {self._current_user_prompt}")
        
        asyncio.create_task(self.connection_manager.broadcast(message))
    
    def _on_state_change(self, state_data, old_state):
        """Handle state changes"""
        asyncio.create_task(self._sync_browser_session_with_state(state_data.state))
    
    async def _sync_browser_session_with_state(self, new_state: AppState):
        """
        Auto start/stop browser session based on state.
        LISTENING -> ensure browser session started
        IDLE -> ensure browser session stopped
        """
        async with self._session_lock:
            if new_state == AppState.LISTENING:
                if not self._browser_session_started:
                    msg = await start_session()
                    logger.info(f"Browser session started: {msg}")
                    self._browser_session_started = True
            
            elif new_state == AppState.IDLE:
                if self._browser_session_started:
                    msg = await stop_session()
                    logger.info(f"Browser session stopped: {msg}")
                    self._browser_session_started = False
    
    # ========================================================================
    # Core Processing Methods
    # ========================================================================
    
    async def parse_and_execute_command(
        self,
        transcript: str,
        conversation_context: Optional[List[dict]] = None,
        initial_intent: Optional[str] = None,
    ) -> dict:
        """
        Parse transcript and execute browser command.
        
        This method uses the CommandOrchestrator to determine if the user's intent
        is clear. If clear, it passes the clarified natural language command to the
        LLM-powered browser controller. If unclear, it asks for clarification.
        
        Flow:
        1. Call orchestrator to check if command is clear
        2. If orchestrator needs clarification -> return user_prompt
        3. If orchestrator returns clarified command -> pass to browser controller
        
        Args:
            transcript: User's transcript
            conversation_context: Conversation history
            
        Returns:
            dict with either:
            - {"needs_input": True, "user_prompt": "...", "context": {...}}
            - {"needs_input": False, "success": True, "command": "...", "details": "..."}
        """
        logger.info(f"parse_and_execute_command: '{transcript}'")
        
        if not transcript.strip():
            raise HTTPException(status_code=400, detail="Empty transcript")
        
        # Check if orchestrator is available
        if not self.command_orchestrator:
            logger.warning("CommandOrchestrator not available, using fallback")
            return await self._fallback_execution(transcript)
        
        try:
            # Step 1: Check if command is clear
            # Convert our conversation context to orchestrator format
            orchestrator_context = []
            if conversation_context:
                for msg in conversation_context:
                    orchestrator_context.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            # Call orchestrator. Pass initial_intent so it can use the first-turn transcript as the cache key.
            response = await self.command_orchestrator.process(
                user_input=transcript,
                conversation_context=orchestrator_context,
                initial_intent=initial_intent,
            )
            
            # Step 2: Handle clarification
            if response.needs_clarification:
                # Orchestrator needs more info from user
                logger.info(f"Orchestrator needs clarification: {response.user_prompt}")
                return {
                    "needs_input": True,
                    "user_prompt": response.user_prompt,
                    "context": response.metadata or {}
                }
            
            # Step 3: Get the clarified natural language command
            clarified_command = response.clarified_command
            self._clarified_command = clarified_command  # Store clarified version
            # Guardrails check before logging or execution
            allowed, guardrails_msg = self.command_orchestrator.apply_guardrails(clarified_command)
            if not allowed:
                return {
                    "needs_input": True,
                    "user_prompt": guardrails_msg or "This command is not allowed. Please try something else.",
                    "context": {
                        "last_command": clarified_command
                    }
                }
            logger.info(f"Orchestrator clarified command: '{clarified_command}'")

            # ----------------------------------------------------------------
            # Complete command ready → enter confirmation gate.
            # We stay in LISTENING state; _awaiting_confirmation=True means
            # the next audio chunk will be routed to _process_confirmation.
            # ----------------------------------------------------------------
            self._pending_command = clarified_command
            self._awaiting_confirmation = True
            # Build user-facing command summary.
            # Remove continuation clauses (e.g., "and keep", "until", "while")
            # so the confirmation prompt shows only the primary action.
            _trim_markers = [" and keep", " and keep it", " until the user", " while ", " unless"]
            _display = clarified_command
            for _marker in _trim_markers:
                if _marker in _display.lower():
                    _idx = _display.lower().index(_marker)
                    _display = _display[:_idx]
            _display = _display.rstrip(".,; ")
            self._current_user_prompt = f'"{_display}" — say yes to confirm or no to cancel.'
            self._last_command = clarified_command

            return {
                "needs_input": True,
                "awaiting_confirmation": True,
                "user_prompt": self._current_user_prompt,
            }
            
        except OrchestratorError as e:
            logger.error(f"Orchestrator error: {e}")
            self.state_manager.transition_to(AppState.ERROR, error=str(e))
            raise HTTPException(status_code=500, detail=f"Orchestrator error: {e}")
        
        except Exception as e:
            logger.error(f"Error in parse_and_execute_command: {e}")
            self.state_manager.transition_to(AppState.ERROR, error=str(e))
            raise HTTPException(status_code=500, detail=f"Command execution error: {e}")
    
    async def _execute_browser_command(self, clarified_command: str) -> dict:
        """
        Execute a natural language command via the browser controller.
        
        The browser controller is LLM-powered and can handle natural language directly.
        
        Args:
            clarified_command: Clarified natural language command (e.g., "Navigate to Gmail")
            
        Returns:
            Execution result
        """
        # Get active browser session
        browser = get_browser()
        if browser is None:
            raise HTTPException(
                status_code=409,
                detail="No active browser session"
            )

        try:
            # Use the LLM-powered browser orchestrator to execute natural language command
            history = await run_command(clarified_command, browser)
            
            return {
                "success": True,
                "history": str(history),
                "command": clarified_command
            }
            
        except Exception as e:
            logger.error(f"Browser execution error: {e}")
            raise
    
    async def _fallback_execution(self, transcript: str) -> dict:
        """
        Fallback execution when orchestrator is not available.
        Uses the browser orchestrator directly.
        """
        browser = get_browser()
        if browser is None:
            raise HTTPException(status_code=409, detail="No active browser session")
        
        self.state_manager.transition_to(AppState.EXECUTING, transcript=transcript)
        
        try:
            history = await run_command(transcript, browser)
            return {
                "needs_input": False,
                "success": True,
                "action": "browser_orchestrator",
                "details": str(history)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Browser error: {e}")

    async def run_shortcut(self, shortcut_id: str) -> dict:
        """Run all commands of a shortcut by id. Returns after all commands are executed."""
        shortcuts = self.shortcut_manager.list_shortcuts()
        shortcut = shortcuts.get(shortcut_id)
        if not shortcut:
            raise HTTPException(status_code=404, detail=f"Shortcut {shortcut_id} not found")
        commands = shortcut.get("commands") or []
        if not commands:
            return {"success": True, "shortcut_id": shortcut_id, "commands_run": 0}
        browser = get_browser()
        if browser is None:
            raise HTTPException(status_code=409, detail="No active browser session")
        for cmd in commands:
            await self._execute_browser_command(cmd)
        return {"success": True, "shortcut_id": shortcut_id, "commands_run": len(commands)}
    
    # ========================================================================
    # Audio Processing Pipeline
    # ========================================================================
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio to text."""
        logger.info(f"Transcribing {len(audio_data)} bytes...")
        transcription = await self.speech_to_text.transcribe(audio_data)
        logger.info(f"Transcription: {transcription}")
        return transcription
    
    async def on_silence_detected(self, audio_data: bytes):
        """
        Called when VAD detects silence.
        Routes to appropriate handler based on current state.
        """
        logger.info(f"Silence detected, processing {len(audio_data)} bytes")
        
        current_state = self.state_manager.current_state
        
        if current_state != AppState.LISTENING:
            logger.warning(f"Received audio in unexpected state: {current_state.value}")
            return None
        async with self._command_processing_lock:
            if self._awaiting_confirmation:
                await self._process_confirmation(audio_data)
            else:
                await self._process_new_command(audio_data)


    async def _process_new_command(self, audio_data: bytes):
        """Process a new voice command from the user."""
        # Clear clarified command (new conversation starting)
        # But keep user_transcript - it will be updated with new input
        self._clarified_command = None
        self._last_action = None  
        self._current_user_prompt = None

        # Transition to processing
        self.state_manager.transition_to(AppState.PROCESSING, audio_data=audio_data)
        print("**SWITCHED TO PROCESSING")
        try:
            # Transcribe
            transcript = await self.transcribe_audio(audio_data)
            self._last_transcript = transcript
            self._user_transcript = transcript  # Store latest user input
            if not self._conversation_context:          # first turn — fresh session
                self._pending_audio_turns = []
                self._pending_transcript_turns = []
            self._pending_audio_turns.append(audio_data)
            self._pending_transcript_turns.append(transcript)

            if not transcript or not transcript.strip():
                self._current_user_prompt = "I didn't catch that."
                self.state_manager.transition_to(
                    AppState.LISTENING,
                    transcript="",
                    metadata={"user_prompt": self._current_user_prompt}
                )
                return

            # ------------------------------------------------------------
            # Shortcut control commands — intercept before orchestration
            # ------------------------------------------------------------
            normalized = transcript.strip().lower().rstrip(".,!?;:")

            if normalized == "start shortcut":
                started = self.shortcut_manager.start_recording()
                self._current_user_prompt = (
                    "Shortcut recording started."
                    if started else
                    "A shortcut is already being recorded."
                )
                self.state_manager.transition_to(
                    AppState.LISTENING,
                    transcript=transcript,
                    metadata={"user_prompt": self._current_user_prompt}
                )
                return

            if normalized == "end shortcut":
                try:
                    shortcut = self.shortcut_manager.stop_recording()
                    self._current_user_prompt = f'Shortcut "{shortcut["name"]}" saved.'
                except ValueError as e:
                    self._current_user_prompt = str(e)
                    logger.warning("Shortcut stop failed: %s", e)

                self.state_manager.transition_to(
                    AppState.LISTENING,
                    transcript=transcript,
                    metadata={"user_prompt": self._current_user_prompt}
                )
                return

            # "run shortcut <id>" — run shortcut by id without confirmation
            run_prefix = "run shortcut "
            if normalized.startswith(run_prefix):
                shortcut_id = normalized[len(run_prefix):].strip()
                if shortcut_id and shortcut_id in self.shortcut_manager.list_shortcuts():
                    try:
                        await self.run_shortcut(shortcut_id)
                        self._current_user_prompt = f'Ran shortcut "{shortcut_id}".'
                    except HTTPException as e:
                        self._current_user_prompt = e.detail or f"Could not run shortcut {shortcut_id}."
                    except Exception as e:
                        logger.exception("Error running shortcut %s", shortcut_id)
                        self._current_user_prompt = f"Error running shortcut: {e}"
                else:
                    self._current_user_prompt = f'Shortcut "{shortcut_id}" not found.'
                self.state_manager.transition_to(
                    AppState.LISTENING,
                    transcript=transcript,
                    metadata={"user_prompt": self._current_user_prompt}
                )
                return
            # ------------------------------------------------------------

            # Capture initial intent before appending — empty context means first turn
            if not self._conversation_context:
                self._initial_intent = (
                    self.command_orchestrator.cache._clean_key(transcript)
                    if self.command_orchestrator else transcript.lower().strip()
                )
                logger.info(f"Session start: initial_intent='{self._initial_intent}'")

            # Add to conversation context
            self._conversation_context.append({
                "role": "user",
                "content": transcript,
                "timestamp": self.state_manager.state_data.timestamp.isoformat()
            })
            
            # Parse and execute. Pass initial_intent for cache consistency.
            result = await self.parse_and_execute_command(
                transcript,
                self._conversation_context,
                initial_intent=self._initial_intent,
            )
            
            # Handle result
            if result.get("needs_input"):
                # Either a clarification question OR a confirmation prompt, return to LISTENING so the user can respond.
                self._current_user_prompt = result["user_prompt"]

                self._conversation_context.append({
                    "role": "assistant",
                    "content": result["user_prompt"],
                    "timestamp": self.state_manager.state_data.timestamp.isoformat()
                })

                self.state_manager.transition_to(
                    AppState.LISTENING,
                    transcript=transcript,
                    metadata={"user_prompt": result["user_prompt"]}
                )
            
            else:
                # Command executed successfully — close the confirmation window
                self._last_command = transcript
                self._current_user_prompt = None
                self._awaiting_confirmation = False
                self._pending_command = None
                # Keep user_transcript and clarified_command for display
                # They will be updated on next user input
                
                # Clear conversation context
                self._conversation_context = []
                
                # Back to listening
                await asyncio.sleep(0.1)
                if self.is_voice_mode_active:
                    self.state_manager.transition_to(AppState.LISTENING)
                else:
                    self.state_manager.transition_to(AppState.IDLE)
        
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            self.state_manager.handle_error(str(e))
            self._current_user_prompt = None
            self._awaiting_confirmation = False
            self._pending_command = None
            self._initial_intent = None
            
            # Return to listening/idle
            await asyncio.sleep(2)
            if self.is_voice_mode_active:
                self.state_manager.transition_to(AppState.LISTENING)
            else:
                self.state_manager.transition_to(AppState.IDLE)
    
    async def _process_confirmation(self, audio_data: bytes):
        """
        Handle a voice yes/no response while _awaiting_confirmation is True.

        yes / yeah / yep          → log verbal feedback, execute pending command
        no / nope / cancel / etc. → log verbal feedback, discard, back to LISTENING
        anything else             → treat as a brand-new command
        """
        self.state_manager.transition_to(AppState.PROCESSING, audio_data=audio_data)

        try:
            transcript = await self.transcribe_audio(audio_data)
            self._user_transcript = transcript
            word = transcript.strip().lower().rstrip(".,!?")

            if word in ("yes", "yeah", "yep"):
                self.feedback_store.log_feedback(
                    command_id=self._pending_command or "unknown",
                    feedback_type="confirmation",
                    value="yes",
                    command_text=self._pending_command,
                    source="verbal",
                )
                logger.info(f"Verbal YES — executing: '{self._pending_command}'")
                await self._execute_confirmed_command()

            elif word in ("no", "nope", "cancel", "nevermind", "never mind", "stop"):
                self.feedback_store.log_feedback(
                    command_id=self._pending_command or "unknown",
                    feedback_type="confirmation",
                    value="no",
                    command_text=self._pending_command,
                    source="verbal",
                )
                logger.info(f"Verbal NO — cancelling: '{self._pending_command}'")
                await self._cancel_confirmed_command()

            else:
                # Ambiguous — not a yes/no. Treat as a new command.
                logger.info(f"Ambiguous reply '{transcript}' — treating as new command")
                self._reset_confirmation_gate()
                await self._process_new_command(audio_data)

        except Exception as e:
            logger.error(f"Error in _process_confirmation: {e}")
            self._reset_confirmation_gate()
            self.state_manager.handle_error(str(e))
            await asyncio.sleep(2)
            target = AppState.LISTENING if self.is_voice_mode_active else AppState.IDLE
            self.state_manager.transition_to(target)

    def _reset_confirmation_gate(self):
        """Clear all confirmation state without triggering any transitions."""
        self._awaiting_confirmation = False
        self._pending_command = None
        self._pending_audio_turns = []
        self._pending_transcript_turns = []
        self._current_user_prompt = None
        self._conversation_context = []
        self._initial_intent = None  # Reset session root transcript
        
    async def _execute_confirmed_command(self):
        """Execute the pending command and return to LISTENING."""
        pending = self._pending_command
        initial_intent = self._initial_intent
        pending_audio_turns = list(self._pending_audio_turns)        # all turns
        pending_transcript_turns = list(self._pending_transcript_turns)
        logger.info(f"[CONFIRMATION] Executing confirmed command: {pending}")
        self._last_action = "confirmed"
        self._reset_confirmation_gate()

        # Cache confirmed command
        if self.command_orchestrator and initial_intent and pending:
            self.command_orchestrator.cache.set(initial_intent, pending)
            logger.info(
                "Cache WRITE (post-confirmation): '%s' -> '%s'",
                initial_intent, pending,
            )

        if self.shortcut_manager.is_recording() and pending:
            self.shortcut_manager.add_recorded_command(pending)
            logger.info("Shortcut buffer append: '%s'", pending)

        self.state_manager.transition_to(AppState.EXECUTING, transcript=pending)
        try:
            await self._execute_browser_command(pending)
        except Exception as e:
            logger.error(f"Execution error after confirmation: {e}")
            self.state_manager.handle_error(str(e))

        # Save all confirmed audio turns for training (only when toggle is on)
        if pending_audio_turns:
            total_duration_s = 0.0
            try:
                prefs = self.profile_manager.load_preferences(DEFAULT_PROFILE_ID)
                if prefs.get("custom_training_enabled", False):
                    for audio, transcript in zip(pending_audio_turns, pending_transcript_turns):
                        duration_s = training_data_recorder.save_sample(
                            audio, transcript
                        )
                        total_duration_s += duration_s
                        logger.info(
                            "Training sample saved: %.2fs | '%s'",
                            duration_s, transcript[:60],
                        )

                    if total_duration_s:
                        current = prefs.get("accumulated_audio_seconds", 0)
                        self.profile_manager.set_training_state(
                            accumulated_audio_seconds=current + int(total_duration_s),
                            profile_id=DEFAULT_PROFILE_ID,
                        )
                        logger.info(
                            "Total training audio this session: %.2fs | running total: %.0fs",
                            total_duration_s, current + total_duration_s,
                        )
            except Exception as e:
                logger.warning("Training sample save failed (non-fatal): %s", e)

        self._last_command = pending
        await asyncio.sleep(0.1)
        target = AppState.LISTENING if self.is_voice_mode_active else AppState.IDLE
        self.state_manager.transition_to(target)
        logger.info(f"[CONFIRMATION] Command executed, state reset to {target}")

        if hasattr(self, 'manager') and self.manager:
            await self.manager.broadcast({"state": self.state_manager.get_state_info()})

    async def _cancel_confirmed_command(self):
        self._last_action = "cancelled"  
        self._reset_confirmation_gate()
        await asyncio.sleep(0.1)
        target = AppState.LISTENING if self.is_voice_mode_active else AppState.IDLE
        self.state_manager.transition_to(target)
        logger.info(f"[CONFIRMATION] Command cancelled, state reset to {target}")
        if hasattr(self, 'manager') and self.manager:
            await self.manager.broadcast({"state": self.state_manager.get_state_info()})

    # ========================================================================
    # Audio Capture Management
    # ========================================================================
    
    async def start_audio_capture(self):
        """Start audio capture with VAD."""
        global _capture_instance
        RealtimeAudioCapture, AudioConfig, import_err = _get_audio_capture()
        if import_err:
            raise RuntimeError(f"Audio capture unavailable: {import_err}")

        
        with _capture_lock:
            if _capture_instance is not None:
                logger.info("Audio capture already running")
                return
            
            config = AudioConfig(sample_rate=16000, vad_aggressiveness=3)
            _capture_instance = RealtimeAudioCapture(
                config=config,
                on_audio_ready=_on_audio_ready
            )
            _capture_instance.start()
        
        logger.info("Audio capture started")
    
    async def stop_audio_capture(self):
        """Stop audio capture."""
        global _capture_instance
        with _capture_lock:
            if _capture_instance is None:
                return
            try:
                _capture_instance.stop()
            except Exception as e:
                logger.error(f"Error stopping audio capture: {e}")
            finally:
                _capture_instance = None
        
        logger.info("Audio capture stopped")
    
    # ========================================================================
    # Public API Methods
    # ========================================================================
    
    async def start_voice_mode(self) -> dict:
        """Start voice mode."""
        current_state = self.state_manager.current_state
        
        if current_state != AppState.IDLE:
            return {
                "success": False,
                "error": f"Cannot start from state: {current_state.value}"
            }
        
        self.is_voice_mode_active = True
        self.state_manager.transition_to(AppState.LISTENING)
        await self.start_audio_capture()
        
        return {"success": True, "state": "listening"}
    
    async def stop_voice_mode(self) -> dict:
        """Stop voice mode."""
        self.is_voice_mode_active = False
        await self.stop_audio_capture()
        self.state_manager.transition_to(AppState.IDLE)
        
        return {"success": True, "state": "idle"}
    
    def get_status(self) -> dict:
        """Get current server status."""
        status = self.state_manager.get_state_info()
        status['voice_mode_active'] = self.is_voice_mode_active
        status['last_transcript'] = self._last_transcript
        status['last_command'] = self._last_command
        status['user_prompt'] = self._current_user_prompt
        status['transcript'] = self._last_transcript or status.get('transcript')
        status['user_transcript'] = self._user_transcript
        status['clarified_command'] = self._clarified_command
        status['has_conversation_context'] = len(self._conversation_context) > 0
        status['awaiting_confirmation'] = self._awaiting_confirmation
        status['pending_command'] = self._pending_command
        status['last_action'] = self._last_action
        if self.command_orchestrator:
            status['cache_stats'] = self.command_orchestrator.get_cache_stats()
        return status
    
    def reset(self) -> dict:
        self.is_voice_mode_active = False
        self._current_user_prompt = None
        self._conversation_context = []
        self._user_transcript = None
        self._clarified_command = None
        self._initial_intent = None
        if self.command_orchestrator:
            # Reset conversation state; persistent cache remains intact.
            self.command_orchestrator.reset()
            logger.info(
                "Orchestrator reset. Cache preserved (%d entries).",
                self.command_orchestrator.cache.size,
            )
        self.state_manager.reset()
        return {"success": True, "state": "idle"}


# ============================================================================
# FastAPI Application Setup
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle."""
    global server, _event_loop
    
    # Startup
    _event_loop = asyncio.get_event_loop()
    server = VoiceBridgeServer()
    logger.info("VoiceBridge Server initialized")
    
    yield
    
    # Shutdown
    if server:
        await server.stop_voice_mode()
    logger.info("VoiceBridge Server shutdown")


app = FastAPI(
    title="VoiceBridge API",
    description="Voice-controlled browser automation",
    version="2.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

server: Optional[VoiceBridgeServer] = None


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "VoiceBridge API",
        "version": "2.0.0",
        "status": "online"
    }


@app.get("/status")
async def get_status():
    """Get current status"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    status_data = server.get_status()
    return StatusResponse(**status_data)

@app.get("/audio/capture/status")
async def audio_capture_status():
    """Get status of backend audio capture."""
    with _capture_lock:
        if _capture_instance is None:
            return {"capturing": False, "state": "idle"}
        return {
            "capturing": True,
            "state": _capture_instance.state,
            "stats": _capture_instance.get_stats(),
        }


@app.post("/audio/capture/start")
async def start_voice_mode_endpoint():
    """Start voice mode"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = await server.start_voice_mode()
    
    if result["success"]:
        return {"ok": True, "message": "Voice mode started", "state": result["state"]}
    else:
        return {"ok": False, "message": result.get("error", "Failed to start")}

@app.post("/audio/capture/stop")
async def stop_voice_mode_endpoint():
    """Stop voice mode"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = await server.stop_voice_mode()
    return {"ok": True, "message": "Voice mode stopped", "state": result["state"]}


@app.post("/reset")
async def reset_server():
    """Reset server to idle state"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = server.reset()
    return {"success": True, "message": "Server reset", "state": result["state"]}


@app.post("/feedback")
async def feedback_endpoint(request: Request):
    """
    Receive UI or verbal feedback for a pending confirmation.

    thumbs_up  → log + execute the pending command
    thumbs_down → log + cancel the pending command
    Other values → log only (no state change)
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    data = await request.json()
    try:
        value = data.get("value", "")
        source = data.get("source", "ui")
        command_id = data.get("command_id") or server._pending_command or "unknown"
        command_text = data.get("command_text") or server._pending_command

        server.feedback_store.log_feedback(
            command_id=command_id,
            feedback_type=data.get("feedback_type", "ui"),
            value=value,
            command_text=command_text,
            source=source,
        )

        if server._awaiting_confirmation:
            if value in ("thumbs_up", "yes"):
                logger.info(f"UI confirm — executing: '{server._pending_command}'")
                asyncio.create_task(server._execute_confirmed_command())
            elif value in ("thumbs_down", "no"):
                logger.info(f"UI cancel — discarding: '{server._pending_command}'")
                asyncio.create_task(server._cancel_confirmed_command())

        return {"success": True}
    except Exception as e:
        logger.error(f"Feedback logging failed: {e}")
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    """Health check"""
    if not server:
        return {"status": "unhealthy", "reason": "Server not initialized"}
    
    return {
        "status": "healthy",
        "state": server.state_manager.current_state.value,
        "voice_mode_active": server.is_voice_mode_active
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    if not server:
        await websocket.close(code=1011, reason="Server not initialized")
        return
    
    await server.connection_manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            'type': 'connected',
            'state': server.state_manager.current_state.value,
            'voice_mode_active': server.is_voice_mode_active,
            'data': server.get_status()
        })
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
    
    except WebSocketDisconnect:
        server.connection_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        server.connection_manager.disconnect(websocket)


@app.get("/cache/stats")
async def cache_stats():
    """Return command cache metadata."""
    if not server or not server.command_orchestrator:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return server.command_orchestrator.get_cache_stats()


@app.delete("/cache/entry")
async def cache_invalidate_entry(
    transcript: str = Query(..., description="The transcript to remove from cache")
):
    """
    Remove a cached entry for the given transcript.
    """
    if not server or not server.command_orchestrator:
        raise HTTPException(status_code=503, detail="Server not initialized")
    removed = server.command_orchestrator.invalidate_cached_command(transcript)
    return {"removed": removed, "transcript": transcript}


@app.delete("/cache/all")
async def cache_clear_all():
    """
    Clear all cached commands and delete the backing file.
    """
    if not server or not server.command_orchestrator:
        raise HTTPException(status_code=503, detail="Server not initialized")
    count = server.command_orchestrator.clear_cache()
    return {"cleared": True, "entries_removed": count}


# Voice Customisation Endpoints
def _prefs_response(prefs: dict) -> dict:
    """Return the subset of preference fields exposed to the client."""
    return {
        "guardrails_enabled":        prefs.get("guardrails_enabled",        True),
        "custom_training_enabled":   prefs.get("custom_training_enabled",   False),
        "custom_shortcuts_enabled":  prefs.get("custom_shortcuts_enabled",  False),
        "training_in_progress":      prefs.get("training_in_progress",      False),
        "training_completed":        prefs.get("training_completed",        False),
    }


@app.get("/preferences")
async def get_preferences():
    """Get user preferences."""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    try:
        prefs = server.profile_manager.load_preferences(DEFAULT_PROFILE_ID)
        return _prefs_response(prefs)
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")


@app.post("/preferences")
async def update_preferences(body: PreferencesUpdate):
    """
    Update user-settable preferences.

    When custom_training_enabled changes, a consent event is recorded
    automatically so the audit log stays consistent without requiring a
    separate client call.
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")

    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(status_code=422, detail="No valid fields provided")

    try:
        # Record consent change whenever training toggle changes.
        if "custom_training_enabled" in updates:
            server.profile_manager.save_consent(
                DEFAULT_PROFILE_ID,
                consent_flag=updates["custom_training_enabled"],
            )

        prefs = server.profile_manager.update_preferences(updates, DEFAULT_PROFILE_ID)
        return _prefs_response(prefs)
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")


@app.get("/training/status")
async def get_training_status():
    """Return current training state and data-collection progress."""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    try:
        prefs = server.profile_manager.load_preferences(DEFAULT_PROFILE_ID)
        return {
            "training_in_progress":      prefs.get("training_in_progress",      False),
            "training_completed":        prefs.get("training_completed",        False),
            "accumulated_audio_seconds": prefs.get("accumulated_audio_seconds", 0),
        }
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")


@app.post("/training/start")
async def start_training():
    """
    Launch a background voice-model training job.

    400 — training not enabled (custom_training_enabled is False)
    409 — a job is already running or has completed
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")

    try:
        prefs = server.profile_manager.load_preferences(DEFAULT_PROFILE_ID)
    except ProfileNotFoundError:
        raise HTTPException(status_code=404, detail="Profile not found")

    if not prefs.get("custom_training_enabled", False):
        raise HTTPException(
            status_code=400,
            detail="Training is not enabled. Turn on 'Collect Training Data' first.",
        )
    if prefs.get("training_in_progress", False):
        raise HTTPException(status_code=409, detail="A training job is already running.")
    if prefs.get("training_completed", False):
        raise HTTPException(status_code=409, detail="Training has already completed.")

    # Mark job as started.
    # TODO: set_training_state() as it progresses.
    server.profile_manager.set_training_state(
        training_in_progress=True,
        profile_id=DEFAULT_PROFILE_ID,
    )
    logger.info("Voice model training job started.")
    return {"started": True}

# Shortcuts Endpoints and Functions
def create_shortcut(self, name: str, commands: list[str]) -> dict: #manually create a shortcut outside of the recording flow
    with self._lock:
        shortcut_id = str(self._next_id())
        shortcut = {
            "id": shortcut_id,
            "name": name.strip(),
            "commands": commands,
        }
        self._shortcuts[shortcut_id] = shortcut
        self._flush()
        return shortcut
    
@app.get("/shortcuts")
async def get_shortcuts():
    """Get all custom shortcuts."""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    try:
        prefs = server.profile_manager.load_preferences(DEFAULT_PROFILE_ID)
    except ProfileNotFoundError:
        prefs = {}
    return {
        "shortcuts": server.shortcut_manager.list_shortcuts(),
        "custom_shortcuts_enabled": prefs.get("custom_shortcuts_enabled", False),
        "recording": server.shortcut_manager.is_recording(),
    }


@app.post("/shortcuts")
async def create_shortcut(body: ShortcutCreate):
    """Add a shortcut manually."""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    if not body.phrase.strip() or not body.command.strip():
        raise HTTPException(status_code=422, detail="phrase and command must not be empty")

    shortcut = server.shortcut_manager.create_shortcut(
        name=body.phrase.strip(),
        commands=[body.command.strip()],
    )
    return shortcut


@app.delete("/shortcuts/{shortcut_id}")
async def delete_shortcut(shortcut_id: str):
    """Delete a shortcut by id."""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    deleted = server.shortcut_manager.delete_shortcut(shortcut_id)
    return {"id": shortcut_id, "deleted": deleted}


@app.post("/shortcuts/{shortcut_id}/run")
async def run_shortcut(shortcut_id: str):
    """Run all commands of a shortcut by id."""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    try:
        result = await server.run_shortcut(shortcut_id)
        return result
    except HTTPException:
        raise


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )