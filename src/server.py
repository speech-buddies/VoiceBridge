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
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from state_manager import StateManager, AppState
from control.browser_orchestrator import run_command
from control.session_manager import get_browser, start_session, stop_session

from app.command_orchestrator import CommandOrchestrator, OrchestratorError
from app.speech_to_text_engine import SpeechToTextEngine

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

# Recording directory
SAVE_DIR = Path(__file__).resolve().parent / "Recordings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

_capture_instance = None
_capture_lock = threading.Lock()
_event_loop = None


# ============================================================================
# Pydantic Models
# ============================================================================

class StatusResponse(BaseModel):
    """Current status response"""
    state: str
    timestamp: str
    has_audio: bool
    has_transcript: bool
    error: Optional[str]
    transcript: Optional[str] = None
    user_transcript: Optional[str] = None  # Latest user input
    clarified_command: Optional[str] = None  # LLM-clarified version
    last_command: Optional[str] = None
    user_prompt: Optional[str] = None


class ManualCommandRequest(BaseModel):
    """Manual command input"""
    command: str
    metadata: Optional[dict] = None


class UserInputRequest(BaseModel):
    """User input in response to orchestrator prompt"""
    input: str
    metadata: Optional[dict] = None


class CommandResponse(BaseModel):
    """Response from command execution"""
    success: bool
    message: str
    command: Optional[str] = None
    result: Optional[dict] = None


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
    else:
        # Standalone: save to file
        timestamp = int(time.time() * 1000)
        filename = SAVE_DIR / f"chunk-{timestamp}.wav"
        sample_rate = metadata.get("sample_rate", 16000)
        audio_int16 = (audio_float32 * 32767).astype(np.int16)
        with wave.open(str(filename), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())


# ============================================================================
# VoiceBridge Server - Improved
# ============================================================================

class VoiceBridgeServer:
    """
    Main server for voice automation pipeline.
    
    Key improvements:
    1. Clean integration with CommandOrchestrator
    2. Clear conversation flow management
    3. Better error handling
    """
    
    def __init__(self):
        self.state_manager = StateManager()
        self.connection_manager = ConnectionManager()
        self.is_voice_mode_active = False
        self._browser_session_started = False
        self._session_lock = asyncio.Lock()
        
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
        # Format: [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        self._conversation_context: List[dict] = []
        self._current_user_prompt: Optional[str] = None
    
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
    # Core Processing Methods - IMPROVED
    # ========================================================================
    
    async def parse_and_execute_command(
        self,
        transcript: str,
        conversation_context: Optional[List[dict]] = None
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
            
            # Call orchestrator
            response = await self.command_orchestrator.process(
                user_input=transcript,
                conversation_context=orchestrator_context
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
                self.state_manager.transition_to(AppState.AWAITING_INPUT, transcript=clarified_command)
                return {
                    "needs_input": True,
                    "user_prompt": guardrails_msg or "This command is not allowed. Please try something else.",
                    "context": {
                        "last_command": clarified_command
                    }
                }
            logger.info(f"Orchestrator clarified command: '{clarified_command}'")
            # Execute via browser controller (which handles natural language)
            execution_result = await self._execute_browser_command(clarified_command)
            return {
                "needs_input": False,
                "success": True,
                "command": clarified_command,
                "details": execution_result
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
        
        # Transition to executing state
        self.state_manager.transition_to(AppState.EXECUTING, transcript=clarified_command)
        
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
        
        # if current_state == AppState.RECORDING:
        
        if current_state == AppState.AWAITING_INPUT:
            # User is responding to a prompt
            await self._process_user_response(audio_data)
        
        else:
            # User spoke a new command
            await self._process_new_command(audio_data)
            # logger.warning(f"Received audio in unexpected state: {current_state.value}")
    
    async def _process_new_command(self, audio_data: bytes):
        """Process a new voice command from the user."""
        # Clear clarified command (new conversation starting)
        # But keep user_transcript - it will be updated with new input
        self._clarified_command = None
        # Transition to processing
        self.state_manager.transition_to(AppState.PROCESSING, audio_data=audio_data)
        
        try:
            # Transcribe
            transcript = await self.transcribe_audio(audio_data)
            self._last_transcript = transcript
            self._user_transcript = transcript  # Store latest user input
            
            # Add to conversation context
            self._conversation_context.append({
                "role": "user",
                "content": transcript,
                "timestamp": self.state_manager.state_data.timestamp.isoformat()
            })
            
            # Parse and execute
            result = await self.parse_and_execute_command(
                transcript,
                self._conversation_context
            )
            
            # Handle result
            if result.get("needs_input"):
                # Need clarification
                self._current_user_prompt = result["user_prompt"]
                self.state_manager.transition_to(
                    AppState.AWAITING_INPUT,
                    transcript=transcript,
                    metadata={
                        "user_prompt": result["user_prompt"],
                        "context": result.get("context", {})
                    }
                )
                
                # Add prompt to conversation
                self._conversation_context.append({
                    "role": "assistant",
                    "content": result["user_prompt"],
                    "timestamp": self.state_manager.state_data.timestamp.isoformat()
                })
                
                # Auto-start listening for response
                if self.is_voice_mode_active:
                    await asyncio.sleep(0.1)
                    # self.state_manager.transition_to(AppState.LISTENING)
            
            else:
                # Command executed successfully
                self._last_command = transcript
                self._current_user_prompt = None
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
            
            # Return to listening/idle
            await asyncio.sleep(2)
            if self.is_voice_mode_active:
                self.state_manager.transition_to(AppState.LISTENING)
            else:
                self.state_manager.transition_to(AppState.IDLE)
    
    async def _process_user_response(self, audio_data: bytes):
        """Process user's response to a clarification prompt."""
        # Transition to processing
        self.state_manager.transition_to(AppState.PROCESSING, audio_data=audio_data)
        
        try:
            # Transcribe response
            response = await self.transcribe_audio(audio_data)
            self._last_transcript = response
            self._user_transcript = response  # Update to show latest user input
            
            # Add to conversation context
            self._conversation_context.append({
                "role": "user",
                "content": response,
                "timestamp": self.state_manager.state_data.timestamp.isoformat()
            })
            
            # Parse and execute with context
            result = await self.parse_and_execute_command(
                response,
                self._conversation_context
            )
            
            # Handle result
            if result.get("needs_input"):
                # Still need more info
                self._current_user_prompt = result["user_prompt"]
                self.state_manager.transition_to(
                    AppState.AWAITING_INPUT,
                    transcript=response,
                    metadata={
                        "user_prompt": result["user_prompt"],
                        "context": result.get("context", {})
                    }
                )
                
                # Add new prompt to conversation
                self._conversation_context.append({
                    "role": "assistant",
                    "content": result["user_prompt"],
                    "timestamp": self.state_manager.state_data.timestamp.isoformat()
                })
                
                # Auto-start listening
                if self.is_voice_mode_active:
                    await asyncio.sleep(0.1)
                    self.state_manager.transition_to(AppState.LISTENING)
            
            else:
                # Got enough info, command executed
                self._last_command = response
                self._current_user_prompt = None
                
                # Clear conversation context
                self._conversation_context = []
                
                # Back to listening/idle
                await asyncio.sleep(0.1)
                if self.is_voice_mode_active:
                    self.state_manager.transition_to(AppState.LISTENING)
                else:
                    self.state_manager.transition_to(AppState.IDLE)
        
        except Exception as e:
            logger.error(f"Error processing response: {e}")
            self.state_manager.handle_error(str(e))
            self._current_user_prompt = None
            self._conversation_context = []
            
            # Return to listening/idle
            await asyncio.sleep(2)
            if self.is_voice_mode_active:
                self.state_manager.transition_to(AppState.LISTENING)
            else:
                self.state_manager.transition_to(AppState.IDLE)
    
    # ========================================================================
    # Audio Capture Management
    # ========================================================================
    
    async def start_audio_capture(self):
        """Start audio capture with VAD."""
        global _capture_instance
        RealtimeAudioCapture, AudioConfig, import_err = _get_audio_capture()
        if import_err:
            raise RuntimeError(f"Audio capture unavailable: {import_err}")
        
        self.state_manager.transition_to(AppState.RECORDING)
        
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
    
    async def execute_manual_command(self, command: str, metadata: Optional[dict] = None) -> dict:
        """Execute a manual text command."""
        current_state = self.state_manager.current_state
        
        if current_state != AppState.IDLE:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot execute in state: {current_state.value}"
            )
        
        self.state_manager.transition_to(AppState.EXECUTING, transcript=command)
        
        try:
            result = await self.parse_and_execute_command(command)
            self._last_transcript = command
            self._last_command = command
            
            self.state_manager.transition_to(AppState.IDLE)
            
            return {
                "success": True,
                "command": command,
                "result": result
            }
        
        except Exception as e:
            logger.error(f"Error executing manual command: {e}")
            self.state_manager.handle_error(str(e))
            raise
    
    async def submit_user_input(self, user_input: str, metadata: Optional[dict] = None) -> dict:
        """Submit manual text input in response to a prompt."""
        current_state = self.state_manager.current_state
        
        if current_state != AppState.AWAITING_INPUT:
            raise HTTPException(
                status_code=400,
                detail=f"Not awaiting input. State: {current_state.value}"
            )
        
        self.state_manager.transition_to(AppState.PROCESSING, transcript=user_input)
        
        try:
            # Add to conversation
            self._conversation_context.append({
                "role": "user",
                "content": user_input,
                "timestamp": self.state_manager.state_data.timestamp.isoformat(),
                "source": "text_input"
            })
            
            # Process
            result = await self.parse_and_execute_command(
                user_input,
                self._conversation_context
            )
            
            # Handle result
            if result.get("needs_input"):
                self._current_user_prompt = result["user_prompt"]
                self.state_manager.transition_to(AppState.AWAITING_INPUT)
                
                self._conversation_context.append({
                    "role": "assistant",
                    "content": result["user_prompt"],
                    "timestamp": self.state_manager.state_data.timestamp.isoformat()
                })
                
                return {
                    "success": True,
                    "needs_more_input": True,
                    "user_prompt": result["user_prompt"]
                }
            
            else:
                self._current_user_prompt = None
                self._conversation_context = []
                self.state_manager.transition_to(AppState.IDLE)
                
                return {
                    "success": True,
                    "needs_more_input": False,
                    "result": result
                }
        
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            self.state_manager.handle_error(str(e))
            raise
    
    def get_status(self) -> dict:
        """Get current server status."""
        status = self.state_manager.get_state_info()
        status['voice_mode_active'] = self.is_voice_mode_active
        status['last_transcript'] = self._last_transcript
        status['last_command'] = self._last_command
        status['user_prompt'] = self._current_user_prompt
        status['transcript'] = self._last_transcript or status.get('transcript')
        status['user_transcript'] = self._user_transcript  # Latest user input
        status['clarified_command'] = self._clarified_command  # LLM version
        status['has_conversation_context'] = len(self._conversation_context) > 0
        return status
    
    def reset(self) -> dict:
        """Reset to idle state."""
        self.is_voice_mode_active = False
        self._current_user_prompt = None
        self._conversation_context = []
        self._user_transcript = None
        self._clarified_command = None
        if self.command_orchestrator:
            self.command_orchestrator.reset()
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


@app.post("/audio")
async def receive_audio(
    audio: UploadFile = File(...),
    mimeType: str = Form("audio/webm"),
    timestamp: str = Form(...),
):
    """Legacy: receives audio blob from browser (when not using backend capture)."""
    suffix = Path(audio.filename).suffix or ".webm"
    filename = SAVE_DIR / f"chunk-{timestamp}{suffix}"
    with filename.open("wb") as f:
        f.write(await audio.read())
    return PlainTextResponse("ok")

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


@app.post("/command/execute")
async def execute_manual_command(request: ManualCommandRequest):
    """Execute a manual text command"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        result = await server.execute_manual_command(
            request.command,
            request.metadata
        )
        return CommandResponse(
            success=True,
            message="Command executed",
            command=result["command"],
            result=result.get("result")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/input/submit")
async def submit_user_input(request: UserInputRequest):
    """Submit user input in response to a prompt"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        result = await server.submit_user_input(
            request.input,
            request.metadata
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_server():
    """Reset server to idle state"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = server.reset()
    return {"success": True, "message": "Server reset", "state": result["state"]}


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