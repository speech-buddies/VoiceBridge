"""
FastAPI Server Entry Point

Minimal API for frontend to access VoiceBridge backend.
The backend handles the entire flow automatically:
  1. VAD detects voice → starts recording
  2. VAD detects silence → processes audio
  3. Transcribes → parses command → executes in browser
  4. Returns to idle, waiting for next voice input

Frontend just needs to:
  - Start/stop the voice listening mode
  - Display current state via WebSocket updates
  - Handle manual commands (optional)
"""

import asyncio
import logging
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
from src.control.browser_orchestrator import run_command
from src.control.session_manager import get_browser, start_session, stop_session

# Windows compatibility for browser asyncio event loop
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Lazy import: audio_capture needs sounddevice + webrtcvad
def _get_audio_capture():
    try:
        from input.audio_capture import RealtimeAudioCapture, AudioConfig
        return RealtimeAudioCapture, AudioConfig, None
    except ImportError as e:
        return None, None, str(e)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Folder where audio recordings are saved (standalone capture + legacy upload)
SAVE_DIR = Path(__file__).resolve().parent / "Recordings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

_capture_instance = None
_capture_lock = threading.Lock()
_event_loop = None


def _on_audio_ready(audio_float32: np.ndarray, metadata: dict):
    """
    Callback when RealtimeAudioCapture has finished a recording.
    If voice mode is active: feed audio into pipeline via on_silence_detected.
    Otherwise (standalone capture): save to Recordings/.
    """
    global server, _event_loop
    if server and server.is_voice_mode_active and _event_loop:
        # Feed into VoiceBridge pipeline
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
# Pydantic Models for API
# ============================================================================

class StatusResponse(BaseModel):
    """Current status response"""
    state: str
    timestamp: str
    has_audio: bool
    has_transcript: bool
    error: Optional[str]
    transcript: Optional[str] = None
    last_command: Optional[str] = None
    user_prompt: Optional[str] = None 


class ManualCommandRequest(BaseModel):
    """Manual command input (for testing/debugging)"""
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
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)


# ============================================================================
# VoiceBridge Server
# ============================================================================

class VoiceBridgeServer:
    """
    Main server that runs the voice automation pipeline.
    
    Flow:
      IDLE → (VAD detects voice) → RECORDING → (VAD detects silence) → 
      PROCESSING → EXECUTING → IDLE
    """
    
    def __init__(self):
        self.state_manager = StateManager()
        self.connection_manager = ConnectionManager()
        self.is_voice_mode_active = False
        self._browser_session_started = False # Tracks if browser session is active
        self._session_lock = asyncio.Lock() # To prevent multiple sessions from starting/stopping at same time
        
        # TODO: Initialize our modules
        # self.audio_capture = AudioCapture()
        # self.speech_to_text = SpeechToText()
        # self.command_orchestrator = CommandOrchestrator()
        # self.browser_controller = BrowserController()
        
        # Register callback to broadcast state changes to WebSocket clients
        self.state_manager.register_callback(None, self._broadcast_state_change)
        self.state_manager.register_callback(None, self._on_state_change) # also sync browser session with state changes
        
        # Store last transcript and command for status endpoint
        self._last_transcript: Optional[str] = None
        self._last_command: Optional[str] = None
        
        # Conversation context for multi-turn interactions
        self._conversation_context: List[dict] = []
        self._current_user_prompt: Optional[str] = None
    
    def _broadcast_state_change(self, state_data, old_state):
        """Broadcast state changes to all WebSocket clients"""
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
        
        logger.info(f"State: {old_state.value} → {state_data.state.value}")
        if self._current_user_prompt:
            logger.info(f"User prompt: {self._current_user_prompt}")
        
        asyncio.create_task(self.connection_manager.broadcast(message))

    def _on_state_change(self, state_data, old_state):
        """
        Called by StateManager on every transition.
        We fan out to:
          1) broadcast to websocket clients
          2) manage browser session automatically based on AppState
        """
        # broadcast 
        self._broadcast_state_change(state_data, old_state)

        # session control based on state
        asyncio.create_task(self._sync_browser_session_with_state(state_data.state))

    # ========================================================================
    # Session Synchronization Logic
    # ========================================================================
    async def _sync_browser_session_with_state(self, new_state: AppState):
        """
        Auto start/stop browser session based on AppState.
        LISTENING => ensure browser session is started
        STOP      => ensure browser session is stopped
        """
        async with self._session_lock:
            # START when we begin listening
            if new_state == AppState.LISTENING:
                if not self._browser_session_started:
                    msg = await start_session()
                    logger.info(f"Browser session started: {msg}")
                    self._browser_session_started = True
                return

            # STOP when we go idle (or after reset / stop voice mode)
            if new_state == AppState.IDLE:
                if self._browser_session_started:
                    msg = await stop_session()
                    logger.info(f"Browser session stopped: {msg}")
                    self._browser_session_started = False
                return
    
    
    # ========================================================================
    # Module Integration - Implement these with our actual modules
    # ========================================================================
    
    async def start_audio_capture(self):
        """
        Start audio capture with VAD using RealtimeAudioCapture.
        When silence is detected, _on_audio_ready feeds audio into on_silence_detected.
        """
        global _capture_instance
        RealtimeAudioCapture, AudioConfig, import_err = _get_audio_capture()
        if import_err:
            raise RuntimeError(
                f"Audio capture unavailable: {import_err}. "
                "Run: pip install -r requirements.txt (sounddevice, webrtcvad)"
            )
        with _capture_lock:
            if _capture_instance is not None:
                logger.info("Audio capture already running")
                return
            config = AudioConfig(sample_rate=16000, vad_aggressiveness=3)
            _capture_instance = RealtimeAudioCapture(
                config=config, on_audio_ready=_on_audio_ready
            )
            _capture_instance.start()
        logger.info("Audio capture started with VAD")
    
    async def stop_audio_capture(self):
        """Stop audio capture."""
        global _capture_instance
        with _capture_lock:
            if _capture_instance is None:
                logger.info("Audio capture not running")
                return
            try:
                _capture_instance.stop()
            except Exception as e:
                logger.error(f"Error stopping audio capture: {e}")
            finally:
                _capture_instance = None
        logger.info("Audio capture stopped")
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio using our speech-to-text engine.
        
        TODO: Implement with our speech-to-text module
        Example:
            return await self.speech_to_text.transcribe(audio_data)
        """
        logger.info(f"Transcribing {len(audio_data)} bytes of audio...")
        
        # our transcription code here
        # Placeholder for testing
        await asyncio.sleep(0.5)
        return "example transcript"
    
    async def parse_and_execute_command(self, transcript: str, user_context: Optional[List[dict]] = None) -> dict:
        """
        Parse transcript and execute browser command.
        Can return a prompt if LLM needs clarification.
        
        TODO: Implement with our command orchestrator and browser controller
        Example:
            result = await self.command_orchestrator.parse(
                transcript, 
                context=self._conversation_context
            )
            
            # If orchestrator needs clarification
            if result.get('needs_clarification'):
                return {
                    "needs_input": True,
                    "user_prompt": result['user_prompt'],
                    "context": result.get('context')
                }
            
            # Otherwise execute command
            command = result['command']
            execution_result = await self.browser_controller.execute(command)
            
            return {
                "needs_input": False,
                "success": True,
                "action": command['action'],
                "details": execution_result
            }
        """
        logger.info(f"Parsing and executing: {transcript}")

        # BROWSER EXECUTION PLACEHOLDER START
        transcript = transcript.strip()
        if not transcript:
            raise HTTPException(status_code=400, detail="transcript cannot be empty")
        browser = get_browser() # retrieving active browser session
        if browser is None:
            self.state_manager.transition_to(AppState.ERROR, error="No active browser session")
            raise HTTPException(
                status_code=409,
                detail="No active browser session (expected LISTENING state to start it)."
            )
        
        # Transition to executing command
        self.state_manager.transition_to(AppState.EXECUTING, transcript=transcript)
        history = await run_command(transcript, browser) # transcript is the text command being sent to the browser by command orchestrator

        try:
            history = await run_command(
                transcript, browser)  # transcript is the text command being sent to the browser by command orchestrator

            return {
                "needs_input": False,
                "success": True,
                "action": "browser_orchestrator",
                "details": str(history),
            }

        except Exception as e:
            # ERROR if orchestrator fails
            self.state_manager.transition_to(AppState.ERROR, error=str(e), transcript=transcript)
            raise HTTPException(status_code=500, detail=f"Browser orchestrator error: {e}")
        
        # BROWSER EXECUTION PLACEHOLDER END
        
        # Our command parsing and execution here
        # This is a placeholder showing the two possible outcomes
        
        # Example 1: Command is clear, execute it
        await asyncio.sleep(0.3)
        return {
            "needs_input": False,
            "success": True,
            "action": "navigate",
            "details": f"Executed command from: {transcript}"
        }
        
        # Example 2: Need clarification (uncomment to test)
        # return {
        #     "needs_input": True,
        #     "prompt": "Which Google product did you want? Google.com, Gmail, or Google Drive?",
        #     "context": {"ambiguous_command": transcript}
        # }
    
    # ========================================================================
    # Callbacks from Audio Capture Module
    # ========================================================================
    
    def on_voice_detected(self):
        """
        Called by Our audio capture module when VAD detects voice.
        Transitions from LISTENING → RECORDING
        """
        logger.info("Voice detected by VAD")
        
        if self.state_manager.current_state == AppState.LISTENING:
            self.state_manager.transition_to(AppState.RECORDING)
    
    async def on_silence_detected(self, audio_data: bytes):
        """
        Called by our audio capture module when VAD detects silence.
        Transitions from RECORDING → PROCESSING and processes the audio.
        
        Args:
            audio_data: The recorded audio bytes
        """
        logger.info(f"Silence detected by VAD, processing {len(audio_data)} bytes")
        
        current_state = self.state_manager.current_state
        
        # Handle different states
        if current_state == AppState.RECORDING:
            # Normal flow: user spoke a new command
            await self._process_new_command(audio_data)
            
        elif current_state == AppState.AWAITING_INPUT:
            # User is responding to a prompt
            await self._process_user_response(audio_data)
            
        else:
            logger.warning(f"Received audio but in unexpected state: {current_state.value}")
    
    async def _process_new_command(self, audio_data: bytes):
        """Process a new voice command from the user"""
        # Transition to processing
        self.state_manager.transition_to(AppState.PROCESSING, audio_data=audio_data)
        
        try:
            # Transcribe the audio
            transcript = await self.transcribe_audio(audio_data)
            self._last_transcript = transcript
            
            # Add to conversation context
            self._conversation_context.append({
                "role": "user",
                "content": transcript,
                "timestamp": self.state_manager.state_data.timestamp.isoformat()
            })
            
            # Parse and potentially execute the command
            result = await self.parse_and_execute_command(transcript, self._conversation_context)
            
            # Check if orchestrator needs clarification
            if result.get("needs_input"):
                # Transition to awaiting input
                self._current_user_prompt = result["user_prompt"]
                self.state_manager.transition_to(
                    AppState.AWAITING_INPUT,
                    transcript=transcript,
                    metadata={"user_prompt": result["user_prompt"], "context": result.get("context")}
                )
                
                # Add user_prompt to conversation context
                self._conversation_context.append({
                    "role": "assistant",
                    "content": result["user_prompt"],
                    "timestamp": self.state_manager.state_data.timestamp.isoformat()
                })
                
                # Automatically start listening for user's response
                if self.is_voice_mode_active:
                    await asyncio.sleep(0.1)  # Small delay
                    self.state_manager.transition_to(AppState.LISTENING)
                
            else:
                # Command executed successfully
                self._last_command = transcript
                self._current_user_prompt = None
                
                # Clear conversation context after successful execution
                self._conversation_context = []
                
                # Transition to executing
                self.state_manager.transition_to(AppState.EXECUTING, transcript=transcript)
                
                # Back to listening (if voice mode still active) or idle
                await asyncio.sleep(0.1)  # Small delay to let execution finish
                if self.is_voice_mode_active:
                    self.state_manager.transition_to(AppState.LISTENING)
                else:
                    self.state_manager.transition_to(AppState.IDLE)
                    
        except Exception as e:
            logger.error(f"Error in voice processing pipeline: {e}")
            self.state_manager.handle_error(str(e))
            self._current_user_prompt = None
            
            # Return to listening or idle after error
            await asyncio.sleep(2)
            if self.is_voice_mode_active:
                self.state_manager.transition_to(AppState.LISTENING)
            else:
                self.state_manager.transition_to(AppState.IDLE)
    
    async def _process_user_response(self, audio_data: bytes):
        """Process user's response to a clarification user prompt"""
        # Transition to processing
        self.state_manager.transition_to(AppState.PROCESSING, audio_data=audio_data)
        
        try:
            # Transcribe the response
            response = await self.transcribe_audio(audio_data)
            self._last_transcript = response
            
            # Add response to conversation context
            self._conversation_context.append({
                "role": "user",
                "content": response,
                "timestamp": self.state_manager.state_data.timestamp.isoformat()
            })
            
            # Parse and execute with full context
            result = await self.parse_and_execute_command(response, self._conversation_context)
            
            # Check if still needs more clarification
            if result.get("needs_input"):
                # Still need more info
                self._current_user_prompt = result["user_prompt"]
                self.state_manager.transition_to(
                    AppState.AWAITING_INPUT,
                    transcript=response,
                    metadata={"user_prompt": result["user_prompt"], "context": result.get("context")}
                )
                
                # Add new user prompt to conversation context
                self._conversation_context.append({
                    "role": "assistant",
                    "content": result["user_prompt"],
                    "timestamp": self.state_manager.state_data.timestamp.isoformat()
                })
                
                # Automatically start listening for next response
                if self.is_voice_mode_active:
                    await asyncio.sleep(0.1)
                    self.state_manager.transition_to(AppState.LISTENING)
                    
            else:
                # Got all the info needed, execute command
                self._last_command = response
                self._current_user_prompt = None
                
                # Clear conversation context after successful execution
                self._conversation_context = []
                
                # Transition to executing
                self.state_manager.transition_to(AppState.EXECUTING, transcript=response)
                
                # Back to listening or idle
                await asyncio.sleep(0.1)
                if self.is_voice_mode_active:
                    self.state_manager.transition_to(AppState.LISTENING)
                else:
                    self.state_manager.transition_to(AppState.IDLE)
                    
        except Exception as e:
            logger.error(f"Error processing user response: {e}")
            self.state_manager.handle_error(str(e))
            self._current_user_prompt = None
            self._conversation_context = []
            
            # Return to listening or idle after error
            await asyncio.sleep(2)
            if self.is_voice_mode_active:
                self.state_manager.transition_to(AppState.LISTENING)
            else:
                self.state_manager.transition_to(AppState.IDLE)
    # ========================================================================
    # Public API Methods
    # ========================================================================
    
    async def start_voice_mode(self) -> dict:
        """
        Start voice mode - app will continuously listen for voice commands.
        Flow: IDLE → LISTENING → (automatic from here based on VAD)
        """
        current_state = self.state_manager.current_state
        
        if current_state != AppState.IDLE:
            return {
                "success": False,
                "error": f"Cannot start voice mode from state: {current_state.value}"
            }
        
        self.is_voice_mode_active = True
        
        # Transition to listening
        self.state_manager.transition_to(AppState.LISTENING)
        
        # Start audio capture (which includes VAD)
        await self.start_audio_capture()
        
        return {"success": True, "state": "listening"}
    
    async def stop_voice_mode(self) -> dict:
        """
        Stop voice mode - return to idle and stop listening.
        """
        self.is_voice_mode_active = False
        
        # Stop audio capture
        await self.stop_audio_capture()
        
        # Return to idle
        self.state_manager.transition_to(AppState.IDLE)
        
        return {"success": True, "state": "idle"}
    
    async def execute_manual_command(self, command: str, metadata: Optional[dict] = None) -> dict:
        """
        Execute a manual text command (for testing/debugging).
        Bypasses voice input but uses the same command orchestrator.
        """
        current_state = self.state_manager.current_state
        
        if current_state != AppState.IDLE:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot execute manual command in state: {current_state.value}. Stop voice mode first."
            )
        
        # Transition to executing
        self.state_manager.transition_to(AppState.EXECUTING, transcript=command, metadata=metadata)
        
        try:
            # Parse and execute the command
            result = await self.parse_and_execute_command(command)
            self._last_transcript = command
            self._last_command = command
            
            # Return to idle
            self.state_manager.transition_to(AppState.IDLE)
            
            return {
                "success": True,
                "command": command,
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Error executing manual command: {e}")
            self.state_manager.handle_error(str(e))
            raise HTTPException(status_code=500, detail=str(e))
    def get_status(self) -> dict:
        """Get current server status"""
        status = self.state_manager.get_state_info()
        status['voice_mode_active'] = self.is_voice_mode_active
        status['last_transcript'] = self._last_transcript
        status['last_command'] = self._last_command
        status['user_prompt'] = self._current_user_prompt
        status['has_conversation_context'] = len(self._conversation_context) > 0
        return status
    
    def reset(self) -> dict:
        """Reset to idle state"""
        self.is_voice_mode_active = False
        self._current_user_prompt = None
        self._conversation_context = []
        self.state_manager.reset()
        return {"success": True, "state": "idle"}
    
    async def submit_user_input(self, user_input: str, metadata: Optional[dict] = None) -> dict:
        """
        Submit user input in response to a prompt (for manual text input).
        Only works when in AWAITING_INPUT state.
        
        This is useful if user wants to type their response instead of speaking.
        """
        current_state = self.state_manager.current_state
        
        if current_state != AppState.AWAITING_INPUT:
            raise HTTPException(
                status_code=400,
                detail=f"Not awaiting input. Current state: {current_state.value}"
            )
        
        # Transition to processing
        self.state_manager.transition_to(AppState.PROCESSING, transcript=user_input, metadata=metadata)
        
        try:
            # Add input to conversation context
            self._conversation_context.append({
                "role": "user",
                "content": user_input,
                "timestamp": self.state_manager.state_data.timestamp.isoformat(),
                "source": "text_input"
            })
            
            # Parse and execute with full context
            result = await self.parse_and_execute_command(user_input, self._conversation_context)
            
            # Check if still needs more clarification
            if result.get("needs_input"):
                # Still need more info
                self._current_user_prompt = result["user_prompt"]
                self.state_manager.transition_to(
                    AppState.AWAITING_INPUT,
                    transcript=user_input,
                    metadata={"user_prompt": result["user_prompt"], "context": result.get("context")}
                )
                
                # Add new prompt to conversation context
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
                # Got all the info needed, execute command
                self._last_command = user_input
                self._current_user_prompt = None
                
                # Clear conversation context after successful execution
                self._conversation_context = []
                
                # Transition to executing
                self.state_manager.transition_to(AppState.EXECUTING, transcript=user_input)
                
                # Back to listening or idle
                await asyncio.sleep(0.1)
                if self.is_voice_mode_active:
                    self.state_manager.transition_to(AppState.LISTENING)
                else:
                    self.state_manager.transition_to(AppState.IDLE)
                
                return {
                    "success": True,
                    "needs_more_input": False,
                    "result": result
                }
                
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            self.state_manager.handle_error(str(e))
            self._current_user_prompt = None
            self._conversation_context = []
            raise HTTPException(status_code=500, detail=str(e))




# ============================================================================
# FastAPI Application
# ============================================================================

# Global server instance
server: Optional[VoiceBridgeServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global server, _event_loop
    
    # Startup
    _event_loop = asyncio.get_running_loop()
    logger.info("Starting VoiceBridge Server...")
    server = VoiceBridgeServer()
    
    yield
    
    # Shutdown
    logger.info("Shutting down VoiceBridge Server...")
    if server:
        await server.stop_audio_capture()


# Create FastAPI app
app = FastAPI(
    title="VoiceBridge API",
    description="API for VoiceBridge",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "VoiceBridge API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            # "start": "POST /voice/start",
            # "stop": "POST /voice/stop",
            "status": "GET /status",
            "manual": "POST /command/manual",
            "websocket": "WS /ws",
            "audio_capture_start": "POST /audio/capture/start",
            "audio_capture_stop": "POST /audio/capture/stop",
            "audio_capture_status": "GET /audio/capture/status",
            "audio_upload": "POST /audio",
        }
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get current application status.
    
    Returns current state, whether voice mode is active, and last transcript/command.
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    return server.get_status()


# @app.post("/voice/start")
# async def start_voice_mode():
#     """
#     Start voice mode.
    
#     The app will:
#       1. Start listening with VAD
#       2. Detect when you speak (auto transition to recording)
#       3. Detect when you stop (auto process, execute, return to listening)
#       4. Repeat until you call /voice/stop
    
#     This is the main endpoint our frontend needs to start the app.
#     """
#     if not server:
#         raise HTTPException(status_code=503, detail="Server not initialized")
    
#     result = await server.start_voice_mode()
    
#     if not result["success"]:
#         raise HTTPException(status_code=400, detail=result["error"])
    
#     return {
#         "success": True,
#         "message": "Voice mode started - listening for commands",
#         "state": result["state"]
#     }


# @app.post("/voice/stop")
# async def stop_voice_mode():
#     """
#     Stop voice mode and return to idle.
    
#     This stops the VAD and audio capture.
#     """
#     if not server:
#         raise HTTPException(status_code=503, detail="Server not initialized")
    
#     result = await server.stop_voice_mode()
    
#     return {
#         "success": True,
#         "message": "Voice mode stopped",
#         "state": result["state"]
#     }


@app.post("/command/manual", response_model=CommandResponse)
async def execute_manual_command(request: ManualCommandRequest):
    """
    Execute a manual text command (for testing/debugging).
    
    Useful for:
      - Testing commands without speaking
      - Debugging the command orchestrator
      - Manual control from text input
    
    Note: Voice mode must be stopped (app in IDLE state) to use this.
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = await server.execute_manual_command(request.command, request.metadata)
    
    return CommandResponse(
        success=True,
        message="Manual command executed",
        command=request.command,
        result=result.get("result")
    )




@app.post("/input/submit")
async def submit_user_input(request: UserInputRequest):
    """
    Submit user input in response to an orchestrator prompt.
    
    Use this when:
      - App is in AWAITING_INPUT state
      - Command orchestrator has asked a clarifying question
      - User wants to type their response instead of speaking
    
    The response will be processed with full conversation context,
    and may result in:
      - Another prompt (if more clarification needed)
      - Command execution (if sufficient info gathered)
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = await server.submit_user_input(request.input, request.metadata)
    
    if result.get("needs_more_input"):
        return {
            "success": True,
            "needs_more_input": True,
            "user_prompt": result["user_prompt"],
            "message": "Orchestrator needs more information"
        }
    else:
        return {
            "success": True,
            "needs_more_input": False,
            "message": "Command executed successfully",
            "result": result.get("result")
        }

@app.post("/reset")
async def reset():
    """
    Emergency reset - return to idle state.
    
    Use this if the app gets stuck in a state.
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = server.reset()
    
    return {
        "success": True,
        "message": "Server reset to idle state",
        "state": result["state"]
    }


# ============================================================================
# Standalone Audio Capture (from backend/server.py)
# ============================================================================

@app.post("/audio/capture/start")
async def start_audio_capture_endpoint():
    """
    Start voice mode.
    
    The app will:
      1. Start listening with VAD
      2. Detect when you speak (auto transition to recording)
      3. Detect when you stop (auto process, execute, return to listening)
      4. Repeat until you call /voice/stop
    
    This is the main endpoint our frontend needs to start the app.
    """
    global _capture_instance
    RealtimeAudioCapture, AudioConfig, import_err = _get_audio_capture()
    if import_err:
        err = str(import_err)
        if "pkg_resources" in err:
            hint = "Run: pip install setuptools"
        elif "webrtcvad" in err:
            hint = "Run: pip install webrtcvad"
        elif "sounddevice" in err:
            hint = "Run: pip install sounddevice"
        else:
            hint = "Run: pip install -r requirements.txt"
        return {"ok": False, "message": f"Audio capture unavailable: {import_err}. {hint}"}
    with _capture_lock:
        if _capture_instance is not None:
            return {"ok": True, "message": "Already capturing"}
        try:
            config = AudioConfig(sample_rate=16000, vad_aggressiveness=3)
            _capture_instance = RealtimeAudioCapture(
                config=config, on_audio_ready=_on_audio_ready
            )
            _capture_instance.start()
            return {"ok": True, "message": "Audio capture started"}
        except Exception as e:
            return {"ok": False, "message": str(e)}


@app.post("/audio/capture/stop")
async def stop_audio_capture_endpoint():
    """Stop backend-driven audio capture."""
    global _capture_instance
    with _capture_lock:
        if _capture_instance is None:
            return {"ok": True, "message": "Not capturing"}
        try:
            _capture_instance.stop()
            _capture_instance = None
            return {"ok": True, "message": "Audio capture stopped"}
        except Exception as e:
            _capture_instance = None
            return {"ok": False, "message": str(e)}


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


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not server:
        return {"status": "unhealthy", "reason": "Server not initialized"}
    
    return {
        "status": "healthy",
        "state": server.state_manager.current_state.value,
        "voice_mode_active": server.is_voice_mode_active,
        "timestamp": server.state_manager.state_data.timestamp.isoformat()
    }


# ============================================================================
# WebSocket Endpoint for Real-time Updates
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time state updates.
    
    our frontend should connect to this to receive live updates about:
      - State changes (idle → listening → recording → processing → executing → listening)
      - Transcripts as they're generated
      - Errors
    
    This is how our frontend knows what to display to the user.
    """
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
            # Receive ping/pong to keep connection alive
            # All state updates are broadcast automatically via state_manager callbacks
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