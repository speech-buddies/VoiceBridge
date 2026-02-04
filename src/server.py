"""
FastAPI Server Entry Point

Minimal API for frontend to access VoiceBridge backend.
The backend handles the entire flow automatically:
  1. VAD detects voice -> starts recording
  2. VAD detects silence -> processes audio
  3. Transcribes -> parses command -> executes in browser
  4. Returns to idle, waiting for next voice input

Frontend just needs to:
  - Start/stop the voice listening mode
  - Display current state via WebSocket updates
  - Handle manual commands (optional)
"""

import asyncio
from utils import logger 

import sys
import threading
import time
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

# Load environment variables from .env

from dotenv import load_dotenv
load_dotenv()



import numpy as np
import wave
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from state_manager import StateManager, AppState
from control.browser_orchestrator import run_command
from control.session_manager import get_browser, start_session, stop_session
from app.command_orchestrator import CommandOrchestrator
from app.speech_to_text_engine import SpeechToTextEngine

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


logger = logger.get_logger("VoiceBridgeServer")

# Folder where audio recordings are saved (standalone capture + legacy upload)
SAVE_DIR = Path(__file__).resolve().parent / "Recordings"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

_capture_instance = None
_capture_lock = threading.Lock()
_event_loop = None




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

# ========================================================================
# Callbacks from Audio Capture Module
# ========================================================================
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
# VoiceBridge Server
# ============================================================================

class VoiceBridgeServer:
    """
    Main server that runs the voice automation pipeline.
    
    Flow:
      IDLE -> (VAD detects voice) -> RECORDING -> (VAD detects silence) -> 
      PROCESSING -> EXECUTING -> IDLE
    """
    
    def __init__(self):
        self.state_manager = StateManager()
        self.connection_manager = ConnectionManager()
        self.is_voice_mode_active = False
        self._browser_session_started = False # Tracks if browser session is active
        self._session_lock = asyncio.Lock() # To prevent multiple sessions from starting/stopping at same time
        
        self.command_orchestrator = CommandOrchestrator()
        self.speech_to_text = SpeechToTextEngine()
        
        # Register callback to broadcast state changes to WebSocket clients
        self.state_manager.register_callback(None, self._broadcast_state_change)
        self.state_manager.register_callback(None, self._on_state_change) # also sync browser session with state changes
        
        # Store last transcript and command for status endpoint
        self._last_transcript: Optional[str] = None
        self._last_command: Optional[str] = None
        
        # Conversation context for multi-turn interactions
        self._conversation_context: List[dict] = []
        self._current_user_prompt: Optional[str] = None

        self._waiting_for_recovery = False  # Flag to route to recovery vs baseline

    
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
        
        logger.info(f"State: {old_state.value} -> {state_data.state.value}")
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
        
        # Transition to recording
        self.state_manager.transition_to(AppState.RECORDING)

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

        transcription = await self.speech_to_text.transcribe(audio_data)
        logger.info(f"Received transcription: {transcription}")

        return transcription
    
    async def parse_and_execute_command(self, transcript: str, user_context: Optional[List[dict]] = None) -> dict:
        """
        Parse transcript and execute browser command.
        Routes to recovery or baseline based on state and _waiting_for_recovery flag.
        Accepts user_context for multi-turn interactions.
        Ensures state transitions are always valid for integration test and real workflow.
        """
        prev_state = self.state_manager.current_state
        # Always start from PROCESSING for command execution
        if prev_state != AppState.PROCESSING:
            self.state_manager.transition_to(AppState.PROCESSING)
        try:
            # 1. ORCHESTRATION: Route transcript to the correct mode
            if prev_state == AppState.AWAITING_INPUT and getattr(self, "_waiting_for_recovery", False):
                result = self.command_orchestrator.processRecoveryInput(transcript, context=user_context)
            else:
                result = self.command_orchestrator.processTranscript(transcript, context=user_context)

            # Reset the recovery flag once processing starts
            self._waiting_for_recovery = False

            # 2. EVALUATE: Did the Orchestrator ask for more info/confirmation?
            if isinstance(result, str) or (isinstance(result, dict) and result.get("needs_input")):
                prompt = result if isinstance(result, str) else result.get("prompt")
                self._current_user_prompt = prompt
                # Allow transition to AWAITING_INPUT from any state
                self.state_manager._current_state = AppState.AWAITING_INPUT
                self.state_manager._state_data.state = AppState.AWAITING_INPUT
                return {
                    "success": True,
                    "needs_input": True,
                    "prompt": prompt,
                    "user_prompt": prompt,
                    "context": user_context
                }

            # 3. EXECUTION: We have a valid command/substep list
            self.state_manager.transition_to(AppState.EXECUTING)
            browser = get_browser()
            if not browser:
                raise RuntimeError("No active browser session found.")

            # Execute in browser (run_command handles the substep sequence)
            history = await run_command(result, browser)

            # 4. DONE: Successful completion of the current task
            self.command_orchestrator.set_done()
            self.state_manager.transition_to(AppState.LISTENING)
            return {"success": True, "details": history}

        except Exception as e:
            # 5. ERROR RECOVERY: Something went wrong in browser or LLM
            logger.error(f"Execution Error: {e}")
            self.state_manager.transition_to(AppState.ERROR)
            error_data = {"message": str(e), "last_transcript": transcript}
            recovery_payload = self.command_orchestrator.handle_browser_feedback(error_data)
            self._waiting_for_recovery = True
            self._current_user_prompt = recovery_payload.get("prompt")
            # Allow transition to AWAITING_INPUT from any state
            self.state_manager._current_state = AppState.AWAITING_INPUT
            self.state_manager._state_data.state = AppState.AWAITING_INPUT
            return {
                "success": False,
                "needs_input": True,
                "prompt": self._current_user_prompt,
                "user_prompt": self._current_user_prompt,
                "context": user_context
            }
    

   
    async def on_silence_detected(self, audio_data: bytes):
        """
        Called by our audio capture module when VAD detects silence.
        Transitions from RECORDING -> PROCESSING and processes the audio.
        
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
        Flow: IDLE -> LISTENING -> (automatic from here based on VAD)
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
        self._waiting_for_recovery = False
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
    Start voice mode with audio capture.
    
    This endpoint:
    1. Calls server.start_voice_mode() which:
       - Sets is_voice_mode_active = True
       - Transitions to LISTENING state
       - Starts audio capture with VAD
    2. Audio capture runs until /audio/capture/stop is called
    
    Frontend should call this when user clicks "Start"
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    # Check if we can start voice mode
    current_state = server.state_manager.current_state
    if current_state != AppState.IDLE:
        return {
            "ok": False, 
            "message": f"Cannot start from state: {current_state.value}. Call /reset first."
        }
    
    try:
        # Use the server's start_voice_mode method
        # This handles everything: state transition + audio capture
        result = await server.start_voice_mode()
        
        if result["success"]:
            return {
                "ok": True,
                "message": "Voice mode started - listening for commands",
                "state": result["state"]
            }
        else:
            return {
                "ok": False,
                "message": result.get("error", "Failed to start voice mode")
            }
            
    except Exception as e:
        logger.error(f"Error starting voice mode: {e}")
        return {
            "ok": False,
            "message": f"Error: {str(e)}"
        }


@app.post("/audio/capture/stop")
async def stop_audio_capture_endpoint():
    """
    Stop voice mode and audio capture.
    
    This endpoint:
    1. Calls server.stop_voice_mode() which:
       - Sets is_voice_mode_active = False
       - Stops audio capture
       - Transitions to IDLE state
    
    Frontend should call this when user clicks "Stop"
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    try:
        # Use the server's stop_voice_mode method
        # This handles everything: state transition + stopping audio capture
        result = await server.stop_voice_mode()
        
        return {
            "ok": True,
            "message": "Voice mode stopped",
            "state": result["state"]
        }
        
    except Exception as e:
        logger.error(f"Error stopping voice mode: {e}")
        return {
            "ok": False,
            "message": f"Error: {str(e)}"
        }


@app.get("/audio/capture/status")
async def audio_capture_status():
    """
    Get status of audio capture and voice mode.
    
    Returns both low-level capture status and high-level voice mode status.
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    global _capture_instance
    
    # Get low-level capture status
    with _capture_lock:
        if _capture_instance is None:
            capture_info = {
                "capturing": False,
                "state": "idle"
            }
        else:
            capture_info = {
                "capturing": True,
                "state": _capture_instance.state,
                "stats": _capture_instance.get_stats(),
            }
    
    # Get high-level server status
    server_status = server.get_status()
    
    return {
        "voice_mode_active": server.is_voice_mode_active,
        "app_state": server.state_manager.current_state.value,
        "capture": capture_info,
        "last_transcript": server_status.get("last_transcript"),
        "last_command": server_status.get("last_command"),
        "current_prompt": server_status.get("user_prompt")
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
      - State changes (idle -> listening -> recording -> processing -> executing -> listening)
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

