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
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from state_manager import StateManager, AppState

# TODO: Import our actual modules here

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


class ManualCommandRequest(BaseModel):
    """Manual command input (for testing/debugging)"""
    command: str
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
        
        # TODO: Initialize our modules
        # self.audio_capture = AudioCapture()
        # self.speech_to_text = SpeechToText()
        # self.command_orchestrator = CommandOrchestrator()
        # self.browser_controller = BrowserController()
        
        # Register callback to broadcast state changes to WebSocket clients
        self.state_manager.register_callback(None, self._broadcast_state_change)
        
        # Store last transcript and command for status endpoint
        self._last_transcript: Optional[str] = None
        self._last_command: Optional[str] = None
    
    def _broadcast_state_change(self, state_data, old_state):
        """Broadcast state changes to all WebSocket clients"""
        message = {
            'type': 'state_change',
            'old_state': old_state.value,
            'new_state': state_data.state.value,
            'timestamp': state_data.timestamp.isoformat(),
            'has_transcript': state_data.transcript is not None,
            'transcript': state_data.transcript,
            'error': state_data.error
        }
        
        logger.info(f"State: {old_state.value} → {state_data.state.value}")
        asyncio.create_task(self.connection_manager.broadcast(message))
    
    # ========================================================================
    # Module Integration - Implement these with our actual modules
    # ========================================================================
    
    async def start_audio_capture(self):
        """
        Start audio capture with VAD.
        This should set up callbacks for when voice is detected and when silence is detected.
        
        TODO: Implement with our audio capture module
        Example:
            self.audio_capture.on_voice_detected = self.on_voice_detected
            self.audio_capture.on_silence_detected = self.on_silence_detected
            await self.audio_capture.start()
        """
        logger.info("Starting audio capture with VAD...")
        
        # our audio capture initialization here
        # The audio capture module should call:
        # - self.on_voice_detected() when VAD detects voice
        # - self.on_silence_detected(audio_data) when VAD detects silence
        
        pass
    
    async def stop_audio_capture(self):
        """
        Stop audio capture.
        
        TODO: Implement with our audio capture module
        Example:
            await self.audio_capture.stop()
        """
        logger.info("Stopping audio capture...")
        # our audio capture cleanup here
        pass
    
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
    
    async def parse_and_execute_command(self, transcript: str) -> dict:
        """
        Parse transcript and execute browser command.
        
        TODO: Implement with our command orchestrator and browser controller
        Example:
            command = await self.command_orchestrator.parse(transcript)
            result = await self.browser_controller.execute(command)
            return result
        """
        logger.info(f"Parsing and executing: {transcript}")
        
        # our command parsing and execution here
        # Placeholder for testing
        await asyncio.sleep(0.3)
        return {
            "success": True,
            "action": "navigate",
            "details": f"Executed command from: {transcript}"
        }
    
    # ========================================================================
    # Callbacks from Audio Capture Module
    # ========================================================================
    
    def on_voice_detected(self):
        """
        Called by our audio capture module when VAD detects voice.
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
        
        if self.state_manager.current_state != AppState.RECORDING:
            logger.warning("Received audio but not in RECORDING state")
            return
        
        # Transition to processing
        self.state_manager.transition_to(AppState.PROCESSING, audio_data=audio_data)
        
        try:
            # Transcribe the audio
            transcript = await self.transcribe_audio(audio_data)
            self._last_transcript = transcript
            
            # Transition to executing
            self.state_manager.transition_to(AppState.EXECUTING, transcript=transcript)
            
            # Parse and execute the command
            result = await self.parse_and_execute_command(transcript)
            self._last_command = transcript
            
            # Back to listening (if voice mode still active) or idle
            if self.is_voice_mode_active:
                self.state_manager.transition_to(AppState.LISTENING)
            else:
                self.state_manager.transition_to(AppState.IDLE)
                
        except Exception as e:
            logger.error(f"Error in voice processing pipeline: {e}")
            self.state_manager.handle_error(str(e))
            
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
        return status
    
    def reset(self) -> dict:
        """Reset to idle state"""
        self.is_voice_mode_active = False
        self.state_manager.reset()
        return {"success": True, "state": "idle"}


# ============================================================================
# FastAPI Application
# ============================================================================

# Global server instance
server: Optional[VoiceBridgeServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global server
    
    # Startup
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
            "start": "POST /voice/start",
            "stop": "POST /voice/stop",
            "status": "GET /status",
            "manual": "POST /command/manual",
            "websocket": "WS /ws"
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


@app.post("/voice/start")
async def start_voice_mode():
    """
    Start voice mode.
    
    The app will:
      1. Start listening with VAD
      2. Detect when you speak (auto transition to recording)
      3. Detect when you stop (auto process, execute, return to listening)
      4. Repeat until you call /voice/stop
    
    This is the main endpoint our frontend needs to start the app.
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = await server.start_voice_mode()
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return {
        "success": True,
        "message": "Voice mode started - listening for commands",
        "state": result["state"]
    }


@app.post("/voice/stop")
async def stop_voice_mode():
    """
    Stop voice mode and return to idle.
    
    This stops the VAD and audio capture.
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = await server.stop_voice_mode()
    
    return {
        "success": True,
        "message": "Voice mode stopped",
        "state": result["state"]
    }


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