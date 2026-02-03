"""
FastAPI Server Entry Point

Simple REST API for frontend to interact with voice browser control system.
"""

import asyncio
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from state_manager import StateManager, AppState

# TODO: Import our actual modules here

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """Main server that exposes API endpoints for frontend"""
    
    def __init__(self):
        self.state_manager = StateManager()
        self.connection_manager = ConnectionManager()
        
        # Register callback to broadcast state changes to WebSocket clients
        self.state_manager.register_callback(None, self._broadcast_state_change)
    
    def _broadcast_state_change(self, state_data, old_state):
        """Broadcast state changes to all WebSocket clients"""
        asyncio.create_task(
            self.connection_manager.broadcast({
                'type': 'state_change',
                'old_state': old_state.value,
                'new_state': state_data.state.value,
                'timestamp': state_data.timestamp.isoformat(),
                'has_transcript': state_data.transcript is not None,
                'transcript': state_data.transcript,
                'error': state_data.error
            })
        )
    
    # ========================================================================
    # Methods to call your modules - implement these with your actual code
    # ========================================================================
    
    async def start_vad_listening(self, config: Optional[dict] = None):
        """
        Start your VAD module listening for speech.
        
        TODO: Implement with your actual VAD code
        Example:
            await your_vad_module.start_listening(config)
        """
        logger.info("Starting VAD listening...")
        # Your VAD code here
        pass
    
    
    async def transcribe_audio(self, audio_data: bytes) -> str:
        """
        Transcribe audio data using your transcription module.
        
        TODO: Implement with your actual transcription code
        Example:
            return await your_transcription_module.transcribe(audio_data)
        """
        logger.info(f"Transcribing {len(audio_data)} bytes of audio...")
        
        # Your transcription code here
        # For now, returning dummy transcript
        await asyncio.sleep(0.5)  # Simulate processing time
        return "example transcript from audio"
    
    async def execute_browser_command(self, command: dict) -> dict:
        """
        Execute browser command using your browser control module.
        
        TODO: Implement with your actual browser control code
        Example:
            return await your_browser_module.execute(command)
        """
        logger.info(f"Executing browser command: {command}")
        
        # Your browser control code here
        # For now, returning dummy result
        await asyncio.sleep(0.3)  # Simulate execution time
        return {"success": True, "action": command.get("action")}
    
    async def parse_transcript_to_command(self, transcript: str) -> dict:
        """
        Parse natural language transcript into browser command.
        
        TODO: Implement your command parsing logic
        Example:
            return your_parser.parse(transcript)
        """
        logger.info(f"Parsing transcript: {transcript}")
        pass 
    
    # ========================================================================
    # Public API methods for FastAPI endpoints
    # ========================================================================
    
    def get_status(self) -> dict:
        """Get current server status"""
        return self.state_manager.get_state_info()
    
    def get_current_state(self) -> str:
        """Get current state as string"""
        return self.state_manager.current_state.value
    
    async def handle_start_listening(self, config: Optional[dict] = None) -> dict:
        """
        Handle frontend request to start listening.
        Called from POST /listen/start endpoint.
        """
        current_state = self.state_manager.current_state
        
        if current_state != AppState.IDLE:
            return {
                "success": False,
                "error": f"Cannot start listening from state: {current_state.value}"
            }
        
        # Transition to listening state
        self.state_manager.transition_to(AppState.LISTENING)
        
        # Start your VAD module
        await self.start_vad_listening(config)
        
        return {"success": True, "state": "listening"}
    
    
    def handle_reset(self) -> dict:
        """Reset to idle state"""
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
    logger.info("Starting Voice Browser Server...")
    server = VoiceBridgeServer()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Voice Browser Server...")
    if server:
        await server.stop_vad_listening()


# Create FastAPI app
app = FastAPI(
    title="VoiceBridge API",
    description="REST API for VoiceBridge backend",
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
        "version": "1.0.0"
    }


@app.get("/status")
async def get_status():
    """Get current application status"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    return server.get_status()


@app.post("/listen/start")
async def start_listening(request):
    """Start listening for voice commands"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = await server.handle_start_listening(request.config)
    
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return {"status": "success", "message": "Started listening", "state": result["state"]}


@app.post("/reset")
async def reset_server():
    """Reset server to idle state"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    result = server.handle_reset()
    return {"status": "success", "message": "Server reset", "state": result["state"]}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not server:
        return {"status": "unhealthy", "reason": "Server not initialized"}
    
    return {
        "status": "healthy",
        "state": server.get_current_state(),
        "timestamp": server.state_manager.state_data.timestamp.isoformat()
    }


# ============================================================================
# WebSocket Endpoint for Real-time Updates
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time state updates.
    Frontend can connect here to receive live updates about state changes.
    """
    if not server:
        await websocket.close(code=1011, reason="Server not initialized")
        return
    
    await server.connection_manager.connect(websocket)
    
    try:
        # Send initial state
        await websocket.send_json({
            'type': 'connected',
            'state': server.get_current_state(),
            'data': server.get_status()
        })
        
        # Keep connection alive
        while True:
            # Just receive pings to keep connection alive
            # All state updates are broadcast automatically
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