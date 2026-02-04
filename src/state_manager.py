"""
State Manager for VoiceBridge Application

Manages application state and coordinates between audio capture, stt, 
and browser control modules.
"""

from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppState(Enum):
    """Application states"""
    IDLE = "idle" 
    LISTENING = "listening" # After startup, awaiting voice detection and starting browser session
    RECORDING = "recording" # After voice detection, awaiting silence
    PROCESSING = "processing" # After silence, awaiting command orchestrator response
    AWAITING_INPUT = "awaiting_input"  # Command orchestrator needs clarification
    EXECUTING = "executing" # executing browser controller
    ERROR = "error" # error in browser controller
    STOP = "stop" # application stopping and closing browser session


@dataclass
class StateData:
    """Data associated with current state"""
    state: AppState
    timestamp: datetime
    audio_data: Optional[bytes] = None
    transcript: Optional[str] = None
    command: Optional[dict] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None


class StateManager:
    """
    Centralized state management for the voice-controlled browser application.
    
    Handles state transitions and provides callbacks for state changes.
    """
    
    def __init__(self):
        self._current_state = AppState.IDLE
        self._state_data = StateData(
            state=AppState.IDLE,
            timestamp=datetime.now()
        )
        self._lock = threading.RLock()
        self._callbacks = {state: [] for state in AppState}
        self._global_callbacks = []
        
    @property
    def current_state(self) -> AppState:
        """Get current application state"""
        with self._lock:
            return self._current_state
    
    @property
    def state_data(self) -> StateData:
        """Get current state data"""
        with self._lock:
            return self._state_data
    
    def transition_to(
        self, 
        new_state: AppState, 
        **kwargs
    ) -> bool:
        """
        Transition to a new state with optional data.
        
        Args:
            new_state: Target state
            **kwargs: Additional data for the state (audio_data, transcript, etc.)
            
        Returns:
            bool: True if transition was valid and successful
        """
        with self._lock:
            if not self._is_valid_transition(self._current_state, new_state):
                logger.warning(
                    f"Invalid transition from {self._current_state.value} "
                    f"to {new_state.value}"
                )
                return False
            
            old_state = self._current_state
            self._current_state = new_state
            
            # Create new state data
            self._state_data = StateData(
                state=new_state,
                timestamp=datetime.now(),
                audio_data=kwargs.get('audio_data'),
                transcript=kwargs.get('transcript'),
                command=kwargs.get('command'),
                error=kwargs.get('error'),
                metadata=kwargs.get('metadata')
            )
            
            logger.info(
                f"State transition: {old_state.value} -> {new_state.value}"
            )
            
            # Execute callbacks
            self._execute_callbacks(new_state, old_state)
            
            return True
    
    def _is_valid_transition(
        self, 
        from_state: AppState, 
        to_state: AppState
    ) -> bool:
        """
        Validate state transitions based on defined rules.
        
        Valid transitions:
        - IDLE -> LISTENING
        - LISTENING -> RECORDING or ERROR
        - RECORDING -> PROCESSING or ERROR
        - PROCESSING -> EXECUTING or AWAITING_INPUT or ERROR
        - AWAITING_INPUT -> LISTENING or EXECUTING or ERROR
        - EXECUTING -> IDLE or ERROR
        - ERROR -> IDLE or LISTENING
        """
        valid_transitions = {
            AppState.IDLE: [AppState.LISTENING],
            AppState.LISTENING: [AppState.RECORDING, AppState.ERROR],
            AppState.RECORDING: [AppState.PROCESSING, AppState.ERROR],
            AppState.PROCESSING: [AppState.EXECUTING, AppState.AWAITING_INPUT, AppState.ERROR],
            AppState.AWAITING_INPUT: [AppState.LISTENING, AppState.EXECUTING, AppState.ERROR],
            AppState.EXECUTING: [AppState.IDLE, AppState.LISTENING, AppState.ERROR],
            AppState.ERROR: [AppState.IDLE, AppState.LISTENING]
        }
        
        return to_state in valid_transitions.get(from_state, [])
    
    def register_callback(
        self, 
        state: Optional[AppState], 
        callback: Callable[[StateData, AppState], None]
    ):
        """
        Register a callback for state changes.
        
        Args:
            state: Specific state to listen for, or None for all states
            callback: Function to call on state change (receives state_data, old_state)
        """
        with self._lock:
            if state is None:
                self._global_callbacks.append(callback)
            else:
                self._callbacks[state].append(callback)
    
    def _execute_callbacks(self, new_state: AppState, old_state: AppState):
        """Execute registered callbacks for state transition"""
        # Execute state-specific callbacks
        for callback in self._callbacks[new_state]:
            try:
                callback(self._state_data, old_state)
            except Exception as e:
                logger.error(f"Error in state callback: {e}")
        
        # Execute global callbacks
        for callback in self._global_callbacks:
            try:
                callback(self._state_data, old_state)
            except Exception as e:
                logger.error(f"Error in global callback: {e}")
    
    def handle_error(self, error: str):
        """Transition to error state with error message"""
        self.transition_to(AppState.ERROR, error=error)
    
    def reset(self):
        """Reset to idle state"""
        self.transition_to(AppState.IDLE)
    
    def get_state_info(self) -> dict:
        """Get current state information as dictionary"""
        with self._lock:
            return {
                'state': self._current_state.value,
                'timestamp': self._state_data.timestamp.isoformat(),
                'has_audio': self._state_data.audio_data is not None,
                'has_transcript': self._state_data.transcript is not None,
                'has_command': self._state_data.command is not None,
                'error': self._state_data.error,
                'metadata': self._state_data.metadata
            }