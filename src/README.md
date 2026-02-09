# VoiceBridge Source Code

The folders and files for this project are as follows:

src/
├── __init__.py
├── main.py                          # Application entry point
│
├── presentation/                    # Presentation Layer
│   ├── __init__.py
│   ├── user_interface.py            # User Interface Module
│   ├── accessibility_layer.py       # Accessibility Layer (WCAG compliance)
│   └── feedback_display.py          # Feedback Display Module
│
├── app/                     # Application (Processing) Layer
│   ├── __init__.py
│   ├── speech_to_text_engine.py     # Speech-to-Text Engine
│   ├── intent_interpreter.py        # Intent Interpreter
│   ├── command_mapping.py           # Command Mapping Module
│   ├── command_execution.py         # Command Execution Layer
│   └── error_feedback.py            # Error Feedback Module
│
├── control/                         # Control (Orchestration) Layer
│   ├── __init__.py
│   ├── browser_controller.py        # Browser Controller
│   ├── session_manager.py           # Session Manager (states: capture, transcribe, confirm, execute)
│   └── error_handling_recovery.py   # Error Handling and Recovery Module
│
├── data/                            # Data Management Layer
│   ├── __init__.py
│   ├── storage_manager.py           # Data Storage Manager
│   ├── user_profile_manager.py      # User Profile Manager
│   └── audit_logger.py              # Audit Logger
│
├── security/                        # Security Layer
│   ├── __init__.py
│   ├── credential_manager.py        # Credential Manager
│   ├── encryption_manager.py        # Encryption Manager
│   └── out_of_scope_handler.py      # Out-of-Scope Handler
│
├── input/                           # Input Processing Layer
│   ├── __init__.py
│   ├── microphone_manager.py        # Microphone Manager
│   └── vad_noise_filter.py          # VAD and Noise Filter
│
├── personalization/                 # Personalization Layer
│   ├── __init__.py
│   ├── prompting_module.py          # Prompting Module
│   ├── model_tuner.py               # Model Tuner
│   └── instruction_registry.py      # Instruction Registry
│
├── models/                          # Data models and schemas
│   ├── __init__.py
│   ├── session.py                   # Session state models
│   ├── command.py                   # Command models
│   ├── user_profile.py              # User profile schema
│   └── audio_data.py                # Audio data models
│
├── config/                          # Configuration management
│   ├── __init__.py
│   ├── settings.py                  # Application settings
│   ├── constants.py                 # System constants
│   └── wcag_standards.py            # WCAG compliance configurations
│
└── utils/                           # Shared utilities
    ├── __init__.py
    ├── logger.py                    # Logging utilities
    ├── validators.py                # Input validation
    └── exceptions.py                # Custom exceptions


For installing all requirements, its best to create a virtual environment:

`python -m venv .venv`

and activate it:

` source venv/scripts/activate` (might differ depending on machine/terminal)

then install:

`pip install -r requirements.txt` 

To allow modules to access and import each other, we need to run 

` pip install -e .` 

# VoiceBridge Application Flow


```
┌─────────────────────────────────────────────────────────────────┐
│                         USER ACTIONS                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Starts application
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POST /audio/capture/start                           │
│                    (Frontend → Backend)                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STATE: IDLE → LISTENING                      │
│                                                                 │
│  • Audio Capture Module starts                                  │
│  • VAD (Voice Activity Detection) begins monitoring             │
│  • Waiting for voice input...                                   │
│  • Audio Capture Module starts recording                        │
│  • Capturing audio while voice is detected                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ VAD detects voice
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STATE: LISTENING → PROCESSING                 │
│                                                                 │
│  • Speech-to-Text Engine receives audio                         │
│  • Transcribing audio to text...                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Transcript ready
                              ▼
                              │ Command Orchestrator
                    ┌─────────┴─────────┐
                    │                   │
            Command Clear?      Command Ambiguous (needs_input)?
                    │                   │
                    ▼                   ▼
┌──────────────────────────┐  ┌──────────────────────────────┐
│  STATE: PROCESSING →     │  │  STATE: PROCESSING →         │
│         EXECUTING        │  │         LISTENING            │
│                          │  │                              │
│  • Command Orchestrator  │  │  • Orchestrator asks:        │
│    parses transcript     │  │    "Which Google product?"   │
│  • Browser Controller    │  │  • Frontend shows prompt     │
│    executes command      │  │  • Auto returns to LISTENING │
└──────────────────────────┘  └──────────────────────────────┘
            │                             │
            │                             │ User responds
            │                             ▼
            │                   ┌───────────────────────┐
            │                   │ LISTENING → PROCESSING│
            │                   
            │                   └───────────────────────┘
            │                             │
            │                    ┌────────┴──────────┐
            │                    │                   │
            │              Still need info?    Got enough info?
            │                    │                   │
            │                    ▼                   ▼
            │          [Loop to LISTENING]  [Go to EXECUTING]
            │                                        │
            └────────────────────────────────────────┘
                              │ Browser Controller finishes
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   STATE: EXECUTING → LISTENING                  │
│                                                                 │
│  • Back to listening for next command                           │
│  • Conversation context cleared                                 │
│  • Loop continues until user clicks "Stop"                      │
└─────────────────────────────────────────────────────────────────┘
```