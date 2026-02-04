# """
# Unit Test: VoiceBridge Backend Workflow (Mocked LLM)
# ====================================================

# This test simulates the backend workflow by hardcoding LLM replies, bypassing the actual LLM API. It verifies that the orchestrator and browser execution logic work as expected without external dependencies.

# Run this test with: python src/unit_test_backend_workflow.py
# """

# import asyncio
# from server import VoiceBridgeServer
# from state_manager import AppState

# # Monkeypatch the orchestrator to return hardcoded LLM responses
# class MockedOrchestrator:
#     def __init__(self):
#         self.primary_goal = None
#         self.awaiting_primary_goal_confirmation = False
#         self.active_browser_error = None
#         self.mode = "BASELINE"
#     def processTranscript(self, text, context=None):
#         # Simulate LLM returning a step list for the initial command
#         return {
#             "success": True,
#             "needs_input": True,
#             "prompt": "1. open amazon.com\n2. search for 'sunscreen'\n3. add to cart",
#             "user_prompt": "1. open amazon.com\n2. search for 'sunscreen'\n3. add to cart",
#             "context": None
#         }
#     def processRecoveryInput(self, text, context=None):
#         # Simulate LLM returning a single recovery step
#         return {
#             "success": True,
#             "needs_input": False,
#             "prompt": "click proceed to shopping",
#             "user_prompt": "click proceed to shopping",
#             "context": None
#         }
#     def handle_browser_feedback(self, error_data):
#         # Simulate error prompt
#         return {
#             "needs_input": True,
#             "prompt": "Please click 'Proceed to Shopping' to continue.",
#             "suggested_state": AppState.AWAITING_INPUT.value
#         }
#     def set_done(self):
#         pass

# async def main():
#     print("\n=== Unit Test: Backend Workflow (Mocked LLM) ===\n")
#     server = VoiceBridgeServer()
#     # Patch the orchestrator
#     server.command_orchestrator = MockedOrchestrator()
#     server.state_manager.transition_to(AppState.LISTENING)
#     transcript = "add sunscreen to cart on Amazon"
#     server.state_manager.transition_to(AppState.PROCESSING)
#     print(f"Transcript: {transcript}")
#     result = await server.parse_and_execute_command(transcript)
#     print("LLM Response (mocked):", result)
#     substeps = []
#     # Extract substeps from mocked LLM response
#     if isinstance(result, dict):
#         for key in ["prompt", "user_prompt", "command"]:
#             if key in result and isinstance(result[key], str):
#                 text = result[key]
#                 lines = [line.strip() for line in text.splitlines() if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
#                 if lines:
#                     substeps = [line[line.find('.')+1:].strip() if '.' in line else line for line in lines]
#                     break
#     if not substeps and isinstance(result, str):
#         lines = [line.strip() for line in result.splitlines() if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
#         if lines:
#             substeps = [line[line.find('.')+1:].strip() if '.' in line else line for line in lines]

#     if not substeps:
#         print("[ERROR] No substeps found in mocked LLM output. Test cannot proceed.")
#         return

#     print("\n[STEP] Executing browser substeps (real browser):")
#     from control.session_manager import get_browser, start_session, stop_session
#     from control.browser_orchestrator import run_command
#     await start_session()
#     browser = get_browser()
#     try:
#         for i, step in enumerate(substeps, 1):
#             print(f"  [{i}] Executing: {step}")
#             try:
#                 history = await run_command(step, browser)
#                 print(f"    [Result]: {history}")
#                 # After opening Amazon, simulate a browser error and recovery
#                 if i == 1 and 'amazon' in step.lower():
#                     print("\n[Simulating browser error/interstitial after opening Amazon...]")
#                     recovery_step = "continue shopping"
#                     print(f"[Hardcoded Recovery Step]: {recovery_step}")
#                     try:
#                         recovery_result = await run_command(recovery_step, browser)
#                         print(f"    [Recovery Result]: {recovery_result}")
#                     except Exception as e:
#                         print(f"    [Recovery ERROR]: {e}")
#             except Exception as e:
#                 print(f"    [ERROR]: {e}")
#                 break
#     finally:
#         await stop_session()

#     print("\n=== Unit Test Complete ===\n")

# if __name__ == "__main__":
#     asyncio.run(main())
"""
Unit Test: VoiceBridge Backend Workflow (Mocked LLM)
====================================================

This test simulates the backend workflow by hardcoding LLM replies, bypassing the actual LLM API. 
It verifies that the orchestrator and browser execution logic work as expected without external dependencies.

CORRECTED VERSION: Mock returns responses in the exact format expected by parse_and_execute_command.

Run this test with: python unit_test_backend_workflow_corrected.py
"""

import asyncio
from server import VoiceBridgeServer
from state_manager import AppState

# Monkeypatch the orchestrator to return hardcoded LLM replies
class MockedOrchestrator:
    def __init__(self):
        self.primary_goal = None
        self.awaiting_primary_goal_confirmation = False
        self.active_browser_error = None
        self.mode = "BASELINE"
        self.app_state = None
    
    def processTranscript(self, text, context=None):
        """
        Return format expected by parse_and_execute_command (server.py line 348-360):
        - If needs_input=True: system waits for user input (AWAITING_INPUT state)
        - If needs_input=False: system proceeds to execute (EXECUTING state)
        
        The "prompt" field contains the substeps that browser_orchestrator will parse.
        """
        print(f"[MOCK] processTranscript: '{text}'")
        # Return command WITHOUT needs_input so it proceeds to execution
        return {
            "success": True,
            "needs_input": False,  # Don't wait for input, proceed to execute
            "prompt": "1. open amazon.com\n2. search for 'sunscreen'\n3. add to cart",
            "user_prompt": "1. open amazon.com\n2. search for 'sunscreen'\n3. add to cart",
            "suggested_state": "processing"
        }
    
    def processRecoveryInput(self, text, context=None):
        """
        Return recovery command after browser error.
        Should NOT have needs_input=True so execution proceeds.
        """
        print(f"[MOCK] processRecoveryInput: '{text}'")
        return {
            "success": True,
            "needs_input": False,  # Ready to execute recovery
            "prompt": "click the continue shopping button",
            "user_prompt": "click the continue shopping button",
            "suggested_state": "processing"
        }
    
    def handle_browser_feedback(self, error_data):
        """
        Called when browser encounters error.
        SHOULD have needs_input=True to prompt user.
        """
        print(f"[MOCK] handle_browser_feedback: {error_data}")
        return {
            "needs_input": True,  # Ask user for input
            "prompt": "Amazon is showing an interstitial. Would you like me to click 'Continue Shopping'?",
            "suggested_state": AppState.AWAITING_INPUT.value
        }
    
    def set_done(self):
        print("[MOCK] set_done called")
        self.app_state = "DONE"


async def main():
    print("\n=== Unit Test: Backend Workflow (Mocked LLM) ===\n")
    server = VoiceBridgeServer()
    
    # Patch the orchestrator
    server.command_orchestrator = MockedOrchestrator()
    server.state_manager.transition_to(AppState.LISTENING)
    
    transcript = "add sunscreen to cart on Amazon"
    server.state_manager.transition_to(AppState.PROCESSING)
    print(f"Transcript: {transcript}")
    
    # This will call processTranscript and then execute in browser
    result = await server.parse_and_execute_command(transcript)
    print("LLM Response (mocked):", result)
    
    # The result contains the substep list in result["prompt"] or result itself
    substeps = []
    
    # Extract substeps from mocked LLM response
    if isinstance(result, dict):
        for key in ["prompt", "user_prompt", "command"]:
            if key in result and isinstance(result[key], str):
                text = result[key]
                lines = [line.strip() for line in text.splitlines() 
                        if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
                if lines:
                    substeps = [line[line.find('.')+1:].strip() if '.' in line else line 
                               for line in lines]
                    break
    
    if not substeps and isinstance(result, str):
        lines = [line.strip() for line in result.splitlines() 
                if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
        if lines:
            substeps = [line[line.find('.')+1:].strip() if '.' in line else line 
                       for line in lines]

    if not substeps:
        print("[ERROR] No substeps found in mocked LLM output. Test cannot proceed.")
        return

    print("\n[STEP] Executing browser substeps (real browser):")
    from control.session_manager import get_browser, start_session, stop_session
    from control.browser_orchestrator import run_command
    
    await start_session()
    browser = get_browser()
    
    try:
        for i, step in enumerate(substeps, 1):
            print(f"  [{i}] Executing: {step}")
            try:
                history = await run_command(step, browser)
                print(f"    [Result]: {history}")
                
                # After opening Amazon, simulate a browser error and recovery
                if i == 1 and 'amazon' in step.lower():
                    print("\n[Simulating browser error/interstitial after opening Amazon...]")
                    
                    # This is what would happen in real flow:
                    # 1. Browser hits error
                    # 2. handle_browser_feedback is called
                    # 3. User prompted (AWAITING_INPUT state)
                    # 4. User responds
                    # 5. processRecoveryInput is called
                    # 6. Recovery command executed
                    
                    # Simulate step 2: error feedback
                    error_data = {
                        "message": "Interstitial page detected",
                        "step": step,
                        "index": i
                    }
                    error_response = server.command_orchestrator.handle_browser_feedback(error_data)
                    print(f"[Error Handler Response]: {error_response}")
                    
                    # Simulate user responding with "continue"
                    recovery_input = "yes, continue"
                    print(f"\n[User Recovery Input]: {recovery_input}")
                    
                    # Get recovery command from orchestrator
                    recovery_response = server.command_orchestrator.processRecoveryInput(recovery_input)
                    print(f"[Recovery Response]: {recovery_response}")
                    
                    # Extract and execute recovery step
                    recovery_step = recovery_response.get("prompt", "")
                    if recovery_step:
                        print(f"[Executing Recovery]: {recovery_step}")
                        try:
                            recovery_result = await run_command(recovery_step, browser)
                            print(f"    [Recovery Result]: {recovery_result}")
                        except Exception as e:
                            print(f"    [Recovery ERROR]: {e}")
            
            except Exception as e:
                print(f"    [ERROR]: {e}")
                break
    
    finally:
        await stop_session()

    print("\n=== Unit Test Complete ===\n")


if __name__ == "__main__":
    asyncio.run(main())