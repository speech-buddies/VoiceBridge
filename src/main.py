import asyncio
import sys
from fastapi import FastAPI, Body, HTTPException
from control.browser_use_runner import run_command
from control.session_manager import get_browser, start_session, stop_session

app = FastAPI()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@app.post("/control/command")
async def control_command(command: str = Body(..., media_type="text/plain")):
    command = command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="command cannot be empty")

    browser = get_browser()
    if browser is None:
        raise HTTPException(status_code=409, detail="No active session. Call /control/session/start first.")

    history = await run_command(command, browser)
    return {"command": command, "history": str(history)}

@app.post("/control/session/{action}")
async def control_session(action: str):
    if action == "start":
        msg = await start_session()
        return {"ok": True, "message": msg}

    if action == "stop":
        msg = await stop_session()
        return {"ok": True, "message": msg}

    return {"ok": False, "message": "action must be start or stop"}