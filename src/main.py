import asyncio
import sys
from fastapi import FastAPI, Body, HTTPException
from control.browser_use_runner import run_command, run_command_sync

app = FastAPI()

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@app.post("/control/command")
async def control_command(command: str = Body(..., media_type="text/plain")):
    command = command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="command cannot be empty")

    history = await asyncio.to_thread(run_command_sync, command)
    return {"command": command, "history": str(history)}
