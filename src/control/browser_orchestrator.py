import os
import sys
from browser_use import Agent, Browser, ChatBrowserUse
import asyncio
from dotenv import load_dotenv
from pathlib import Path


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"   # src/.env

load_dotenv()
BROWSER_USE_API_KEY = os.getenv("BROWSER_USE_API_KEY")

async def run_command(command: str, browser: Browser):

    llm = ChatBrowserUse()

    agent = Agent(
        task=command,
        llm=llm,
        browser=browser,
    )

    history = await agent.run()
    return history