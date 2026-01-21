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

async def run_command(command: str):
    browser = Browser(keep_alive=True)
    llm = ChatBrowserUse()

    agent = Agent(
        task=command,
        llm=llm,
        browser=browser,
    )

    history = await agent.run()
    return history


def run_command_sync(command: str):
    # lets FastAPI call it without you thinking about event loops
    return asyncio.run(run_command(command))

# async def example():
#     browser = Browser(
#         # use_cloud=True,  # Uncomment to use a stealth browser on Browser Use Cloud
#         keep_alive=True,  # Keep the browser session alive for multiple tasks
#     )

#     llm = ChatBrowserUse()

#     agent = Agent(
#         task="Go to amazon and buy a book for me.",
#         llm=llm,
#         browser=browser,
#     )

#     history = await agent.run()
#     return history

# if __name__ == "__main__":
#     print("Key loaded?", bool(os.getenv("BROWSER_USE_API_KEY")))
#     history = asyncio.run(example())