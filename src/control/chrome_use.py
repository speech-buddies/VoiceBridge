import asyncio
import os
import sys
from pathlib import Path

# so imports work from project root if needed
sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

from browser_use import Agent, Browser, ChatBrowserUse, ChatGoogle  # if this import fails, tell me your browser-use version

llm = ChatBrowserUse()
browser = Browser(
    executable_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    user_data_dir=r"C:\Users\HP\AppData\Local\Google\Chrome\User Data",
    profile_directory="Default",  # or "Profile 1"
)

async def main():
    agent = Agent(
        llm=llm,
        task="go to amazon.com and search for pens to draw on whiteboards",
        browser=browser,
    )
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())
