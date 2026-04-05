"""
Manages a single shared browser session for the application lifecycle.

Provides async utilities to start and stop a persistent browser instance,
ensuring reuse across commands, along with a getter to access the current session.

Author: Mazen Youssef
"""
from browser_use import Browser

# single shared browser instance for the whole app
_browser: Browser | None = None


async def start_session() -> str:
    global _browser

    # if already started, do nothing
    if _browser is not None:
        return "Browser already running"

    _browser = Browser(keep_alive=True)  # keep it alive across commands
    await _browser.start()
    return "Browser started"


async def stop_session() -> str:
    global _browser

    # if not running, do nothing
    if _browser is None:
        return "Browser already stopped"

    await _browser.stop()
    _browser = None
    return "Browser stopped"

def get_browser() -> Browser | None:
    return _browser