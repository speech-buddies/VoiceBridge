import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import unittest.mock as _mock

# ---------------------------------------------------------------------------
# Stub out browser_use before importing the module under test so the import
# succeeds even when the real browser_use package is not installed.
# ---------------------------------------------------------------------------
_browser_use_stub = _mock.MagicMock()
sys.modules.setdefault("browser_use", _browser_use_stub)

import session_manager  # noqa: E402 — module under test
from session_manager import start_session, stop_session, get_browser  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_browser():
    """Return a mock Browser with async start/stop methods."""
    browser = MagicMock()
    browser.start = AsyncMock()
    browser.stop = AsyncMock()
    return browser


@pytest.fixture(autouse=True)
def reset_browser_state():
    """Reset the module-level _browser to None before every test."""
    session_manager._browser = None
    yield
    session_manager._browser = None


# ---------------------------------------------------------------------------
# Tests for start_session
# ---------------------------------------------------------------------------

class TestStartSession:

    @pytest.mark.asyncio
    async def test_start_session_returns_started_message(self):
        """start_session should return 'Browser started' when no browser is running."""
        mock_browser = _make_mock_browser()
        with patch("session_manager.Browser", return_value=mock_browser):
            result = await start_session()
        assert result == "Browser started"

    @pytest.mark.asyncio
    async def test_start_session_creates_browser_with_keep_alive(self):
        """start_session should instantiate Browser with keep_alive=True."""
        mock_browser = _make_mock_browser()
        with patch("session_manager.Browser", return_value=mock_browser) as MockBrowser:
            await start_session()
        MockBrowser.assert_called_once_with(keep_alive=True)

    @pytest.mark.asyncio
    async def test_start_session_calls_browser_start(self):
        """start_session should call browser.start() exactly once."""
        mock_browser = _make_mock_browser()
        with patch("session_manager.Browser", return_value=mock_browser):
            await start_session()
        mock_browser.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_session_sets_global_browser(self):
        """start_session should assign the new Browser instance to _browser."""
        mock_browser = _make_mock_browser()
        with patch("session_manager.Browser", return_value=mock_browser):
            await start_session()
        assert session_manager._browser is mock_browser

    @pytest.mark.asyncio
    async def test_start_session_already_running_returns_message(self):
        """start_session should return 'Browser already running' if a browser is active."""
        session_manager._browser = _make_mock_browser()
        result = await start_session()
        assert result == "Browser already running"

    @pytest.mark.asyncio
    async def test_start_session_already_running_does_not_create_new_browser(self):
        """start_session should not instantiate a second Browser if one is already set."""
        existing = _make_mock_browser()
        session_manager._browser = existing
        with patch("session_manager.Browser") as MockBrowser:
            await start_session()
        MockBrowser.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_session_already_running_does_not_call_start(self):
        """start_session should not call browser.start() when a session is already active."""
        existing = _make_mock_browser()
        session_manager._browser = existing
        await start_session()
        existing.start.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_start_session_already_running_preserves_existing_browser(self):
        """start_session should leave the existing _browser instance untouched."""
        existing = _make_mock_browser()
        session_manager._browser = existing
        await start_session()
        assert session_manager._browser is existing


# ---------------------------------------------------------------------------
# Tests for stop_session
# ---------------------------------------------------------------------------

class TestStopSession:

    @pytest.mark.asyncio
    async def test_stop_session_returns_stopped_message(self):
        """stop_session should return 'Browser stopped' when a browser is running."""
        session_manager._browser = _make_mock_browser()
        result = await stop_session()
        assert result == "Browser stopped"

    @pytest.mark.asyncio
    async def test_stop_session_calls_browser_stop(self):
        """stop_session should call browser.stop() exactly once."""
        mock_browser = _make_mock_browser()
        session_manager._browser = mock_browser
        await stop_session()
        mock_browser.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_session_clears_global_browser(self):
        """stop_session should set _browser back to None after stopping."""
        session_manager._browser = _make_mock_browser()
        await stop_session()
        assert session_manager._browser is None

    @pytest.mark.asyncio
    async def test_stop_session_not_running_returns_message(self):
        """stop_session should return 'Browser already stopped' when no browser is active."""
        result = await stop_session()
        assert result == "Browser already stopped"

    @pytest.mark.asyncio
    async def test_stop_session_not_running_does_not_call_stop(self):
        """stop_session should not attempt to call stop() when _browser is None."""
        # _browser is None (reset by autouse fixture)
        # If stop() were called on None this would raise; the test ensures it doesn't.
        await stop_session()  # must not raise

    @pytest.mark.asyncio
    async def test_stop_session_browser_remains_none_after_already_stopped(self):
        """_browser should remain None if stop_session is called with no active session."""
        await stop_session()
        assert session_manager._browser is None


# ---------------------------------------------------------------------------
# Tests for get_browser
# ---------------------------------------------------------------------------

class TestGetBrowser:

    def test_get_browser_returns_none_when_not_started(self):
        """get_browser should return None when no session has been started."""
        assert get_browser() is None

    def test_get_browser_returns_browser_instance_when_running(self):
        """get_browser should return the active Browser instance."""
        mock_browser = _make_mock_browser()
        session_manager._browser = mock_browser
        assert get_browser() is mock_browser

    def test_get_browser_returns_none_after_session_cleared(self):
        """get_browser should return None after _browser is reset to None."""
        session_manager._browser = _make_mock_browser()
        session_manager._browser = None
        assert get_browser() is None

    def test_get_browser_does_not_modify_state(self):
        """get_browser should be a pure read — it must not change _browser."""
        mock_browser = _make_mock_browser()
        session_manager._browser = mock_browser
        get_browser()
        assert session_manager._browser is mock_browser


# ---------------------------------------------------------------------------
# Tests for start → stop → start lifecycle
# ---------------------------------------------------------------------------

class TestSessionLifecycle:

    @pytest.mark.asyncio
    async def test_full_start_stop_cycle(self):
        """A full start → stop cycle should leave _browser as None."""
        mock_browser = _make_mock_browser()
        with patch("session_manager.Browser", return_value=mock_browser):
            await start_session()
            await stop_session()
        assert session_manager._browser is None

    @pytest.mark.asyncio
    async def test_restart_after_stop_creates_new_browser(self):
        """After stopping, start_session should create a fresh Browser instance."""
        first = _make_mock_browser()
        second = _make_mock_browser()
        with patch("session_manager.Browser", side_effect=[first, second]):
            await start_session()
            await stop_session()
            await start_session()
        assert session_manager._browser is second

    @pytest.mark.asyncio
    async def test_double_start_only_one_browser_created(self):
        """Calling start_session twice should only ever create one Browser."""
        mock_browser = _make_mock_browser()
        with patch("session_manager.Browser", return_value=mock_browser) as MockBrowser:
            await start_session()
            await start_session()
        assert MockBrowser.call_count == 1

    @pytest.mark.asyncio
    async def test_double_stop_only_calls_browser_stop_once(self):
        """Calling stop_session twice should only call browser.stop() once."""
        mock_browser = _make_mock_browser()
        session_manager._browser = mock_browser
        await stop_session()
        await stop_session()
        mock_browser.stop.assert_awaited_once()