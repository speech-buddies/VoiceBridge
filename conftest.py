# conftest.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "input"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "app"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "control"))


import pytest
pytest_plugins = ('pytest_asyncio',)