"""Configuration for pytest.

This module configures pytest to ensure proper import paths and setup for tests.
"""

import pytest
import os
import sys

# Add the src directory to the Python path so we can import our modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if src_path not in sys.path:
    sys.path.insert(0, src_path) 