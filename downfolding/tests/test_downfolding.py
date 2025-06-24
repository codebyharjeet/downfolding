"""
Unit and regression test for the downfolding package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import downfolding


def test_downfolding_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "downfolding" in sys.modules
