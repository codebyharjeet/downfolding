"""
Unit and regression test for the ducc package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import ducc


def test_ducc_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ducc" in sys.modules
