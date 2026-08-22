"""PLAN.md §10 -- Phase 7 demo.

The test that matters most here is the offline one. §10 step 8: "Claiming
offline operation without proving it is the weakest possible version of this
demo." A `--assert-offline` flag that prints a reassuring line without
enforcing anything is worse than no flag at all, because it converts an
unverified claim into an apparently verified one. So the guard is tested by
actually trying to open a socket inside it.
"""
from __future__ import annotations

import socket

import pytest

from pipeline.demo import OfflineViolation, no_network


def test_no_network_blocks_socket_creation():
    with pytest.raises(OfflineViolation):
        with no_network(True):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)


def test_no_network_disabled_allows_socket():
    with no_network(False):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.close()


def test_no_network_restores_socket_afterwards():
    """A guard that leaks its patch would break every later test in the run."""
    original = socket.socket
    with no_network(True):
        pass
    assert socket.socket is original


def test_no_network_restores_socket_even_when_body_raises():
    original = socket.socket
    with pytest.raises(ValueError):
        with no_network(True):
            raise ValueError("boom")
    assert socket.socket is original


def test_offline_violation_is_not_silently_swallowed():
    """The failure must propagate, not be counted and reported later."""
    with pytest.raises(OfflineViolation, match="assert-offline"):
        with no_network(True):
            socket.create_connection(("127.0.0.1", 9))
