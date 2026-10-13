from __future__ import annotations

import socket
import subprocess
from collections.abc import Callable

PROBE_HOST = "1.1.1.1"
PROBE_PORT = 443
PROBE_TIMEOUT_SECONDS = 2.0
OFFLINE = "offline"
ONLINE = "online"
UNKNOWN = "unknown"


def default_route() -> bool | None:
    """True when the machine has a default route; None when the routing table cannot be read."""
    try:
        completed = subprocess.run(
            ["route", "-n", "get", "default"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.returncode == 0 and "gateway" in completed.stdout


def tcp_connect(host: str, port: int, timeout: float) -> None:
    """Open and close one TCP connection without sending any data."""
    with socket.create_connection((host, port), timeout=timeout):
        return None


def network_state(
    *,
    route: Callable[[], bool | None] = default_route,
    connect: Callable[[str, int, float], None] = tcp_connect,
) -> str:
    """Whether the machine can currently reach the internet: offline, online or unknown."""
    present = route()
    if present is None:
        return UNKNOWN
    if not present:
        return OFFLINE
    try:
        connect(PROBE_HOST, PROBE_PORT, PROBE_TIMEOUT_SECONDS)
    except OSError:
        return OFFLINE
    return ONLINE
