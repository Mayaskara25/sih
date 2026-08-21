#!/usr/bin/env python3
"""Report which services are configured. Prints booleans only, never values.

Safe to run in front of an agent, in CI logs, or in a screen share.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import credentials  # noqa: E402

print(f"credentials file: {credentials.CRED_FILE}"
      f"  ({'present' if credentials.CRED_FILE.exists() else 'ABSENT'})")
try:
    st = credentials.status()
except PermissionError as e:
    print(f"  {e}"); raise SystemExit(2)
for svc, ok in st.items():
    print(f"  {svc:8} {'configured' if ok else 'NOT configured'}")
raise SystemExit(0 if all(st.values()) else 1)
