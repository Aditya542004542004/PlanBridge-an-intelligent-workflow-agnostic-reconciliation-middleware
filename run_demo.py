"""
run_demo.py
=============
PlanBridge — Intelligent Planning-to-Execution Reconciliation Middleware
SIH 2026 | Problem Statement SIH26122 (Oil India Limited)

PHASE 8 DELIVERABLE — One-Click Presentation Demo Launcher
------------------------------------------------------------------
Single command to go from a fresh checkout to a running, browser-open demo:

    python run_demo.py

What it does, in order:
    1. Verifies the core Python dependencies are importable (fails fast
       with a clear instruction, rather than letting uvicorn crash
       confusingly three steps later).
    2. Ensures data/activities.json and data/dprs.json exist, running
       synthetic_generator.py automatically if they don't.
    3. Starts the FastAPI backend (`uvicorn main:app --port 8000`) as a
       background subprocess, waits for it to actually report healthy
       (not just "process started"), and keeps it running for the
       lifetime of this script.
    4. Opens frontend/index.html in the default web browser.
    5. Prints a clean presentation banner, then blocks until Ctrl+C —
       at which point it cleanly terminates the backend subprocess rather
       than leaving an orphaned server running in the background.

Flags
-----
    --port 8000         Port to run the backend on.
    --no-browser        Don't auto-open the browser (useful headless/CI).
    --skip-checks       Skip dependency/data checks (faster iteration once
                        you already know the environment is set up).
    --yes               Auto-confirm any prompts (e.g. auto-install missing
                        dependencies) instead of asking interactively.
"""

from __future__ import annotations

import argparse
import atexit
import importlib
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent
ACTIVITIES_PATH = BASE_DIR / "data" / "activities.json"
DPRS_PATH = BASE_DIR / "data" / "dprs.json"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"
REQUIREMENTS_PATH = BASE_DIR / "requirements.txt"

# Maps the pip package name (as it appears in requirements.txt) to the
# actual importable module name, where they differ.
REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "spacy": "spacy",
    "sentence-transformers": "sentence_transformers",
    "torch": "torch",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "jinja2": "jinja2",
}

HEALTH_CHECK_TIMEOUT_S = 30
HEALTH_CHECK_INTERVAL_S = 0.5


def _print_step(step_num: int, total: int, message: str) -> None:
    print(f"\n[{step_num}/{total}] {message}")


def _importable(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_dependencies(auto_yes: bool) -> bool:
    """
    Verify every required package is importable. If any are missing,
    prints exactly what's missing and either prompts to auto-install (via
    `pip install -r requirements.txt`) or exits with clear instructions.

    Returns True if the environment is (now) ready to proceed, False if
    the caller should abort.
    """
    missing = [pip_name for pip_name, module_name in REQUIRED_PACKAGES.items() if not _importable(module_name)]

    if not missing:
        print("  All required dependencies are installed.")
        return True

    print(f"  Missing dependencies: {', '.join(missing)}")

    if not REQUIREMENTS_PATH.exists():
        print(f"  ERROR: {REQUIREMENTS_PATH} not found -- cannot auto-install. Install manually and re-run.")
        return False

    if not auto_yes:
        answer = input(f"  Install missing dependencies now via 'pip install -r {REQUIREMENTS_PATH.name}'? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("  Skipped. Install manually with:")
            print(f"      pip install -r {REQUIREMENTS_PATH}")
            return False

    print(f"  Installing from {REQUIREMENTS_PATH} (this may take a few minutes for torch/spacy)...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_PATH)],
        cwd=str(BASE_DIR),
    )
    if result.returncode != 0:
        print("  ERROR: pip install failed. See output above.")
        return False

    still_missing = [pip_name for pip_name, module_name in REQUIRED_PACKAGES.items() if not _importable(module_name)]
    if still_missing:
        print(f"  ERROR: still missing after install: {', '.join(still_missing)}")
        return False

    print("  Dependencies installed successfully.")
    return True


def ensure_data_generated() -> bool:
    """
    Ensures data/activities.json and data/dprs.json exist, generating them
    via synthetic_generator.py if not. Returns True if the data is ready.
    """
    if ACTIVITIES_PATH.exists() and DPRS_PATH.exists():
        print(f"  Found existing dataset: {ACTIVITIES_PATH.name}, {DPRS_PATH.name}")
        return True

    generator_path = BASE_DIR / "synthetic_generator.py"
    if not generator_path.exists():
        print(f"  ERROR: dataset missing and {generator_path} not found -- cannot auto-generate.")
        return False

    print("  Dataset not found -- running synthetic_generator.py to create it...")
    result = subprocess.run([sys.executable, str(generator_path)], cwd=str(BASE_DIR))
    if result.returncode != 0:
        print("  ERROR: synthetic_generator.py failed. See output above.")
        return False

    if not (ACTIVITIES_PATH.exists() and DPRS_PATH.exists()):
        print("  ERROR: generator ran but expected output files still don't exist.")
        return False

    print("  Dataset generated successfully.")
    return True


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def _wait_for_health(port: int, timeout_s: float) -> bool:
    """Poll the backend's health-check endpoint until it responds or the
    timeout elapses. Uses only the standard library (urllib) so this
    launcher has no import dependency on the app it's launching."""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL_S)
    return False


def start_backend(port: int) -> Optional[subprocess.Popen]:
    """
    Starts `uvicorn main:app --port <port>` as a background subprocess and
    waits for it to report healthy. Returns the Popen handle on success,
    or None on failure (port already in use, server crashed on startup,
    or it never became healthy within the timeout).
    """
    if not _port_is_free(port):
        print(f"  Port {port} is already in use -- another instance may already be running.")
        print(f"  If that's expected, just open {FRONTEND_PATH} in your browser; it will connect automatically.")
        return None

    main_path = BASE_DIR / "main.py"
    if not main_path.exists():
        print(f"  ERROR: {main_path} not found.")
        return None

    print(f"  Starting backend: uvicorn main:app --port {port}")
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(port)],
        cwd=str(BASE_DIR),
    )

    print(f"  Waiting for backend to become healthy (up to {HEALTH_CHECK_TIMEOUT_S}s -- first run loads "
          f"spaCy/Sentence-BERT/torch, this can take a while)...")
    if _wait_for_health(port, HEALTH_CHECK_TIMEOUT_S):
        print(f"  Backend is healthy at http://127.0.0.1:{port}")
        return process

    if process.poll() is not None:
        print(f"  ERROR: backend process exited early with code {process.returncode}. Check the output above.")
    else:
        print(f"  WARNING: backend did not report healthy within {HEALTH_CHECK_TIMEOUT_S}s -- it may still be "
              f"loading (e.g. downloading the Sentence-BERT model on first run). Leaving it running; the "
              f"frontend will connect automatically once it's ready, or fall back to offline demo mode.")
    return process


def open_browser(url: str) -> None:
    print(f"  Opening {url}")
    try:
        webbrowser.open(url)
    except Exception as exc:  # webbrowser can fail on headless/misconfigured systems
        print(f"  Could not auto-open a browser ({exc}). Open this URL manually:\n      {url}")


def print_banner(port: int, backend_running: bool) -> None:
    status = f"http://127.0.0.1:{port}" if backend_running else "OFFLINE (frontend will use built-in demo mode)"
    banner = f"""
================================================================================
  PLANBRIDGE -- OIL RECONCILIATION CONSOLE
  SIH 2026 | Problem Statement SIH26122 | Oil India Limited
================================================================================

  Backend API : {status}
  API docs    : http://127.0.0.1:{port}/docs
  Frontend    : {FRONTEND_PATH}

  --------------------------------------------------------------------------
  QUICK DEMO SCRIPT
  --------------------------------------------------------------------------
  1. Dashboard tab    -- point out the 4 headline metrics, especially the
                         False Auto-Accept Rate.
  2. Simulator tab    -- click "Easy - Auto-Match", then "Run Engine".
                         Walk through the 4-step pipeline trace live.
  3. Simulator tab    -- click "Medium - Queue", run it, then switch to the
                         Planner Review Queue tab and Accept the match.
  4. Audit Log tab    -- show the SHA-256 hash-chained ledger; mention it
                         re-verifies integrity on every read.
  5. XER Exporter tab -- show the live .XER preview, click Download.

  --------------------------------------------------------------------------
  Press Ctrl+C in this terminal to stop the backend server when you're done.
  --------------------------------------------------------------------------
"""
    print(banner)


def main() -> int:
    parser = argparse.ArgumentParser(description="PlanBridge one-click demo launcher.")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI backend on (default: 8000).")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open the frontend in a browser.")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency/data checks.")
    parser.add_argument("--yes", action="store_true", help="Auto-confirm prompts (e.g. dependency install).")
    args = parser.parse_args()

    total_steps = 4
    process: Optional[subprocess.Popen] = None

    if not args.skip_checks:
        _print_step(1, total_steps, "Checking Python dependencies...")
        if not check_dependencies(args.yes):
            return 1

        _print_step(2, total_steps, "Checking synthetic benchmark dataset...")
        if not ensure_data_generated():
            return 1
    else:
        print("Skipping dependency/data checks (--skip-checks).")

    _print_step(3, total_steps, "Starting backend server...")
    process = start_backend(args.port)
    backend_healthy = process is not None and process.poll() is None

    if process is not None:
        def _cleanup() -> None:
            if process.poll() is None:
                print("\nShutting down PlanBridge backend...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        atexit.register(_cleanup)

    _print_step(4, total_steps, "Opening frontend...")
    if not FRONTEND_PATH.exists():
        print(f"  WARNING: {FRONTEND_PATH} not found -- nothing to open.")
    elif not args.no_browser:
        open_browser(FRONTEND_PATH.as_uri())
    else:
        print(f"  --no-browser set; open manually: {FRONTEND_PATH.as_uri()}")

    print_banner(args.port, backend_healthy)

    if process is None:
        print("Backend is not running (see warnings above) -- the frontend will operate in offline demo mode.")
        return 0

    try:
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print(f"\nBackend process exited unexpectedly (code {process.returncode}). Stopping launcher.")
                return 1
    except KeyboardInterrupt:
        # Swallow a second/third rapid Ctrl+C too (an impatient presenter
        # mashing the key is a very real scenario) -- the actual cleanup
        # is handled by the atexit hook registered above regardless of how
        # many interrupts arrive here; this block only needs to print a
        # clean message without letting a repeat KeyboardInterrupt during
        # that print turn into an ugly traceback.
        try:
            print("\nStopping PlanBridge demo (Ctrl+C received)...")
        except KeyboardInterrupt:
            pass
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # Extremely rare timing window (interrupt arriving between main()
        # returning and sys.exit() being called) -- still exit cleanly.
        sys.exit(0)
