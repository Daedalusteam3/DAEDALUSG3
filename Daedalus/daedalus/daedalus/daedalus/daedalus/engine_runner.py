# daedalus/engine_runner.py
import subprocess
import sys
from pathlib import Path

# Path
SCRIPT_PATH = Path(r"E:\Uni\Proyecto_jf\intelligence_engine.py")

def run_optimization(custom_params=None):
    """
    Launches intelligence_engine.py 
    as a child process and returns stdout / stderr / returncode.
    """
    if not SCRIPT_PATH.exists():
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Script no encontrado en {SCRIPT_PATH}",
        }

    cwd = SCRIPT_PATH.parent

    if custom_params is None:
        custom_params = {}

    args = []
    for k, v in custom_params.items():
        args.append(f"--{k}={v}")

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=str(SCRIPT_PATH.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }