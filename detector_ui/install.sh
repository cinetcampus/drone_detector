#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

apt_hint() {
  local pyver
  pyver="$("$PYTHON" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo 3)"
  echo
  echo "Could not create a virtual environment automatically."
  echo "On Debian/Ubuntu install the venv package and re-run this script:"
  echo
  echo "    sudo apt install python${pyver}-venv"
  echo
}

create_venv() {
  # 1) Standard venv (works when ensurepip / python3-venv is available).
  if "$PYTHON" -m venv "$VENV_DIR" 2>/dev/null; then
    return 0
  fi

  echo "Standard venv failed (python3-venv/ensurepip missing). Trying fallbacks..."
  rm -rf "$VENV_DIR"

  # 2) virtualenv, if installed.
  if command -v virtualenv >/dev/null 2>&1; then
    echo "Using virtualenv"
    virtualenv -p "$PYTHON" "$VENV_DIR" && return 0
    rm -rf "$VENV_DIR"
  fi

  # 3) venv without pip, then bootstrap pip.
  if "$PYTHON" -m venv --without-pip "$VENV_DIR" 2>/dev/null; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    if python -m ensurepip --upgrade 2>/dev/null; then
      return 0
    fi
    echo "Bootstrapping pip with get-pip.py"
    local getpip
    getpip="$(mktemp)"
    if command -v curl >/dev/null 2>&1; then
      curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$getpip"
    elif command -v wget >/dev/null 2>&1; then
      wget -qO "$getpip" https://bootstrap.pypa.io/get-pip.py
    else
      rm -f "$getpip"
      deactivate || true
      return 1
    fi
    python "$getpip" && rm -f "$getpip" && return 0
    rm -f "$getpip"
    deactivate || true
  fi

  return 1
}

echo "Creating virtual environment in $VENV_DIR"
if ! create_venv; then
  apt_hint
  exit 1
fi

# Ensure the venv is active for the install step.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi

echo "Upgrading pip"
python -m pip install --upgrade pip

echo "Installing dependencies from requirements.txt"
python -m pip install -r requirements.txt

echo
echo "Done. Activate with: source $VENV_DIR/bin/activate"
echo "Run the server with: uvicorn web:app --reload"
