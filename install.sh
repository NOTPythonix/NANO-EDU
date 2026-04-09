#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this script as a normal user (not root). It will use sudo for apt installs." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[1/6] Updating apt package index..."
sudo apt-get update

echo "[2/6] Installing apt packages required for this project..."
# Core runtime/build tools + audio + OpenCV + GPIO support for Raspberry Pi.
sudo apt-get install -y --no-install-recommends \
  python3 \
  python3-venv \
  python3-pip \
  python3-dev \
  build-essential \
  git \
  git-lfs \
  ffmpeg \
  pkg-config \
  libportaudio2 \
  portaudio19-dev \
  libsndfile1 \
  libatlas-base-dev \
  libopenblas-dev \
  liblapack-dev \
  libjpeg-dev \
  zlib1g-dev \
  libglib2.0-0 \
  libgl1 \
  python3-opencv \
  python3-rpi.gpio

echo "[3/6] Preparing virtual environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Installing Python packages into the virtual environment..."
python -m pip install --prefer-binary \
  rich \
  keyboard \
  sounddevice \
  vosk \
  numpy \
  ultralytics

echo "[5/6] Pulling Git LFS model files (if this is a git clone)..."
if [ -d "$ROOT_DIR/.git" ]; then
  git -C "$ROOT_DIR" lfs install
  git -C "$ROOT_DIR" lfs pull || true
fi

echo "[6/6] Installation complete."
echo
echo "Start the robot with:"
echo "  sudo bash start.sh"
