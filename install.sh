#!/usr/bin/env bash
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
	echo "Run this script as a normal user (not root). It will use sudo for apt installs." >&2
	exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WITH_INFERENCE=0

for arg in "$@"; do
	case "$arg" in
		--with-inference)
			WITH_INFERENCE=1
			;;
		--client-only)
			WITH_INFERENCE=0
			;;
		*)
			echo "Unknown option: $arg" >&2
			echo "Usage: bash install.sh [--with-inference|--client-only]" >&2
			exit 2
			;;
	esac
done

apt_packages=(
	python3
	python3-venv
	python3-pip
	python3-dev
	build-essential
	git
	git-lfs
	ffmpeg
	pkg-config
	libportaudio2
	portaudio19-dev
	libsndfile1
	libopenblas-dev
	liblapack-dev
	libjpeg-dev
	zlib1g-dev
	libglib2.0-0
	libgl1
	python3-opencv
	python3-rpi.gpio
)

python_packages=(
	rich
	keyboard
	sounddevice
	vosk
	numpy
)

if [ "$WITH_INFERENCE" -eq 1 ]; then
	python_packages+=(ultralytics)
fi

echo "[1/6] Updating apt package index..."
sudo apt-get update

echo "[2/6] Installing apt packages required for this project..."
sudo apt-get install -y --no-install-recommends "${apt_packages[@]}"

echo "[3/6] Preparing virtual environment at: $VENV_DIR"
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --no-cache-dir --upgrade pip setuptools wheel

echo "[4/6] Installing Python packages into the virtual environment..."
if [ "$WITH_INFERENCE" -eq 1 ]; then
	echo "Installing with inference packages (includes ultralytics/torch dependencies)."
else
	echo "Installing client-only packages (skips ultralytics/torch to keep disk usage low on Pi)."
fi
python -m pip install --no-cache-dir --prefer-binary "${python_packages[@]}"

echo "[5/6] Pulling Git LFS model files (if this is a git clone)..."
if [ -d "$ROOT_DIR/.git" ]; then
	git -C "$ROOT_DIR" lfs install
	git -C "$ROOT_DIR" lfs pull || true
fi

echo "[6/6] Installation complete."
echo
if [ "$WITH_INFERENCE" -eq 0 ]; then
	echo "Note: inference/server packages were skipped by default."
	echo "If you need local YOLO inference on this machine, run:"
	echo "  bash install.sh --with-inference"
	echo
fi
echo "Start the robot with:"
echo "  sudo bash start.sh --remote"
