#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Determine default VENV_DIR when not explicitly set.
# If running as root via sudo, prefer the original caller's home venv.
if [ -z "${VENV_DIR:-}" ]; then
	if [ "$(id -u)" -eq 0 ] && [ -n "${SUDO_USER:-}" ]; then
		if getent passwd "$SUDO_USER" >/dev/null 2>&1; then
			SUDO_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
			VENV_DIR="$SUDO_HOME/.venv"
		else
			VENV_DIR="/home/$SUDO_USER/.venv"
		fi
	else
		VENV_DIR="$HOME/.venv"
	fi
fi

VENV_EXE="$VENV_DIR/bin/python"

# Enforce venv-only behavior: if running as root, prefer the original
# invoking user's venv (via SUDO_USER). If not root, use the current
# user's $HOME/.venv. If the venv is missing, exit with an error.
if [ "$(id -u)" -eq 0 ]; then
	if [ -n "${SUDO_USER:-}" ]; then
		if getent passwd "$SUDO_USER" >/dev/null 2>&1; then
			SUDO_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
			VENV_DIR="$SUDO_HOME/.venv"
			VENV_EXE="$VENV_DIR/bin/python"
		else
			echo "Cannot determine invoking user's home for SUDO_USER=$SUDO_USER" >&2
			exit 1
		fi
	else
		echo "Running as root without SUDO_USER; please set VENV_DIR to the desired venv." >&2
		exit 1
	fi
else
	VENV_DIR="${VENV_DIR:-$HOME/.venv}"
	VENV_EXE="$VENV_DIR/bin/python"
fi

# Require the virtualenv to exist and be executable; do not fallback.
if [ ! -x "$VENV_EXE" ]; then
	echo "Virtualenv python not found or not executable: $VENV_EXE" >&2
	echo "Create the venv at $VENV_DIR or set VENV_DIR to point to an existing venv." >&2
	exit 1
fi

# Activate the venv for environment consistency, then exec it.
if [ -f "$VENV_DIR/bin/activate" ]; then
	# shellcheck disable=SC1091
	. "$VENV_DIR/bin/activate"
fi

# Tunnel all CLI options through to the Python script so flags like
# --dry-run / --dry_run are forwarded unchanged.
exec "$VENV_EXE" motor_test.py "$@"