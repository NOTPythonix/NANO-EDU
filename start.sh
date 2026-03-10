#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Use $VENV_DIR if set, else default to $HOME/.venv
VENV_DIR=~/.venv
VENV_EXE="$VENV_DIR/bin/python"


source $VENV_DIR/bin/activate


# Tunnel all CLI options through to the Python script so flags like
# --dry-run / --dry_run are forwarded unchanged.
exec "$VENV_EXE" motor_test.py "$@"