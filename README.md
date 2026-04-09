# NANO-EDU
a school robot.

This robot will help in lots of schools around the globe!

## Run

### First-time setup

- On Linux / DietPi:
	- `bash install.sh`

### Client (Robot / Pi)

- Run the robot TUI:
	- `bash start.sh`

The client can optionally connect to the server-side TUI for remote driving and server-side autonomy.

### Server (Laptop / GPU box)

- Run the server TUI (in a separate folder so it stays standalone):
	- `python server/tui_server.py`

The server TUI will refuse to proceed until the robot connects.
