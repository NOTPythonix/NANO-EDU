# Server-side code (standalone)

- Run the server TUI (waits for the robot client to connect):
  - `python tui_server.py`

- Default bind: `0.0.0.0:8765` (override with env vars `ROBOT_BIND_HOST`, `ROBOT_BIND_PORT`).

The robot (client) side lives in the repo root and connects outbound to this server.
