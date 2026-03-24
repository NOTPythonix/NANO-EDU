from __future__ import annotations

# Client-side entrypoint (runs on the robot / Pi).
# Kept as a thin wrapper so we can have explicit server/client TUIs.

import tui


def main() -> int:
    return tui.main()


if __name__ == "__main__":
    raise SystemExit(main())
