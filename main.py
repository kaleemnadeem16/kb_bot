"""Kept so `python main.py` still works; it is the same as `python -m kb_bot`."""

import sys

from kb_bot.cli import main

if __name__ == "__main__":
    sys.exit(main())
