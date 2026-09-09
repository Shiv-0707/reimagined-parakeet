#!/usr/bin/env python3
"""
JARVIS - entry point.

Run:   python main.py          (serve on 0.0.0.0:8000)
       PORT=8080 python main.py
"""
from jarvis.server import main

if __name__ == "__main__":
    main()
