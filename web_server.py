"""TrustLens Web Application Entry Point.

Runs the FastAPI web server on http://127.0.0.1:8000 (or specified port).
Usage:
    python web_server.py [--port 8000]
    python -m webapp
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from webapp.server import app

def main():
    parser = argparse.ArgumentParser(description="TrustLens Web Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    print(f"Starting TrustLens Web Server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
