"""Static server + API proxy for the EasyNotes frontend.

Serves ``index.html`` and forwards every ``/notes`` request to the Express
app, so the browser only ever talks to one origin and CORS never comes up --
the notes app itself ships no CORS headers, and this avoids modifying it.

Standard library only; no new dependencies.

Usage:
    python frontend/serve.py [--port 8080] [--api http://localhost:3000]
"""

from __future__ import annotations

import argparse
import http.server
import urllib.error
import urllib.request
from pathlib import Path

FRONTEND_DIR = Path(__file__).resolve().parent

# Headers that describe the hop, not the payload; forwarding them corrupts
# the relayed response.
HOP_HEADERS = {"connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade"}


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serves frontend files; proxies /notes* to the API."""

    api_base = "http://localhost:3000"  # overwritten from --api in main()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    # -- routing ----------------------------------------------------------

    def _is_api(self) -> bool:
        return self.path == "/notes" or self.path.startswith("/notes/")

    def do_GET(self):
        if self._is_api():
            self._proxy()
        else:
            super().do_GET()

    def do_POST(self):
        if self._is_api():
            self._proxy()
        else:
            self.send_error(405)

    def do_PUT(self):
        if self._is_api():
            self._proxy()
        else:
            self.send_error(405)

    def do_DELETE(self):
        if self._is_api():
            self._proxy()
        else:
            self.send_error(405)

    # -- proxy ------------------------------------------------------------

    def _proxy(self) -> None:
        """Relay the current request to the API and stream the reply back."""
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        request = urllib.request.Request(
            self.api_base + self.path,
            data=body,
            method=self.command,
        )
        content_type = self.headers.get("Content-Type")
        if content_type:
            request.add_header("Content-Type", content_type)

        try:
            with urllib.request.urlopen(request, timeout=30) as upstream:
                self._relay(upstream.status, upstream.headers, upstream.read())
        except urllib.error.HTTPError as exc:
            # 4xx/5xx from Express still carry a JSON body worth relaying.
            self._relay(exc.code, exc.headers, exc.read())
        except OSError as exc:
            self.send_error(
                502, f"API at {self.api_base} is unreachable: {exc}"
            )

    def _relay(self, status: int, headers, payload: bytes) -> None:
        self.send_response(status)
        for name, value in headers.items():
            if name.lower() not in HOP_HEADERS:
                self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args) -> None:
        print(f"  {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default 8080).")
    parser.add_argument(
        "--api",
        default="http://localhost:3000",
        help="Base URL of the running notes API (default http://localhost:3000).",
    )
    args = parser.parse_args()

    Handler.api_base = args.api.rstrip("/")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"EasyNotes frontend: http://localhost:{args.port}")
    print(f"proxying /notes* -> {Handler.api_base}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
