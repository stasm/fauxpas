#!/usr/bin/env python3
"""Local dev server with cross-origin isolation headers for high-res timers."""

import http.server

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        super().end_headers()

print("Serving on http://localhost:8000 (cross-origin isolated)")
http.server.HTTPServer(("", 8000), Handler).serve_forever()
