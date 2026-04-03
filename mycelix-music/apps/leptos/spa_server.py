# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import http.server
import os, sys

class SPAHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        file_path = os.path.join(os.getcwd(), path.lstrip('/'))
        if not os.path.exists(file_path) and '.' not in os.path.basename(path):
            self.path = '/index.html'
        return super().do_GET()

    def log_message(self, format, *args):
        pass  # Quiet

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dist'))
server = http.server.HTTPServer(('0.0.0.0', 8121), SPAHandler)
print("Serving on http://0.0.0.0:8121/")
server.serve_forever()
