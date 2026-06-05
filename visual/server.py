import http.server
import json
import os
import subprocess
import sys

PORT = 8080
BINARY_PATH = os.path.join(os.path.dirname(__file__), "..", "build", "MathVisualizer")

class KairoMathHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_POST(self):
        if not self.path.startswith("/api/"):
            self.send_error(404, "API endpoint not found")
            return

        endpoint = self.path[5:] # Remove "/api/"
        
        # Read content length
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length).decode('utf-8')
        
        try:
            payload = json.loads(post_data) if post_data else {}
        except Exception as e:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": f"Invalid JSON: {str(e)}"}).encode())
            return

        # Prepare input string to pass to the C++ CLI
        cpp_input = self.format_cpp_input(endpoint, payload)
        if cpp_input is None:
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "message": f"Unsupported API endpoint: {endpoint}"}).encode())
            return

        # Execute the C++ executable in API mode
        if not os.path.exists(BINARY_PATH):
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "error", 
                "message": f"C++ visualizer binary not found at {BINARY_PATH}. Please run: cmake --build build"
            }).encode())
            return

        try:
            proc = subprocess.Popen(
                [BINARY_PATH, "--api"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = proc.communicate(input=cpp_input, timeout=5)
            
            if proc.returncode != 0:
                response = {"status": "error", "message": f"C++ execution failed: {stderr.strip()}"}
            else:
                try:
                    response = json.loads(stdout.strip())
                except Exception:
                    response = {"status": "error", "message": f"Failed to parse C++ stdout as JSON: {stdout}"}
        except subprocess.TimeoutExpired:
            proc.kill()
            response = {"status": "error", "message": "C++ execution timed out"}
        except Exception as e:
            response = {"status": "error", "message": f"Execution error: {str(e)}"}

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def format_cpp_input(self, endpoint, payload):
        """Converts JSON payload to the text-based protocol expected by kairo_math_cli."""
        if endpoint == "project":
            # ax, ay, bx, by
            return f"project {payload.get('ax', 0.0)} {payload.get('ay', 0.0)} {payload.get('bx', 0.0)} {payload.get('by', 0.0)}\n"
        
        elif endpoint == "refract":
            # ix, iy, iz, nx, ny, nz, eta
            return f"refract {payload.get('ix', 0.0)} {payload.get('iy', 0.0)} {payload.get('iz', 0.0)} {payload.get('nx', 0.0)} {payload.get('ny', 0.0)} {payload.get('nz', 0.0)} {payload.get('eta', 1.0)}\n"
        
        elif endpoint == "transform":
            # tx, ty, tz, rx, ry, rz, sx, sy, sz
            return f"transform {payload.get('tx', 0.0)} {payload.get('ty', 0.0)} {payload.get('tz', 0.0)} {payload.get('rx', 0.0)} {payload.get('ry', 0.0)} {payload.get('rz', 0.0)} {payload.get('sx', 1.0)} {payload.get('sy', 1.0)} {payload.get('sz', 1.0)}\n"
            
        elif endpoint == "solve":
            matrix = payload.get("matrix", [])
            b = payload.get("b", [])
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            output = f"solve {rows} {cols}\n"
            for r in matrix:
                output += " ".join(map(str, r)) + "\n"
            output += " ".join(map(str, b)) + "\n"
            return output
            
        elif endpoint == "decompose":
            method = payload.get("method", "lu")
            matrix = payload.get("matrix", [])
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            output = f"decompose {method} {rows} {cols}\n"
            for r in matrix:
                output += " ".join(map(str, r)) + "\n"
            return output
            
        elif endpoint == "eigen":
            matrix = payload.get("matrix", [])
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            output = f"eigen {rows} {cols}\n"
            for r in matrix:
                output += " ".join(map(str, r)) + "\n"
            return output
            
        elif endpoint == "svd":
            matrix = payload.get("matrix", [])
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            output = f"svd {rows} {cols}\n"
            for r in matrix:
                output += " ".join(map(str, r)) + "\n"
            return output
            
        elif endpoint == "pca":
            matrix = payload.get("matrix", [])
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            output = f"pca {rows} {cols}\n"
            for r in matrix:
                output += " ".join(map(str, r)) + "\n"
            return output
            
        elif endpoint == "regression":
            matrix = payload.get("matrix", [])
            y = payload.get("y", [])
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            output = f"regression {rows} {cols}\n"
            for r in matrix:
                output += " ".join(map(str, r)) + "\n"
            output += " ".join(map(str, y)) + "\n"
            return output
            
        elif endpoint in ["determinant", "inverse", "rank", "condition_number"]:
            matrix = payload.get("matrix", [])
            rows = len(matrix)
            cols = len(matrix[0]) if rows > 0 else 0
            
            output = f"{endpoint} {rows} {cols}\n"
            for r in matrix:
                output += " ".join(map(str, r)) + "\n"
            return output
            
        return None

def run_server():
    server_address = ('', PORT)
    httpd = http.server.HTTPServer(server_address, KairoMathHTTPHandler)
    print(f"=========================================================")
    print(f"      KairoMath Interactive Dashboard Server             ")
    print(f"      Running on http://localhost:{PORT}                 ")
    print(f"=========================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        sys.exit(0)

if __name__ == "__main__":
    run_server()
