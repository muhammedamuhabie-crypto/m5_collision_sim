"""
/api/montecarlo  —  runs many randomized two-car collisions at once
using NumPy's vectorized math, and returns a histogram of how much
kinetic energy was lost as a percentage of the total.

This is the kind of task Python is genuinely better suited for than
hand-written JavaScript: generating thousands of random trials and
summarizing them statistically in one vectorized pass.
"""

from http.server import BaseHTTPRequestHandler
import json
import numpy as np


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body) if body else {}

            trials = min(int(data.get("trials", 500)), 5000)
            m_lo, m_hi = data.get("mass_range", [800, 3000])
            v_lo, v_hi = data.get("speed_range", [5, 35])

            rng = np.random.default_rng()
            m1 = rng.uniform(m_lo, m_hi, trials)
            m2 = rng.uniform(m_lo, m_hi, trials)
            v1 = rng.uniform(v_lo, v_hi, trials)
            v2 = np.zeros(trials)  # target car parked, as in a typical rear-end scenario
            e = rng.uniform(0, 1, trials)

            v1f = ((m1 - e * m2) * v1 + (1 + e) * m2 * v2) / (m1 + m2)
            v2f = ((m2 - e * m1) * v2 + (1 + e) * m1 * v1) / (m1 + m2)

            ke0 = 0.5 * m1 * v1 ** 2 + 0.5 * m2 * v2 ** 2
            ke1 = 0.5 * m1 * v1f ** 2 + 0.5 * m2 * v2f ** 2
            loss_pct = np.divide(100 * (ke0 - ke1), ke0, out=np.zeros_like(ke0), where=ke0 > 0)

            counts, edges = np.histogram(loss_pct, bins=20, range=(0, 100))

            result = {
                "trials": trials,
                "mean_loss_pct": round(float(np.mean(loss_pct)), 2),
                "median_loss_pct": round(float(np.median(loss_pct)), 2),
                "std_loss_pct": round(float(np.std(loss_pct)), 2),
                "bin_edges": [round(float(x), 1) for x in edges],
                "counts": [int(c) for c in counts],
            }
            self._send(200, result)
        except Exception as ex:
            self._send(400, {"error": str(ex)})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _send(self, code, payload):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode())
