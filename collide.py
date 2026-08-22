"""
/api/collide  —  event-driven 1D multi-car collision solver.

Given a list of cars (mass, velocity, position along the track) and a
coefficient of restitution e (0 = perfectly inelastic, 1 = perfectly
elastic), this walks forward in time, finds the next pair of adjacent
cars about to touch, resolves that collision with the standard
impulse-momentum formulas, and repeats — producing a full chain-reaction
timeline (useful for pile-ups of 3+ cars).

This mirrors the JS physics used for the live 3D animation, so the
frontend can call this endpoint to get a "server verified" result.
"""

from http.server import BaseHTTPRequestHandler
import json


def resolve_pair(m1, v1, m2, v2, e):
    """Standard 1D collision with restitution e."""
    v1f = ((m1 - e * m2) * v1 + (1 + e) * m2 * v2) / (m1 + m2)
    v2f = ((m2 - e * m1) * v2 + (1 + e) * m1 * v1) / (m1 + m2)
    return v1f, v2f


def simulate_chain(cars, e, car_length, max_time=10.0, max_events=25):
    cars = sorted(cars, key=lambda c: c["x"])
    n = len(cars)
    m = [c["mass"] for c in cars]
    v = [c["v"] for c in cars]
    x = [c["x"] for c in cars]

    p0 = sum(m[i] * v[i] for i in range(n))
    ke0 = sum(0.5 * m[i] * v[i] ** 2 for i in range(n))

    events = []
    t_total = 0.0

    for _ in range(max_events):
        best_t, best_i = None, None
        for i in range(n - 1):
            gap = (x[i + 1] - x[i]) - car_length
            closing = v[i] - v[i + 1]
            if gap <= 1e-6 and closing > 0:
                t = 0.0
            elif closing > 1e-9 and gap > 0:
                t = gap / closing
            else:
                continue
            if best_t is None or t < best_t:
                best_t, best_i = t, i

        if best_t is None or t_total + best_t > max_time:
            remain = max(0.0, max_time - t_total)
            x = [x[i] + v[i] * remain for i in range(n)]
            t_total = max_time
            break

        x = [x[i] + v[i] * best_t for i in range(n)]
        t_total += best_t

        i = best_i
        v1f, v2f = resolve_pair(m[i], v[i], m[i + 1], v[i + 1], e)
        mid = (x[i] + x[i + 1]) / 2
        x[i] = mid - car_length / 2
        x[i + 1] = mid + car_length / 2

        events.append({
            "time": round(t_total, 3),
            "pair": [i, i + 1],
            "position": round(mid, 2),
            "v_before": [round(v[i], 2), round(v[i + 1], 2)],
            "v_after": [round(v1f, 2), round(v2f, 2)],
        })
        v[i], v[i + 1] = v1f, v2f

    p1 = sum(m[i] * v[i] for i in range(n))
    ke1 = sum(0.5 * m[i] * v[i] ** 2 for i in range(n))

    return {
        "events": events,
        "final_v": [round(vv, 2) for vv in v],
        "final_x": [round(xx, 2) for xx in x],
        "momentum_before": round(p0, 1),
        "momentum_after": round(p1, 1),
        "ke_before": round(ke0, 1),
        "ke_after": round(ke1, 1),
        "energy_lost": round(ke0 - ke1, 1),
        "energy_lost_pct": round(100 * (ke0 - ke1) / ke0, 1) if ke0 > 0 else 0.0,
    }


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            data = json.loads(body) if body else {}
            cars = data.get("cars", [])
            e = float(data.get("e", 0.0))
            car_length = float(data.get("car_length", 4.4))
            if not cars:
                raise ValueError("No cars provided")
            result = simulate_chain(cars, e, car_length)
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
