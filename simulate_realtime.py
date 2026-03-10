"""
REMHART Digital Twin - Real-time Data Simulator
================================================
Sends 100 grid data points to the webhook endpoint at 1-second intervals.
Open your dashboard BEFORE running this — you'll see live updates on all 4 ML pages.

Usage:
    python3 simulate_realtime.py                     # normal conditions, 100 points
    python3 simulate_realtime.py --scenario sag      # voltage sag event
    python3 simulate_realtime.py --scenario imbalance # phase imbalance event
    python3 simulate_realtime.py --scenario overload  # overcurrent / overload
    python3 simulate_realtime.py --points 200        # 200 seconds of data
    python3 simulate_realtime.py --interval 0.5      # 2 points per second
"""

import requests
import time
import math
import random
import argparse
from datetime import datetime, timedelta

WEBHOOK_URL = "http://127.0.0.1:8001/api/webhook/data"

def generate_point(t: int, scenario: str) -> dict:
    """
    Generate a realistic grid data point.
    t = time step index (0-based)
    """
    # --- Base nominal values ---
    v_nom   = 230.0   # V  (Australian 230V nominal)
    i_nom   = 15.0    # A
    f_nom   = 50.0    # Hz
    p_nom   = 3450.0  # W per phase
    q_nom   = 240.0   # VAR per phase

    # --- Small natural variation (always present) ---
    noise_v = lambda: random.gauss(0, 0.3)
    noise_i = lambda: random.gauss(0, 0.05)
    noise_f = lambda: random.gauss(0, 0.005)

    # --- Scenario modifiers ---
    if scenario == "normal":
        v_a = v_nom + noise_v()
        v_b = v_nom + noise_v()
        v_c = v_nom + noise_v()
        i_a = i_nom + noise_i()
        i_b = i_nom + noise_i()
        i_c = i_nom + noise_i()
        freq = f_nom + noise_f()
        p_a, p_b, p_c = p_nom + random.gauss(0, 10), p_nom + random.gauss(0, 10), p_nom + random.gauss(0, 10)
        q_a, q_b, q_c = q_nom + random.gauss(0, 5), q_nom + random.gauss(0, 5), q_nom + random.gauss(0, 5)

    elif scenario == "sag":
        # Voltage sag progresses: drops from t=20 to t=40, recovers after
        if 20 <= t < 40:
            sag_depth = 0.85 - 0.005 * (t - 20)  # drops to ~75% of nominal at worst
        elif 40 <= t < 60:
            sag_depth = 0.75 + 0.01 * (t - 40)   # gradual recovery
        else:
            sag_depth = 1.0
        v_a = v_nom * sag_depth + noise_v()
        v_b = v_nom * sag_depth + noise_v()
        v_c = v_nom * sag_depth + noise_v()
        i_a = (i_nom / sag_depth) + noise_i()  # current rises as voltage drops
        i_b = (i_nom / sag_depth) + noise_i()
        i_c = (i_nom / sag_depth) + noise_i()
        freq = f_nom - (1 - sag_depth) * 0.3 + noise_f()  # slight frequency dip
        p_a = p_nom * sag_depth + random.gauss(0, 10)
        p_b = p_nom * sag_depth + random.gauss(0, 10)
        p_c = p_nom * sag_depth + random.gauss(0, 10)
        q_a = q_nom * (2 - sag_depth) + random.gauss(0, 5)  # reactive demand rises
        q_b = q_nom * (2 - sag_depth) + random.gauss(0, 5)
        q_c = q_nom * (2 - sag_depth) + random.gauss(0, 5)

    elif scenario == "imbalance":
        # Phase A progressively overloads from t=15 onward
        imb = min(1.0, (t - 15) / 30) if t >= 15 else 0.0
        v_a = v_nom - imb * 12 + noise_v()  # Phase A voltage drops
        v_b = v_nom + imb * 2 + noise_v()
        v_c = v_nom + imb * 2 + noise_v()
        i_a = i_nom + imb * 18 + noise_i()  # Phase A current spikes
        i_b = i_nom - imb * 2 + noise_i()
        i_c = i_nom - imb * 2 + noise_i()
        freq = f_nom + noise_f()
        p_a = p_nom + imb * 4000 + random.gauss(0, 10)
        p_b = p_nom - imb * 500 + random.gauss(0, 10)
        p_c = p_nom - imb * 500 + random.gauss(0, 10)
        q_a = q_nom + imb * 300 + random.gauss(0, 5)
        q_b = q_nom + random.gauss(0, 5)
        q_c = q_nom + random.gauss(0, 5)

    elif scenario == "overload":
        # Load ramps up steadily from t=10, exceeds limits around t=60
        load_factor = 1.0 + min(0.8, (max(0, t - 10) / 50) * 0.8)
        v_a = v_nom - (load_factor - 1) * 8 + noise_v()  # voltage droops under load
        v_b = v_nom - (load_factor - 1) * 8 + noise_v()
        v_c = v_nom - (load_factor - 1) * 8 + noise_v()
        i_a = i_nom * load_factor + noise_i()
        i_b = i_nom * load_factor + noise_i()
        i_c = i_nom * load_factor + noise_i()
        freq = f_nom - (load_factor - 1) * 0.2 + noise_f()  # frequency dips under heavy load
        p_a = p_nom * load_factor + random.gauss(0, 10)
        p_b = p_nom * load_factor + random.gauss(0, 10)
        p_c = p_nom * load_factor + random.gauss(0, 10)
        q_a = q_nom * load_factor + random.gauss(0, 5)
        q_b = q_nom * load_factor + random.gauss(0, 5)
        q_c = q_nom * load_factor + random.gauss(0, 5)

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Clamp frequency to schema limits (49.0 - 51.0)
    freq = max(49.01, min(50.99, freq))

    # Clamp voltages to schema limits (0 - 300)
    v_a = max(0.0, min(299.0, v_a))
    v_b = max(0.0, min(299.0, v_b))
    v_c = max(0.0, min(299.0, v_c))
    v_avg = round((v_a + v_b + v_c) / 3, 4)

    # Clamp currents (>= 0)
    i_a = max(0.0, i_a)
    i_b = max(0.0, i_b)
    i_c = max(0.0, i_c)
    i_avg = round((i_a + i_b + i_c) / 3, 4)

    return {
        "timestamp": datetime.now().isoformat(),
        "voltage": {
            "phaseA": round(v_a, 4),
            "phaseB": round(v_b, 4),
            "phaseC": round(v_c, 4),
            "average": v_avg
        },
        "current": {
            "phaseA": round(i_a, 4),
            "phaseB": round(i_b, 4),
            "phaseC": round(i_c, 4),
            "average": i_avg
        },
        "frequency": {
            "frequency_value": round(freq, 4)
        },
        "active_power": {
            "phaseA": round(p_a, 2),
            "phaseB": round(p_b, 2),
            "phaseC": round(p_c, 2),
            "total": round(p_a + p_b + p_c, 2)
        },
        "reactive_power": {
            "phaseA": round(q_a, 2),
            "phaseB": round(q_b, 2),
            "phaseC": round(q_c, 2),
            "total": round(q_a + q_b + q_c, 2)
        }
    }


def run(scenario: str, points: int, interval: float):
    print(f"\nREMHART Digital Twin — Real-time Simulator")
    print(f"  Scenario : {scenario}")
    print(f"  Points   : {points}")
    print(f"  Interval : {interval}s")
    print(f"  Target   : {WEBHOOK_URL}")
    print(f"\n  Open your dashboard at http://127.0.0.1:8000 before continuing.")
    print(f"  Press Ctrl+C to stop early.\n")
    input("  Press Enter to start...\n")

    success = 0
    failed  = 0

    for t in range(points):
        payload = generate_point(t, scenario)

        try:
            resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
            if resp.status_code == 200:
                result = resp.json()
                data_id = result.get("data_id", "?")
                ml_ran  = result.get("ml_inference_ran", False)
                clients = result.get("websocket_clients_notified", 0)
                print(
                    f"  [{t+1:3d}/{points}] OK  "
                    f"id={data_id}  "
                    f"V_avg={payload['voltage']['average']:.1f}V  "
                    f"I_avg={payload['current']['average']:.2f}A  "
                    f"f={payload['frequency']['frequency_value']:.3f}Hz  "
                    f"ML={'yes' if ml_ran else 'no '}  "
                    f"WS_clients={clients}"
                )
                success += 1
            else:
                print(f"  [{t+1:3d}/{points}] ERROR {resp.status_code}: {resp.text[:80]}")
                failed += 1

        except requests.exceptions.ConnectionError:
            print(f"  [{t+1:3d}/{points}] CANNOT CONNECT — is the backend running on port 8001?")
            failed += 1
            if failed >= 3:
                print("\n  Too many failures. Start the backend first:\n")
                print("    cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload\n")
                break

        except KeyboardInterrupt:
            print(f"\n  Stopped at point {t+1}.")
            break

        time.sleep(interval)

    print(f"\n  Done. Sent {success} points successfully, {failed} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REMHART real-time data simulator")
    parser.add_argument("--scenario", default="normal",
                        choices=["normal", "sag", "imbalance", "overload"],
                        help="Grid scenario to simulate")
    parser.add_argument("--points",   type=int,   default=100, help="Number of data points")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between points")
    args = parser.parse_args()

    run(args.scenario, args.points, args.interval)
