"""End-to-end validation for main.py (not part of the graded entrypoint).

Builds one synthetic case per comparable launch (known = its hour-1
print, comparables = the other 37, forecast = hours 2..24), runs
main.py on each through the same environment the grader uses, and
scores the pinball loss on the log scale against that launch's real
release-day curve.  Also runs the house baseline the same way, runs
main.py on the kit's four practice cases, and checks determinism
(two runs, byte-identical output) and wall time.

Usage:  python3 tools/validate.py [kit-dir]
"""

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
ROOT = Path(__file__).resolve().parent.parent
KIT = Path(sys.argv[1] if len(sys.argv) > 1 else
           os.environ.get("KIT_DIR", str(Path.home() / "workspace" / "makefun" /
                                         "gta6-twitch-ccu-forecast-01-kit")))


def parse(t):
    return datetime.fromisoformat(t.replace("Z", "+00:00"))


def pinball(y, q, tau):
    d = math.log(max(1.0, y)) - math.log(max(q, 1e-9))
    return 2 * max(tau * d, (tau - 1) * d)


def run_program(prog: Path, in_dir: Path, out_dir: Path) -> float:
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = str(in_dir)
    env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
    start = time.time()
    subprocess.run([sys.executable, str(prog)], env=env, check=True,
                   capture_output=True, timeout=120)
    return time.time() - start


def score_case(out_file: Path, actual: dict) -> float:
    pred = json.loads(out_file.read_text())
    total, n = 0.0, 0
    for row in pred["hours"]:
        for tau, q in zip(LEVELS, row["q"]):
            total += pinball(actual[row["hour"]], q, tau)
            n += 1
    return total / n


def release_day(comparable):
    ds = parse(comparable["day_start"])
    day = {}
    for p in comparable["hourly"]:
        t = parse(p["from"])
        if ds <= t < ds + timedelta(days=1):
            day[t.hour + 1] = p["viewers"]
    return day if len(day) == 24 and day.get(1, 0) > 0 else None


def main():
    case0 = json.loads((KIT / "input" / "p00.in").read_text())
    comparables = [c for c in case0["comparables"]
                   if "Grand Theft Auto VI" not in c["name"]]
    usable = [(c, release_day(c)) for c in comparables]
    usable = [(c, d) for c, d in usable if d is not None]
    print(f"comparables with complete release-day curves: {len(usable)}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # --- leave-one-out over the comparables, through main.py itself ---
        losses, base_losses = [], []
        for held, day in usable:
            others = [c for c, _ in usable if c is not held]
            case = {"schema": "makefun/twitch-ccu-case/v1", "id": "loo",
                    "game": {"name": held["name"]},
                    "day_start": held["day_start"],
                    "checkpoint": held["day_start"][:11] + "01:00:00Z",
                    "levels": LEVELS,
                    "known": [{"hour": 1, "from": held["day_start"],
                               "until": held["day_start"][:11] + "01:00:00Z",
                               "viewers": day[1], "channels": 0, "reads": 1}],
                    "forecast": [{"hour": h} for h in range(2, 25)],
                    "history": [], "history_hourly": [], "streams": None,
                    "categories_hourly": None, "comparables": others,
                    "notes": "leave-one-out validation case"}
            in_dir, out_dir = tmp / "in", tmp / "out"
            in_dir.mkdir(exist_ok=True)
            (in_dir / "loo.in").write_text(json.dumps(case))
            run_program(ROOT / "main.py", in_dir, out_dir)
            losses.append(score_case(out_dir / "loo.out", day))
            run_program(KIT / "starter" / "main.py", in_dir, out_dir)
            base_losses.append(score_case(out_dir / "loo.out", day))
        print(f"LOO all usable:        model {sum(losses)/len(losses):.4f}  "
              f"baseline {sum(base_losses)/len(base_losses):.4f}")
        roll_names = {c["name"] for c, d in usable
                    if "rolling" in c["unlock"].lower()
                    or "midnight local" in c["unlock"].lower()}
        idx = [i for i, (c, _) in enumerate(usable) if c["name"] in roll_names]
        print(f"LOO rolling launches:  model "
              f"{sum(losses[i] for i in idx)/len(idx):.4f}  baseline "
              f"{sum(base_losses[i] for i in idx)/len(idx):.4f}  (n={len(idx)})")

        # --- practice cases: run main.py, report implied peaks ---
        in_dir = KIT / "input"
        out_dir = tmp / "pout"
        run_program(ROOT / "main.py", in_dir, out_dir)
        for p in sorted(in_dir.glob("p*.in")):
            case = json.loads(p.read_text())
            pred = json.loads((out_dir / f"{p.stem}.out").read_text())
            med = {r["hour"]: r["q"][2] for r in pred["hours"]}
            peak_h = max(med, key=med.get)
            h1 = next(k["viewers"] for k in case["known"] if k["hour"] == 1)
            print(f"{p.stem} {case['game']['name']}: h1={h1:,} "
                  f"implied peak={med[peak_h]:,.0f} @h{peak_h} "
                  f"(ratio {med[peak_h]/h1:.2f})")

        # --- determinism: two runs on the practice cases, byte-identical ---
        out2 = tmp / "pout2"
        run_program(ROOT / "main.py", in_dir, out2)
        same = all((out_dir / f).read_bytes() == (out2 / f).read_bytes()
                   for f in os.listdir(out_dir))
        print("determinism (two runs byte-identical):", same)

        # --- timing on one case ---
        single_in, single_out = tmp / "sin", tmp / "sout"
        single_in.mkdir()
        (single_in / "p00.in").write_text((KIT / "input" / "p00.in").read_text())
        dt = run_program(ROOT / "main.py", single_in, single_out)
        print(f"wall time, one case: {dt:.2f}s (cap 8s)")


if __name__ == "__main__":
    main()
