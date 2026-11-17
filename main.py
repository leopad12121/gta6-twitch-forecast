"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Entry for make.fun bounty gta6-twitch-ccu-forecast-01.

What it does
------------
Reads every ``<id>.in`` case in the input directory and writes
``<id>.out`` to the output directory::

    {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ..., {"hour": 24, ...}]}

Each case gives the Twitch viewer count of the Grand Theft Auto VI
category up to the end of the first hour of release day 2026-11-19 (UTC)
-- the ``known`` row for hour 1 -- and asks for five quantiles of that
hour's count for each of hours 2..24.  The score is the mean pinball loss
on the log scale, so the quantiles reported here are the ones the model
really believes.

Model (all of it fitted on the 38 past launches the kit ships as
``comparables`` -- no data from after the checkpoint is used anywhere)
-----------------------------------------------------------------------
For each hour h = 2..24 the model forecasts the log-ratio
``r_h = ln(viewers_h / viewers_1)`` and reports

    q_tau(h) = p1 * exp(mu_h + z_tau * sigma_h)

where p1 is hour 1's print, z_tau is the standard normal quantile at
each level (0.05, 0.25, 0.5, 0.75, 0.95) and (mu_h, sigma_h) blend two
views of the same 38 comparable launch days:

A. Rolling-midnight regression.  On the 12 comparables whose unlock was
   a rolling midnight local (the unlock GTA VI itself has: New Zealand
   first, 2026-11-18T11:00Z), fit per hour an OLS line of the log-ratio
   on ln(p1).  Big first hours mean the world was already watching, so
   the curve stays flat (slope about -0.4 to -0.5 from hour 6 on);
   small first hours mean the peak is still ahead.  Across those 12
   launches log(h1) and log(peak/h1) correlate at -0.94.  mu_A and
   sigma_A are that line's prediction and residual spread.

B. Clean rolling shape.  The per-hour median log-ratio and IQR-based
   spread over the 9 rolling-midnight comparables without a
   PC-simultaneous unlock or an early-access window (ELDEN RING,
   Cyberpunk 2077 and Hogwarts Legacy are left out: their hour-1 prints
   already hold the whole world, so their ratios are not rolling curves
   at all).  This is the raw launch-day ramp: the median ratio rises
   from ~0.9 in the small hours to ~2.5 at hours 19-21 UTC, when
   Europe's evening meets the US afternoon.

Blend: mu = 0.7 * mu_A + 0.3 * mu_B,
       sigma = sqrt((0.7*sigma_A)^2 + (0.3*sigma_B)^2) * 1.15.
The 1.15 widens the bands to the honest 90% coverage the
leave-one-out check shows (0.86 at 1.15 vs 0.82 at 1.0, same loss).

The program fits both components from the comparables inside the case
file it is given, so it adapts to whatever case it runs on.  If a case
carries too few usable comparables (fewer than 5 rolling launches with
complete 24-hour curves), it falls back to the coefficients below,
fitted offline from the same 38 comparables the kit ships
(FALLBACK_OLS / FALLBACK_SHAPE).  The fallback is only a safety net;
on the practice cases the runtime fit is used.

Validation (leave-one-out over the 37 launch comparables, scored with
the contest's pinball loss on the log scale; see tools/validate.py)
-----------------------------------------------------------------
- house baseline (persistence, sigma 0.15):  0.930 on all 37,
  0.357 on the 12 rolling-midnight launches (house's own practice
  figure on its 4 practice cases is 0.387)
- this model:  0.246 on the 12 rolling launches, 0.672 on all 37
  (the blend trades accuracy on simultaneous-global-unlock launches,
  where the shape component's ramp is wrong, for accuracy on
  rolling-midnight launches, which is what GTA VI and all four
  practice cases are; the pure regression scores 0.439 on all 37
  and 0.248 on the rolling 12)
- implied launch-day peaks for the practice cases: God of War
  Ragnarok ~491k (published peak ~478k), Spider-Man 2 ~280k
  (published peak ~257k) -- both within ~10%.

Why the first hour is small and the peak is late: at 00:00 UTC on
release day only New Zealand, Australia, Japan and Korea have
unlocked, plus the first hour of continental Europe; the US East
unlocks at 05:00 UTC, the US West at 08:00 UTC, and the peak lands
in the European-evening / US-afternoon window, hours 19-22 UTC.

Sources for the magnitudes above: the kit's comparables (SullyGnome
hourly Twitch category averages, quoted with attribution in the case
files); Rockstar's 19 Nov 2026 date and midnight-local unlock
(PlayStation support page); trailer viewing figures (Guinness /
Rockstar via trade press); launch-day peaks for Elden Ring (~910k),
God of War Ragnarok (~478k) and Spider-Man 2 (~257k) from
Streams Charts / Final Weapon / TwitchStats.  No figure the model
uses was invented: every number it forecasts from is either in the
case file or fitted on the comparables above.

Runtime: Python 3.13 standard library only, no network, no randomness,
no wall-clock reads; two runs on the same case write the same bytes.
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a module shipped beside this file (none is needed: this file is whole).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
HOURS = list(range(2, 25))

# Blend weights and spread calibration (see the module docstring).
W_OLS = 0.7
W_SHAPE = 1.0 - W_OLS
SIG_MULT = 1.15

# Comparables whose hour-1 print already holds a simultaneous global
# unlock or an early-access window: not rolling curves, left out of the
# clean shape component (but kept in the regression, where ln(p1) absorbs
# them -- see the validation notes).
CONTAMINATED = ("ELDEN RING", "Cyberpunk 2077", "Hogwarts Legacy")

# Fallback coefficients, fitted offline from the 38 comparables the kit
# ships (input/p00.in).  Per hour: OLS (a, b, sigma) on the 12
# rolling-midnight launches, and SHAPE (median log-ratio, sigma) on the
# 9 clean rolling launches.  Used only when a case carries too few
# usable comparables to fit at runtime.
FALLBACK_OLS = {
    2: (-0.6829, 0.0479, 0.1397), 3: (-1.694, 0.1142, 0.3005),
    4: (-0.5448, 0.0133, 0.2748), 5: (1.0715, -0.1076, 0.2173),
    6: (3.3032, -0.2755, 0.406), 7: (4.2558, -0.3536, 0.48),
    8: (4.625, -0.3865, 0.4717), 9: (4.9666, -0.419, 0.4638),
    10: (4.9058, -0.4143, 0.4703), 11: (4.9055, -0.4121, 0.4672),
    12: (5.006, -0.4153, 0.4571), 13: (5.3288, -0.4307, 0.4307),
    14: (5.3807, -0.4231, 0.4198), 15: (5.5376, -0.4264, 0.4121),
    16: (5.5088, -0.4177, 0.4424), 17: (5.1555, -0.3816, 0.4484),
    18: (5.6656, -0.4193, 0.4487), 19: (6.0368, -0.4463, 0.46),
    20: (6.5262, -0.4864, 0.4339), 21: (6.5813, -0.4904, 0.3942),
    22: (6.6116, -0.4953, 0.3668), 23: (6.4968, -0.4909, 0.3692),
    24: (6.6971, -0.516, 0.4017),
}
FALLBACK_SHAPE = {
    2: (-0.1425, 0.15), 3: (-0.3369, 0.1749),
    4: (-0.2876, 0.3711), 5: (-0.1789, 0.2711),
    6: (0.1085, 0.2232), 7: (0.1112, 0.15),
    8: (0.1455, 0.1834), 9: (0.0833, 0.2099),
    10: (0.123, 0.2454), 11: (0.1132, 0.2697),
    12: (0.0944, 0.3099), 13: (0.2093, 0.4588),
    14: (0.3982, 0.5028), 15: (0.5087, 0.5079),
    16: (0.6227, 0.473), 17: (0.7173, 0.4843),
    18: (0.8264, 0.4708), 19: (0.9343, 0.4268),
    20: (0.8799, 0.2065), 21: (0.8726, 0.15),
    22: (0.847, 0.1781), 23: (0.8025, 0.2753),
    24: (0.7348, 0.3578),
}


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def _is_rolling(unlock: str) -> bool:
    text = (unlock or "").lower()
    return "rolling" in text or "midnight local" in text


def _release_day_curve(comparable: dict) -> dict | None:
    """A comparable's release-day hourly viewers, hours 1..24, or None.

    The case files carry each comparable's hourly series from 14 days
    before its release; only the 24 hours of the release day itself are
    a launch curve, and only a complete one with a positive first hour
    is usable.
    """
    from datetime import datetime, timedelta

    try:
        day_start = datetime.fromisoformat(
            comparable["day_start"].replace("Z", "+00:00")
        )
    except (KeyError, ValueError):
        return None
    day: dict[int, float] = {}
    for point in comparable.get("hourly") or []:
        try:
            at = datetime.fromisoformat(point["from"].replace("Z", "+00:00"))
            viewers = float(point["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        if day_start <= at < day_start + timedelta(days=1):
            day[at.hour + 1] = viewers
    if len(day) != 24 or day.get(1, 0) <= 0:
        return None
    return day


def _ols_fit(curves: list[dict]) -> dict:
    """Per-hour OLS of ln(viewers_h / viewers_1) on ln(viewers_1).

    Returns {hour: (a, b, sigma)} with sigma the residual spread.
    """
    fit: dict[int, tuple[float, float, float]] = {}
    for h in HOURS:
        xs = [math.log(max(c[1], 1.0)) for c in curves]
        ys = [math.log(max(c[h], 1.0) / max(c[1], 1.0)) for c in curves]
        mx = statistics.mean(xs)
        my = statistics.mean(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        b = sxy / sxx if sxx > 0 else 0.0
        a = my - b * mx
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sigma = max(0.10, statistics.pstdev(resid)) if len(resid) > 1 else 0.5
        fit[h] = (a, b, sigma)
    return fit


def _shape_fit(curves: list[dict]) -> dict:
    """Per-hour median log-ratio and IQR-based spread.

    Returns {hour: (median, sigma)}.
    """
    fit: dict[int, tuple[float, float]] = {}
    for h in HOURS:
        ratios = sorted(
            math.log(max(c[h], 1.0) / max(c[1], 1.0)) for c in curves
        )
        n = len(ratios)
        median = ratios[n // 2] if n % 2 else (ratios[n // 2 - 1] + ratios[n // 2]) / 2
        q25 = ratios[int(0.25 * (n - 1))]
        q75 = ratios[int(0.75 * (n - 1))]
        fit[h] = (median, max(0.15, (q75 - q25) / 1.349))
    return fit


def _fit_from_case(case: dict) -> tuple[dict, dict] | None:
    """Fit both components from the case's comparables, or None.

    Needs at least 5 rolling-midnight launches with complete curves;
    the clean shape needs at least 3 of those uncontaminated.
    """
    curves: list[tuple[str, dict]] = []
    for comparable in sorted(
        case.get("comparables") or [], key=lambda c: str(c.get("name", ""))
    ):
        if not _is_rolling(str(comparable.get("unlock", ""))):
            continue
        day = _release_day_curve(comparable)
        if day is not None:
            curves.append((str(comparable.get("name", "")), day))
    if len(curves) < 5:
        return None
    rolling = [day for _, day in curves]
    clean = [day for name, day in curves if name not in CONTAMINATED]
    if len(clean) < 3:
        clean = rolling
    return _ols_fit(rolling), _shape_fit(clean)


def _hour_quantiles(p1: float, h: int, ols: dict, shape: dict) -> list:
    """The five quantiles for hour h around hour 1's print p1."""
    a, b, sigma_ols = ols[h]
    mu_ols = a + b * math.log(max(p1, 1.0))
    mu_shape, sigma_shape = shape[h]
    mu = W_OLS * mu_ols + W_SHAPE * mu_shape
    sigma = math.sqrt((W_OLS * sigma_ols) ** 2 + (W_SHAPE * sigma_shape) ** 2)
    sigma *= SIG_MULT
    quantiles: list[float | int] = []
    floor = 0.0
    for z in Z:
        q = p1 * math.exp(mu + z * sigma)
        q = min(max(q, 0.01), 1e9)
        q = round2(q)
        # Quantiles must be non-decreasing; rounding can only break that
        # by a cent, so carry the previous value forward when it does.
        if isinstance(q, (int, float)) and q < floor:
            q = floor
        quantiles.append(q)
        floor = q  # type: ignore[assignment]
    return quantiles


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    fitted = _fit_from_case(case)
    if fitted is None:
        ols, shape = FALLBACK_OLS, FALLBACK_SHAPE
    else:
        ols, shape = fitted
    return {
        "hours": [
            {"hour": row["hour"], "q": _hour_quantiles(p1, row["hour"], ols, shape)}
            for row in case["forecast"]
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
