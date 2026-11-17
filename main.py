"""Forecast GTA VI launch-day Twitch viewers, hour by hour, from hour 1.

Method (comparables nearest-neighbour blended with a size regression):
for the case at hand, take hour 1's print p1.  Every comparable launch
contributes its release-day curve as log-ratios log(viewers(h) /
viewers(1)).  The shape of a launch day depends strongly on how big
hour 1 already is -- a launch whose first UTC hour is still pre-unlock
(tiny p1) explodes later, while a rolling-midnight console launch with
a large p1 stays near it and then rides the US prime-time wave -- so
the centre of the forecast blends two views of that size effect:

  * the median log-ratio among the 12 comparables whose hour-1 print
    is nearest to this case's p1 in log space (kNN-12), and
  * a per-hour least-squares regression of log-ratio on log(p1) over
    all comparables (the size effect is near-linear: correlation
    about -0.93 at hour 19).

The regression weight adapts to p1: 0.7 below 50,000 viewers (where
the kNN median badly under-forecasts the surge -- Wolverine, p1
23,357, rose to 9x its hour-1 print), 0.2 from 50,000 to 150,000,
and 0 above 150,000 (where the neighbours alone are best).  The
spread around the blended centre is the kNN-12 neighbours' actual
empirical quantiles at that hour, so the interval is wide late in
the day and narrow early.

Evidence, pinball loss on the log scale, scored with the official
formula against the practice answers (SullyGnome hourly averages,
the kit's own source): starter baseline 0.387; kNN-12 alone 0.242;
regression alone 0.237; this adaptive blend 0.188 (per case 0.189 /
0.218 / 0.146 / 0.200).  Leave-one-out over the 37 comparable days
agrees the blend costs almost nothing when p1 is large and insures
heavily when it is small.  Quantiles are interpolated linearly; if
too few neighbours carry an hour, the pool widens to all
comparables, and if none do, that hour falls back to persistence
around p1 with a 0.15 * sqrt(h - 1) log spread.

Pure standard library, deterministic, no network.  Reads every
<input>/<id>.in and writes <output>/<id>.out with 23 rows, hours 2..24.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
NEIGHBOURS = 12
FALLBACK_SIGMA = 0.15


def regression_weight(p1):
    """Weight on the size regression's median: high when hour 1 is
    small (the surge is then systematically under-forecast by the
    neighbours alone), fading to zero for very large hour-1 prints."""
    if p1 < 50000:
        return 0.7
    if p1 < 150000:
        return 0.2
    return 0.0


def regression_prediction(log_p1, points):
    """Least-squares prediction of the log-ratio at log_p1 from
    (log hour-1, log-ratio) points; None when the fit is degenerate."""
    n = len(points)
    if n < 5:
        return None
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    var_x = sum((p[0] - mean_x) ** 2 for p in points)
    if var_x == 0:
        return None
    slope = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / var_x
    return mean_y + slope * (log_p1 - mean_x)


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as ints."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value.is_integer() else value


def interp_quantile(sorted_values, tau):
    """Linear-interpolation quantile of a sorted list (type 7)."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = lo + 1
    if hi >= n:
        return sorted_values[-1]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def day_curve(comparable):
    """Hour -> viewers on the comparable's release day (day_start date)."""
    day = str(comparable.get("day_start", ""))[:10]
    curve = {}
    for row in comparable.get("hourly") or []:
        stamp = str(row.get("from", ""))
        if stamp[:10] != day:
            continue
        try:
            hour = int(stamp[11:13]) + 1
            viewers = row.get("viewers")
        except (ValueError, TypeError):
            continue
        if 1 <= hour <= 24 and isinstance(viewers, (int, float)) and viewers > 0:
            curve[hour] = float(viewers)
    return curve


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(name + " asks for levels " + repr(case.get("levels")))
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(name + " has no hour-1 print")
    p1 = max(1.0, float(first["viewers"]))
    log_p1 = math.log(p1)

    # Rank comparables by distance in log hour-1 size; ties broken by
    # name so the order is deterministic.
    ranked = []
    for comp in case.get("comparables") or []:
        curve = day_curve(comp)
        if 1 not in curve:
            continue
        ranked.append((abs(math.log(curve[1]) - log_p1), str(comp.get("name", "")), curve))
    ranked.sort(key=lambda item: (item[0], item[1]))

    hours_out = []
    weight = regression_weight(p1)
    for row in case.get("forecast") or []:
        h = int(row["hour"])
        log_ratios = []
        for _dist, _name, curve in ranked:
            if h in curve:
                log_ratios.append(math.log(curve[h] / curve[1]))
            if len(log_ratios) >= NEIGHBOURS:
                break
        if len(log_ratios) < 3:
            # Too few neighbours carry this hour: widen to every
            # comparable that has it.
            log_ratios = []
            for _dist, _name, curve in ranked:
                if h in curve:
                    log_ratios.append(math.log(curve[h] / curve[1]))
        if log_ratios:
            log_ratios.sort()
            median = interp_quantile(log_ratios, 0.5)
            centre = median
            if weight > 0:
                points = [
                    (math.log(curve[1]), math.log(curve[h] / curve[1]))
                    for _dist, _name, curve in ranked
                    if h in curve
                ]
                prediction = regression_prediction(log_p1, points)
                if prediction is not None:
                    centre = (1.0 - weight) * centre + weight * prediction
            quantiles = [
                p1 * math.exp(centre + (interp_quantile(log_ratios, tau) - median))
                for tau in LEVELS
            ]
        else:
            # No comparable covers this hour: persistence fallback.
            spread = FALLBACK_SIGMA * math.sqrt(max(0, h - 1))
            quantiles = [p1 * math.exp(z * spread) for z in Z]
        cleaned = []
        prev = 0.0
        for value in quantiles:
            value = min(1e9, max(1.0, value))
            if value < prev:
                value = prev
            cleaned.append(round2(value))
            prev = float(cleaned[-1])
        hours_out.append({"hour": h, "q": cleaned})
    return {"hours": hours_out}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
