"""Roblox Up-and-Coming top-10 forecaster (make.fun roblox-up-and-coming-forecast-01).

Reads /input/history.ndjson + /input/task.json, writes /output/forecast.json.
Python 3.13, standard library only, no network, deterministic.

Model: horizon-averaged rank occupancy.
- For every game seen in the history we estimate its rank trajectory:
  last rank, rank velocity (recency-weighted regression over ~48h) and
  log-playing growth (recency-weighted regression over ~36h, standardised
  across games). Playing growth is a leading indicator: the chart sorts by
  user growth, so a game whose playing is compounding climbs before its
  rank shows it (e.g. Sweep The Moon +599%/24h while ranked #29).
- At each horizon h in 0..72h the game's rank is modelled as Normal with
  centre = last rank + velocity saturated over ~24h - growth shift, and
  spread growing like sqrt(h) (diffusion). P(game at place k) is the
  Normal mass on [k-0.5, k+0.5], averaged over horizons (each scored hour
  counts equally) and multiplied by an age-out survival factor when the
  game's creation date is known (only games published in the last 28 days
  are on the chart).
- Per place the named probabilities are capped so a share stays unnamed:
  brand-new games that never appeared in the history still take places
  (3 of the 17 games in the practice window's top 10s were never in the
  practice history), and the unnamed share is their only cover.
- top10 orders games by probability-weighted place score.
"""

import json
import math
import os

# --- tuned parameters (see dev/tune.py for the grid search) ---
PARAMS = {
    "tau_h": 12.0,        # velocity saturation timescale (hours)
    "vel_clip": 1.0,      # max |rank velocity| in places/hour
    "growth_shift": 9.0,  # places of rank shift per std of playing growth
    "sigma0": 2.25,       # rank spread at horizon 0 (places)
    "sigma1": 0.65,       # spread growth per sqrt(hour)
    "vol_weight": 0.0,    # extra spread per unit rank residual volatility
    "unnamed": 0.10,      # share of each place left unnamed
    "horizons": [0, 6, 12, 24, 36, 48, 60, 72],
    "mix": 0.3,           # weight of the persistence (starter) distribution mixed in
}

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")


def parse_time(s):
    # ISO 8601 UTC -> epoch seconds (no datetime import needed beyond this)
    import datetime
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wlinreg(xs, ys, ws):
    """Weighted least-squares slope/intercept; returns (slope, intercept, resid_std)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-9:
        return 0.0, my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    inter = my - slope * mx
    var = sum(w * (y - (inter + slope * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / sw
    return slope, inter, math.sqrt(max(var, 0.0))


def build_forecast(history, task, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    places = task["places"]
    snaps = []
    for rec in history:
        t = parse_time(rec["at"])
        rows = sorted(rec.get("up_and_coming") or [], key=lambda r: r["rank"])
        if rows:
            snaps.append((t, rows))
    snaps.sort(key=lambda x: x[0])
    t0 = snaps[-1][0]
    chart_now = {r["universe_id"]: r for _, rows in [snaps[-1]] for r in rows}

    # per-game series: (t_hours_before_t0, rank, playing); created date if given
    series = {}
    created = {}
    for t, rows in snaps:
        for r in rows:
            uid = r["universe_id"]
            series.setdefault(uid, []).append(((t - t0) / 3600.0, r["rank"],
                                               r.get("playing")))
    # details live on the snapshot records, keyed by universe id (dict) or list
    for rec in history:
        det = rec.get("details")
        if isinstance(det, dict):
            for k, v in det.items():
                if isinstance(v, dict) and v.get("created"):
                    try:
                        created[int(k)] = parse_time(v["created"])
                    except Exception:
                        pass
        elif isinstance(det, list):
            for v in det:
                if isinstance(v, dict) and v.get("created") and v.get("universe_id"):
                    try:
                        created[int(v["universe_id"])] = parse_time(v["created"])
                    except Exception:
                        pass

    # --- features per game ---
    feats = {}
    growths = []
    for uid in sorted(series):
        pts = series[uid]
        last_dt, last_rank, last_play = pts[-1]
        # rank velocity: regression over last 48h, exp weights (24h scale)
        pairs = [(x, y) for x, y, _ in pts if x >= -48.0]
        w = [math.exp(x / 24.0) for x, _ in pairs]
        vel, _, vol = wlinreg([x for x, _ in pairs], [y for _, y in pairs], w)
        vel = max(-p["vel_clip"], min(p["vel_clip"], vel))
        # playing growth: log-playing regression over last 36h
        pp = [(x, math.log(play)) for x, _, play in pts
              if x >= -36.0 and play and play > 0]
        if len(pp) >= 2:
            w2 = [math.exp(x / 18.0) for x, _ in pp]
            g, _, _ = wlinreg([x for x, _ in pp], [y for _, y in pp], w2)
        else:
            g = 0.0
        g = max(-0.10, min(0.20, g))
        growths.append(g)
        # absent from the last snapshot: rank drifts out (chart size ~100)
        if uid not in chart_now:
            hours_absent = -last_dt
            r0 = last_rank + 0.5 * hours_absent + 3.0
        else:
            r0 = float(last_rank)
        feats[uid] = {"r0": r0, "vel": vel, "growth": g, "vol": vol,
                      "last_seen_h": last_dt}
    if growths:
        mg = sum(growths) / len(growths)
        sg = math.sqrt(sum((g - mg) ** 2 for g in growths) / len(growths)) or 1e-6
    else:
        mg, sg = 0.0, 1.0
    for uid, f in feats.items():
        f["growth_z"] = (f["growth"] - mg) / sg

    # --- horizon-averaged occupancy ---
    horizons = p["horizons"]
    hw = 1.0 / len(horizons)
    probs = {}  # (uid, place) -> p
    for uid in sorted(feats):
        f = feats[uid]
        shift = p["growth_shift"] * max(f["growth_z"], 0.0)
        # (negative growth_z gives no bonus, but still counts via velocity)
        sd_extra = p["vol_weight"] * f["vol"]
        age0 = None
        if uid in created:
            age0 = (t0 - created[uid]) / 86400.0
        for place in range(1, places + 1):
            tot = 0.0
            for h in horizons:
                sat = 1.0 - math.exp(-h / p["tau_h"]) if h > 0 else 0.0
                centre = f["r0"] + f["vel"] * p["tau_h"] * sat - shift * (1.0 - math.exp(-h / 24.0) if h > 0 else 0.0)
                centre = max(1.0, min(130.0, centre))
                sd = p["sigma0"] + p["sigma1"] * math.sqrt(h) + sd_extra
                lo = norm_cdf((place - 0.5 - centre) / sd)
                hi = norm_cdf((place + 0.5 - centre) / sd)
                mass = max(0.0, hi - lo)
                surv = 1.0
                if age0 is not None:
                    age_h = age0 + h / 24.0
                    if age_h > 28.5:
                        surv = 0.0
                    elif age_h > 26.5:
                        surv = (28.5 - age_h) / 2.0
                tot += hw * mass * surv
            if tot > 1e-6:
                probs[(uid, place)] = tot

    # --- mix in a persistence (starter) distribution: the chart's most
    # stable places reward staying put; the mix recovers those without
    # giving up the momentum gains elsewhere ---
    mix = p.get("mix", 0.0)
    if mix > 0:
        last_chart = [r["universe_id"] for r in
                      sorted(snaps[-1][1], key=lambda r: r["rank"])]
        persist = {}
        for place in range(1, places + 1):
            if place > len(last_chart):
                break
            persist[(last_chart[place - 1], place)] = 0.5
            if place == 1:
                if len(last_chart) > 1:
                    persist[(last_chart[1], place)] = 0.3
                continue
            persist[(last_chart[place - 2], place)] = 0.15
            if place < len(last_chart):
                persist[(last_chart[place], place)] = 0.15
        keys = set(probs) | set(persist)
        probs = {k: (1.0 - mix) * probs.get(k, 0.0) + mix * persist.get(k, 0.0)
                 for k in keys}
        probs = {k: v for k, v in probs.items() if v > 1e-6}

    # --- per-place cap: leave `unnamed` share unnamed ---
    cap = 1.0 - p["unnamed"]
    place_sum = {}
    for (uid, place), v in probs.items():
        place_sum[place] = place_sum.get(place, 0.0) + v
    for place, s in place_sum.items():
        if s > cap:
            fct = cap / s
            for (uid, pl), v in list(probs.items()):
                if pl == place:
                    probs[(uid, place)] = v * fct
    # --- per-game cap: a game holds one place at a time ---
    game_sum = {}
    for (uid, place), v in probs.items():
        game_sum[uid] = game_sum.get(uid, 0.0) + v
    for uid, s in game_sum.items():
        if s > 1.0:
            fct = 1.0 / s
            for (uid2, place), v in list(probs.items()):
                if uid2 == uid:
                    probs[(uid2, place)] = v * fct

    rows = [{"universe_id": uid, "place": place, "p": round(v, 6)}
            for (uid, place), v in sorted(probs.items(), key=lambda kv: (kv[0][1], kv[0][0]))
            if v >= 0.0005]
    # top10 by probability-weighted place score
    score = {}
    for (uid, place), v in probs.items():
        score[uid] = score.get(uid, 0.0) + v * (places + 1 - place)
    top10 = [uid for uid, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:places]]
    return {"probabilities": rows, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json")) as f:
        task = json.load(f)
    history = []
    with open(os.path.join(INPUT_DIR, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    fc = build_forecast(history, task)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task["output"]), "w") as f:
        json.dump(fc, f, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
