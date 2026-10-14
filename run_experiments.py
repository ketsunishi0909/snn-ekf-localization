"""
Runs the Week 2 (Oct 19-25) experiments and writes figures + tables to results/.

    python run_experiments.py            # full run (~1-2 min)
    python run_experiments.py --quick    # fewer Monte Carlo runs
"""
import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import SimConfig, simulate
from ekf import FilterConfig, dead_reckoning, run_ekf, pos_error, rmse, nees

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


def one_run(scfg, fcfg):
    d = simulate(scfg)
    dr = dead_reckoning(d["x_true"][0], d["v_meas"], d["w_meas"], scfg.dt)
    t0 = time.perf_counter()
    xs, Ps, rej = run_ekf(d, scfg.dt, fcfg, beacons=scfg.beacons)
    ms_per_step = 1e3 * (time.perf_counter() - t0) / len(d["t"])
    return d, dr, xs, Ps, rej, ms_per_step


def monte_carlo(scfg, fcfg, n_runs):
    rows = []
    for s in range(n_runs):
        d, dr, xs, Ps, rej, ms = one_run(replace(scfg, seed=s), fcfg)
        rows.append(dict(
            dr_rmse=rmse(dr, d["x_true"]),
            ekf_rmse=rmse(xs, d["x_true"]),
            ekf_max=float(pos_error(xs, d["x_true"]).max()),
            dr_final=float(pos_error(dr, d["x_true"])[-1]),
            ekf_final=float(pos_error(xs, d["x_true"])[-1]),
            nees=float(nees(xs, Ps, d["x_true"]).mean()),
            rejected=rej, ms=ms))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def plot_single(scfg, fcfg, name, title):
    d, dr, xs, Ps, _, _ = one_run(scfg, fcfg)
    xt, t = d["x_true"], d["t"]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(xt[:, 0], xt[:, 1], "k", lw=1.5, label="truth")
    ax[0].plot(dr[:, 0], dr[:, 1], color="tab:red", lw=1, label="dead reckoning")
    ax[0].plot(xs[:, 0], xs[:, 1], color="tab:blue", lw=1, label="EKF")
    if scfg.meas_mode == "position":
        z = np.array([m[0][1] for m in d["meas"] if m])
        ax[0].scatter(z[:, 0], z[:, 1], s=4, c="gray", alpha=0.5, label="position fixes")
    else:
        ax[0].scatter(*scfg.beacons.T, marker="^", s=60, c="tab:green", label="beacons")
    ax[0].set_aspect("equal"); ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].legend(loc="best", fontsize=7); ax[0].set_title(title)

    e_ekf = pos_error(xs, xt)
    sig3 = 3 * np.sqrt(Ps[:, 0, 0] + Ps[:, 1, 1])
    ax[1].plot(t, pos_error(dr, xt), color="tab:red", lw=1, label="dead reckoning")
    ax[1].plot(t, e_ekf, color="tab:blue", lw=1, label="EKF")
    ax[1].plot(t, sig3, "--", color="tab:blue", lw=0.8, label="EKF 3σ bound")
    if scfg.outage:
        ax[1].axvspan(*scfg.outage, color="orange", alpha=0.15, label="outage")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("position error (m)")
    ax[1].set_title(f"Error  (EKF RMSE {rmse(xs, xt):.2f} m, DR RMSE {rmse(dr, xt):.2f} m)")
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / name); plt.close(fig)


def main(n_runs):
    base = SimConfig()
    fbase = FilterConfig()
    rb = replace(base, meas_mode="range_bearing")

    # ---- 1. single-run figures ----
    plot_single(base, fbase, "fig1_position_fixes.png", "Position fixes every 1 s")
    plot_single(rb, fbase, "fig2_range_bearing.png", "Range-bearing to 4 beacons")

    # ---- 2. dropout sweep ----
    drops = [0.0, 0.25, 0.5, 0.75, 0.9, 0.97]
    sweep = {"pos": [], "rb": []}
    for p in drops:
        sweep["pos"].append(monte_carlo(replace(base, dropout=p), fbase, n_runs))
        sweep["rb"].append(monte_carlo(replace(rb, dropout=p), fbase, n_runs))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(drops, [r["dr_rmse"] for r in sweep["pos"]], "o-", c="tab:red", label="dead reckoning")
    ax.plot(drops, [r["ekf_rmse"] for r in sweep["pos"]], "o-", c="tab:blue", label="EKF, position fixes")
    ax.plot(drops, [r["ekf_rmse"] for r in sweep["rb"]], "s-", c="tab:green", label="EKF, range-bearing")
    ax.set_xlabel("measurement dropout probability"); ax.set_ylabel("position RMSE (m)")
    ax.set_yscale("log"); ax.set_title(f"Dropout sweep ({n_runs} runs per point)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(OUT / "fig3_dropout_sweep.png"); plt.close(fig)

    # ---- 3. failure cases ----
    cases = [
        ("Nominal", base, fbase),
        ("Unmodeled gyro bias 0.01 rad/s", replace(base, gyro_bias=0.01), fbase),
        ("Gyro bias + 75% dropout", replace(base, gyro_bias=0.01, dropout=0.75), fbase),
        ("100 s outage (t=100-200 s)", replace(base, outage=(100, 200)), fbase),
        ("Filter overconfident (assumes σ_pos=0.2 m, true 1.0 m)", base, replace(fbase, sigma_pos=0.2)),
        ("Filter underconfident (assumes σ_pos=5 m)", base, replace(fbase, sigma_pos=5.0)),
        ("5% outliers (10x noise), no gating", replace(base, outlier_prob=0.05), fbase),
        ("5% outliers, chi-square gating", replace(base, outlier_prob=0.05), replace(fbase, gate=True)),
        ("Range-bearing, nominal", rb, fbase),
        ("Range-bearing, 5% outliers, no gating", replace(rb, outlier_prob=0.05), fbase),
        ("Range-bearing, 5% outliers, gating", replace(rb, outlier_prob=0.05), replace(fbase, gate=True)),
    ]
    results = [(name, monte_carlo(s, f, n_runs)) for name, s, f in cases]
    plot_single(replace(base, outage=(100, 200)), fbase, "fig4_outage.png", "100 s measurement outage")

    # ---- write tables ----
    lines = ["# Results", "",
             f"Monte Carlo: {n_runs} seeds per row, 300 s lawnmower survey, dt = 0.1 s, "
             "measurements attempted every 1 s.", "",
             "## Dropout sweep (position RMSE, m)", "",
             "| dropout | dead reckoning | EKF (position fixes) | EKF (range-bearing) |",
             "|---|---|---|---|"]
    for p, a, b in zip(drops, sweep["pos"], sweep["rb"]):
        lines.append(f"| {p:.2f} | {a['dr_rmse']:.2f} | {a['ekf_rmse']:.2f} | {b['ekf_rmse']:.2f} |")
    lines += ["", "## Failure cases", "",
              "NEES: average normalized error squared. ~3 means the filter's covariance "
              "is honest; much higher means it is overconfident, much lower means underconfident.", "",
              "| case | EKF RMSE (m) | EKF max err (m) | DR RMSE (m) | mean NEES | rejected meas. | ms/step |",
              "|---|---|---|---|---|---|---|"]
    for name, r in results:
        lines.append(f"| {name} | {r['ekf_rmse']:.2f} | {r['ekf_max']:.2f} | {r['dr_rmse']:.2f} "
                     f"| {r['nees']:.1f} | {r['rejected']:.0f} | {r['ms']:.3f} |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(n_runs=5 if a.quick else 30)
