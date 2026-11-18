"""
Generic two-stage least-squares calibration of affine option pricing models.

Usage (from the code/ directory):
    python -m Calibration.calibration --model svjj --objective vega
    python -m Calibration.calibration --model heston --objective bidask
    python -m Calibration.calibration --model heston --objective sse

Objectives (residual_i = w_i * (model_i - market_i), minimise sum residual_i^2):
    vega   : w = 1 / max(vega, floor)   -> residuals ~ implied-vol errors
    bidask : w = 1 / max(spread, floor) -> residuals in units of bid-ask spread
    sse    : w = 1                      -> plain price SSE
where floor = min_weight_frac * median(vega or spread). Options below the floor
("at_floor") are down-weighted relative to the pure weighting.

Stage 1 screens n_starts random starting points with cheap fits; stage 2
tightly refines up to n_refine *distinct* basins and keeps the best.
"""

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import ndtr

from Calibration.specs import SPECS
from data_handler import load_data
from Fourier import PRICERS

# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

MODEL = "svjj"  # key in SPECS
OBJECTIVE = "vega"  # "vega", "bidask" or "sse"
METHOD = "lewis"

# Minimum weight fraction for vega or bidask objectives. Options with weight below this fraction of the median are down-weighted to the floor.
MIN_WEIGHT_FRAC = 0.05
RUN_FLOOR_SENSITIVITY = True
FLOOR_GRID = (0.10, 0.03, 0.01)

# Two refined candidates are "the same basin" if their x_scale-normalised
# distance is below this.
BASIN_MIN_DIST = 0.05

# Residual for failed pricing, in implied-vol units. It is converted to the
# objective's units via the median of vega * weight (= 1 for the vega objective).
PENALTY = 1.0

# (label, multiplier) used to report residual RMS in readable units
OBJECTIVES = {
    "vega": ("vol pts", 100.0),
    "bidask": ("spreads", 1.0),
    "sse": ("price units", 1.0),
}


# ---------------------------------------------------------
# Black-Scholes helpers (vectorised)
# ---------------------------------------------------------


def bs_price(S, K, tau, r, q, vol, is_call):
    sq = vol * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * tau) / sq
    d2 = d1 - sq
    disc_q = S * np.exp(-q * tau)
    disc_r = K * np.exp(-r * tau)
    call = disc_q * ndtr(d1) - disc_r * ndtr(d2)
    put = disc_r * ndtr(-d2) - disc_q * ndtr(-d1)
    return np.where(is_call, call, put)


def bs_vega(S, K, tau, r, q, vol):
    sq = vol * np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * tau) / sq
    return (
        S * np.exp(-q * tau) * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * np.sqrt(tau)
    )


def implied_vol(price, S, K, tau, r, q, is_call, lo=1e-4, hi=5.0, n_iter=80):
    """Vectorised bisection. Returns (iv, valid_mask)."""
    price = np.asarray(price, dtype=float)
    a = np.full_like(price, lo)
    b = np.full_like(price, hi)

    p_lo = bs_price(S, K, tau, r, q, a, is_call)
    p_hi = bs_price(S, K, tau, r, q, b, is_call)
    valid = np.isfinite(price) & (price > p_lo) & (price < p_hi)

    for _ in range(n_iter):
        m = 0.5 * (a + b)
        too_high = bs_price(S, K, tau, r, q, m, is_call) > price
        b = np.where(too_high, m, b)
        a = np.where(too_high, a, m)

    return 0.5 * (a + b), valid


# ---------------------------------------------------------
# Problem set-up
# ---------------------------------------------------------


def build_groups(market_data, r_by_maturity):
    """Group options by maturity (one characteristic-function pass each)."""
    exdates = pd.to_datetime(market_data["exdate"]).to_numpy()
    strikes = market_data["Strike"].to_numpy(dtype=float)
    taus = market_data["time_to_maturity"].to_numpy(dtype=float)
    is_call = (market_data["cp_flag"] == "C").to_numpy()

    groups = []
    for exdate in np.unique(exdates):
        idx = np.flatnonzero(exdates == exdate)
        groups.append(
            {
                "idx": idx,
                "K": strikes[idx],
                "tau": float(taus[idx[0]]),
                "is_call": is_call[idx],
                "r": r_by_maturity[pd.Timestamp(exdate)],
            }
        )
    return groups


def compute_weights(objective, vega, spread, min_weight_frac):
    """Residual weights and the mask of options sitting on the weight floor."""
    if objective == "sse":
        return np.ones_like(vega), np.zeros(len(vega), dtype=bool)
    if objective == "vega":
        scale = vega
    elif objective == "bidask":
        scale = spread
    else:
        raise ValueError(f"Unknown objective {objective!r}; use {list(OBJECTIVES)}")

    floor = min_weight_frac * np.nanmedian(scale)
    at_floor = ~(scale >= floor)  # NaN spreads count as on the floor
    return 1.0 / np.fmax(scale, floor), at_floor


def prepare_problem(
    spec,
    market_data,
    S0,
    q,
    r_by_maturity,
    objective=OBJECTIVE,
    min_weight_frac=MIN_WEIGHT_FRAC,
):
    """Clean the data and compute market implied vols and weights once."""
    r_by_maturity = {pd.Timestamp(k): v for k, v in r_by_maturity.items()}

    df = market_data.dropna(subset=["Mid"])
    df = df[df["Mid"] > 0].reset_index(drop=True)

    r_opt = pd.to_datetime(df["exdate"]).map(r_by_maturity).to_numpy(dtype=float)
    K = df["Strike"].to_numpy(dtype=float)
    tau = df["time_to_maturity"].to_numpy(dtype=float)
    is_call = (df["cp_flag"] == "C").to_numpy()
    mid = df["Mid"].to_numpy(dtype=float)

    if objective == "bidask":
        if not {"best_bid", "best_offer"}.issubset(df.columns):
            raise ValueError("bidask objective needs best_bid / best_offer columns")
        spread = (df["best_offer"] - df["best_bid"]).to_numpy(dtype=float)
    else:
        spread = np.full(len(df), np.nan)

    iv, valid = implied_vol(mid, S0, K, tau, r_opt, q, is_call)
    n_drop = int((~valid).sum())
    if n_drop:
        print(f"Dropping {n_drop} options whose mid has no Black-Scholes implied vol")

    df = df[valid].reset_index(drop=True)
    r_opt, K, tau, is_call, mid, iv, spread = (
        a[valid] for a in (r_opt, K, tau, is_call, mid, iv, spread)
    )

    vega = bs_vega(S0, K, tau, r_opt, q, iv)
    weights, at_floor = compute_weights(objective, vega, spread, min_weight_frac)

    return {
        "spec": spec,
        "objective": objective,
        "df": df,
        "groups": build_groups(df, r_by_maturity),
        "S0": S0,
        "q": q,
        "mid": mid,
        "mkt_iv": iv,
        "vega": vega,
        "spread": spread,
        "weights": weights,
        "at_floor": at_floor,
        "min_weight_frac": min_weight_frac,
        "penalty": PENALTY * np.median(vega * weights),
        "r_opt": r_opt,
        "K": K,
        "tau": tau,
        "is_call": is_call,
    }


def describe_problem(problem):
    """Options per expiry and weight-floor usage."""
    p = problem
    tab = pd.DataFrame(
        {
            "tau_days": np.round(p["tau"] * 365).astype(int),
            "mny": moneyness(p),
            "at_floor": p["at_floor"],
        }
    )
    g = tab.groupby("tau_days")
    out = pd.DataFrame(
        {
            "n": g.size(),
            "min_K/S0": g["mny"].min(),
            "max_K/S0": g["mny"].max(),
            "at_floor": g["at_floor"].sum(),
        }
    )
    print("\nOptions per expiry")
    print(out.round(3).to_string())
    if p["objective"] != "sse":
        n_fl = int(p["at_floor"].sum())
        print(
            f"\nOn the {p['objective']} weight floor "
            f"(floor = {p['min_weight_frac']:.2f} x median): "
            f"{n_fl}/{len(p['mid'])} options ({n_fl / len(p['mid']):.1%})"
        )


# ---------------------------------------------------------
# Pricing and residuals
# ---------------------------------------------------------


def price_options(params, problem):
    """Model prices in the original option order."""
    spec, S0, q = problem["spec"], problem["S0"], problem["q"]
    prices = np.empty(len(problem["mid"]))

    # State vector (log S_t, V_t); V0 is the last parameter
    Xt = np.array([np.log(S0), params[-1]])

    for g in problem["groups"]:
        r, tau, K = g["r"], g["tau"], g["K"]
        model = spec.make_model(params, r, q)
        calls = PRICERS[METHOD](model, S0, K, tau, Xt)
        puts = PRICERS["put_call_parity"](calls, S0, K, tau, r, q)
        prices[g["idx"]] = np.where(g["is_call"], calls, puts)

    return prices


def residuals(params, problem):
    spec, pen = problem["spec"], problem["penalty"]
    try:
        model_prices = price_options(params, problem)
    except Exception:
        return np.full(len(problem["mid"]) + len(spec.priors), pen)

    res = (model_prices - problem["mid"]) * problem["weights"]
    res = np.where(np.isfinite(res), res, pen)
    res = np.clip(res, -5 * pen, 5 * pen)

    if spec.priors:
        prior = [s * (params[spec.idx[n]] - c) for n, c, s in spec.priors]
        res = np.append(res, prior)

    return res


# ---------------------------------------------------------
# Optimisation
# ---------------------------------------------------------


def sample_starts(spec, initial_guess, n_starts, seed):
    rng = np.random.default_rng(seed)
    starts = [np.asarray(initial_guess, dtype=float)]

    for _ in range(n_starts - 1):
        x0 = np.empty(len(spec.param_names))
        for name, (lo, hi) in spec.log_sample_ranges.items():
            x0[spec.idx[name]] = np.exp(rng.uniform(np.log(lo), np.log(hi)))
        for name, (lo, hi) in spec.uniform_sample_ranges.items():
            x0[spec.idx[name]] = rng.uniform(lo, hi)
        starts.append(x0)

    margin = 1e-6 * (spec.upper - spec.lower)
    return [np.clip(x, spec.lower + margin, spec.upper - margin) for x in starts]


def select_distinct(spec, results, n_select, min_dist=BASIN_MIN_DIST):
    """Indices of the best results that are mutually distinct in x_scale units."""
    costs = np.array([r.cost if np.isfinite(r.cost) else np.inf for r in results])
    chosen = []
    for j in np.argsort(costs):
        if not np.isfinite(costs[j]):
            break
        x = results[j].x
        if all(
            np.linalg.norm((x - results[k].x) / spec.x_scale) > min_dist for k in chosen
        ):
            chosen.append(int(j))
        if len(chosen) == n_select:
            break
    return chosen


def run_optimisation(x0, problem, max_nfev, tight=False):
    spec = problem["spec"]
    return least_squares(
        residuals,
        x0,
        args=(problem,),
        bounds=(spec.lower, spec.upper),
        method="trf",
        x_scale=spec.x_scale,
        diff_step=1e-6,
        loss="linear",
        ftol=1e-12 if tight else 1e-8,
        xtol=1e-12 if tight else 1e-8,
        gtol=1e-10 if tight else 1e-6,
        max_nfev=max_nfev,
    )


def rms(res):
    return np.sqrt(2 * res.cost / len(res.fun))


def calibrate(
    spec,
    market_data,
    price_underlying,
    q,
    r_by_maturity,
    objective=OBJECTIVE,
    initial_guess=None,
    n_starts=20,
    n_refine=3,
    screen_nfev=60,
    refine_nfev=300,
    min_weight_frac=MIN_WEIGHT_FRAC,
    seed=42,
):
    """Returns (parameters dict, best result, all stage-1 results, problem)."""
    if initial_guess is None:
        initial_guess = spec.initial_guess

    problem = prepare_problem(
        spec,
        market_data,
        price_underlying,
        q,
        r_by_maturity,
        objective,
        min_weight_frac,
    )
    unit, mult = OBJECTIVES[objective]

    print(
        f"\nCalibrating {spec.name} to {len(problem['mid'])} options, "
        f"{len(problem['groups'])} maturities"
    )
    print(
        f"Underlying: {price_underlying:.4f} | q: {q:.6f} | method: {METHOD} "
        f"| objective: {objective}"
    )
    print(f"Starts: {n_starts} | refine up to {n_refine} distinct basins")
    describe_problem(problem)
    print()

    # ----- Stage 1: screening ------------------------------------------
    stage1 = []
    for i, x0 in enumerate(sample_starts(spec, initial_guess, n_starts, seed), 1):
        res = run_optimisation(x0, problem, screen_nfev, tight=False)
        stage1.append(res)
        print(
            f"[screen {i:2d}/{n_starts}] cost={res.cost:.6f} "
            f"(rms={rms(res) * mult:.4f} {unit}) nfev={res.nfev} status={res.status}"
        )

    chosen = select_distinct(spec, stage1, n_refine)
    print(f"\n{len(chosen)} distinct basin(s) among the best screening results")

    # ----- Stage 2: refinement -----------------------------------------
    refined = []
    for rank, j in enumerate(chosen, start=1):
        res = run_optimisation(stage1[j].x, problem, refine_nfev, tight=True)
        refined.append(res)
        print(
            f"[refine {rank}/{len(chosen)}] from start {j + 1}: cost={res.cost:.8f} "
            f"nfev={res.nfev} status={res.status} ({res.message})"
        )

    # Parameters that differ across basins at equal cost are weakly identified.
    basin_tab = pd.DataFrame(
        [r.x for r in refined],
        columns=spec.param_names,
        index=[f"basin{i + 1}" for i in range(len(refined))],
    )
    basin_tab["cost"] = [r.cost for r in refined]
    print("\nRefined basins")
    print(basin_tab.to_string(float_format=lambda v: f"{v:.6f}"))

    best = min(refined, key=lambda r: r.cost)
    params = dict(zip(spec.param_names, best.x))
    return params, best, stage1, problem


# ---------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------


def iv_errors(x, problem):
    """Model-minus-market implied vol (NaN where the model price is not invertible)."""
    model_iv, ok = implied_vol(
        price_options(x, problem),
        problem["S0"],
        problem["K"],
        problem["tau"],
        problem["r_opt"],
        problem["q"],
        problem["is_call"],
    )
    return np.where(ok, model_iv - problem["mkt_iv"], np.nan), ok


def moneyness(problem):
    return problem["K"] / problem["S0"]


def _summarise(tab, by):
    g = tab.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "n": g.size(),
            "rmse_pts": np.sqrt(g["sq"].mean()) * 100,
            "mae_pts": g["abs"].mean() * 100,
            "bias_pts": g["err"].mean() * 100,
        }
    )
    return out.round(3).to_string()


def diagnostics(x, problem, best=None, n_worst=10):
    """Implied-vol error overall, by bucket, and the worst-fitting options."""
    err, ok = iv_errors(x, problem)
    e = err[ok]

    print(f"\nIV fit: {ok.sum()}/{len(ok)} model prices invertible")
    print(f"IV RMSE: {np.sqrt(np.mean(e**2)) * 100:.3f} vol pts")
    print(f"IV MAE : {np.mean(np.abs(e)) * 100:.3f} vol pts")
    print(f"IV bias: {np.mean(e) * 100:+.3f} vol pts")
    print(
        "IV |err| percentiles (pts): "
        + ", ".join(
            f"p{p}={np.percentile(np.abs(e), p) * 100:.2f}" for p in (50, 90, 99)
        )
    )
    if best is not None:
        unit, mult = OBJECTIVES[problem["objective"]]
        print(f"Weighted RMS residual: {rms(best) * mult:.3f} {unit}")

    mny = moneyness(problem)
    tab = pd.DataFrame(
        {
            "tau": problem["tau"],
            "mny": mny,
            "err": err,
            "at_floor": problem["at_floor"],
        }
    )[ok]
    tab["sq"] = tab["err"] ** 2
    tab["abs"] = tab["err"].abs()
    tab["maturity"] = pd.cut(
        tab["tau"],
        [0, 30 / 365, 90 / 365, 180 / 365, 1.0, np.inf],
        labels=["<1m", "1-3m", "3-6m", "6-12m", ">1y"],
    )
    tab["moneyness"] = pd.cut(
        tab["mny"],
        [0, 0.8, 0.9, 0.97, 1.03, 1.10, np.inf],
        labels=["<0.80", "0.80-0.90", "0.90-0.97", "0.97-1.03", "1.03-1.10", ">1.10"],
    )

    print("\nIV error by maturity")
    print(_summarise(tab, "maturity"))
    print("\nIV error by moneyness (K/S0)")
    print(_summarise(tab, "moneyness"))
    if problem["objective"] != "sse":
        print("\nIV error: on weight floor vs not")
        print(_summarise(tab, "at_floor"))

    idx_ok = np.flatnonzero(ok)
    worst = idx_ok[np.argsort(-np.abs(err[idx_ok]))[:n_worst]]
    w = pd.DataFrame(
        {
            "K": problem["K"][worst],
            "tau_d": np.round(problem["tau"][worst] * 365).astype(int),
            "K/S0": mny[worst],
            "cp": np.where(problem["is_call"][worst], "C", "P"),
            "mkt_iv": problem["mkt_iv"][worst],
            "err_pts": err[worst] * 100,
            "at_floor": problem["at_floor"][worst],
        }
    )
    print(f"\nWorst {n_worst} options by |IV error|")
    print(w.round(3).to_string(index=False))


def flag_bounds(spec, x, tol=1e-3):
    print("\nParameters at / near bounds:")
    hit = False
    for name, v, lo, hi in zip(spec.param_names, x, spec.lower, spec.upper):
        span = hi - lo
        if v - lo < tol * span or hi - v < tol * span:
            print(f"  {name}: {v:.6f} (bounds {lo}, {hi})")
            hit = True
    if not hit:
        print("  none")


def parameter_uncertainty(res, flat_tol=1e-8, rcond=1e-12):
    """
    Indicative standard errors and correlations from the Jacobian at the
    solution: cov = s^2 (J'J)^+ with s^2 = 2*cost / (m - n_free). Assumes
    roughly iid residuals, so treat as a guide to identifiability.

    Parameters on a bound get NaN; parameters with a (numerically) zero
    Jacobian column are unidentified (std_err = inf). Returns (se, corr, cond).
    """
    J = res.jac
    m, n = J.shape
    free = res.active_mask == 0

    col_norm = np.linalg.norm(J, axis=0)
    flat = col_norm < flat_tol * max(col_norm.max(), 1e-300)
    use = free & ~flat
    nf = int(use.sum())

    se = np.full(n, np.nan)
    se[flat & free] = np.inf
    corr = np.full((n, n), np.nan)
    if nf == 0 or m <= nf:
        return se, corr, np.nan

    # unit-norm columns for numerical stability
    d = col_norm[use]
    Js = J[:, use] / d
    s2 = 2.0 * res.cost / (m - nf)
    cov = s2 * np.linalg.pinv(Js.T @ Js, rcond=rcond) / np.outer(d, d)

    sef = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        cf = np.clip(cov / np.outer(sef, sef), -1.0, 1.0)

    se[use] = sef
    corr[np.ix_(use, use)] = cf

    sv = np.linalg.svd(Js, compute_uv=False)
    return se, corr, sv[0] / max(sv[-1], 1e-300)


def report_uncertainty(spec, res, corr_threshold=0.95):
    names = spec.param_names
    se, corr, cond = parameter_uncertainty(res)
    tab = pd.DataFrame(
        {"value": res.x, "std_err": se, "|se/value|": np.abs(se / res.x)},
        index=names,
    )
    print("\nIndicative parameter uncertainty (from Jacobian)")
    print(tab.to_string(float_format=lambda v: f"{v:.5f}"))
    print(f"Condition number of scaled Jacobian: {cond:.2e}")
    if np.isinf(se).any():
        print(
            "Parameters with no sensitivity in the data (std_err = inf): "
            + ", ".join(np.array(names)[np.isinf(se)])
        )
    if cond > 1e6:
        print(
            "Warning: Jacobian is ill-conditioned; some standard errors above "
            "are likely too small (near-flat directions are truncated)."
        )

    pairs = [
        (names[i], names[j], corr[i, j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
        if np.isfinite(corr[i, j]) and abs(corr[i, j]) > corr_threshold
    ]
    if pairs:
        print(f"Parameter pairs with |corr| > {corr_threshold}:")
        for a, b, c in pairs:
            print(f"  {a} / {b}: {c:+.3f}")
    else:
        print(f"No parameter pairs with |corr| > {corr_threshold}")
    return se


def floor_sensitivity(
    spec,
    x_best,
    market_data,
    S0,
    q,
    r_by_maturity,
    objective=OBJECTIVE,
    floors=FLOOR_GRID,
    max_nfev=200,
):
    """
    Refit from the best solution under different weight floors and compare IV
    errors on a common option set (the set does not depend on the floor).
    """
    rows = []
    for f in floors:
        prob = prepare_problem(spec, market_data, S0, q, r_by_maturity, objective, f)
        res = run_optimisation(x_best, prob, max_nfev, tight=True)
        err, ok = iv_errors(res.x, prob)
        mny = moneyness(prob)

        def rmse(mask):
            e = err[mask & ok]
            return np.sqrt(np.mean(e**2)) * 100 if len(e) else np.nan

        rows.append(
            {
                "floor": f,
                "n_floor": int(prob["at_floor"].sum()),
                "IV_RMSE_all": rmse(np.ones(len(err), bool)),
                "IV_RMSE_core(0.9-1.1)": rmse((mny >= 0.9) & (mny <= 1.1)),
                "IV_RMSE_wing(<0.9)": rmse(mny < 0.9),
                "IV_RMSE_wing(>1.1)": rmse(mny > 1.1),
                **{n: res.x[spec.idx[n]] for n in spec.sensitivity_params},
            }
        )
    out = pd.DataFrame(rows).set_index("floor")
    print(
        f"\n{objective} weight-floor sensitivity "
        "(local refit from best solution; RMSE in vol pts)"
    )
    print(out.round(4).to_string())
    return out


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------


def main(
    model=MODEL,
    objective=OBJECTIVE,
    n_starts=20,
    n_refine=3,
    seed=42,
    floor_sens=RUN_FLOOR_SENSITIVITY,
):
    spec = SPECS[model]
    data = load_data(NoArb=True)
    df = data["cleaned_dataframe"]
    q = data["q"]
    r_by_maturity = data["r_by_maturity"]
    S0 = data["underlying_close"]

    parameters, best, _, problem = calibrate(
        spec,
        df,
        S0,
        q,
        r_by_maturity,
        objective=objective,
        n_starts=n_starts,
        n_refine=n_refine,
        seed=seed,
    )

    print("\n" + "=" * 55)
    print(f"Calibrated {spec.name} parameters ({objective} objective)")
    print("=" * 55)
    for name, value in parameters.items():
        print(f"{name:10s}: {value:.8f}")

    print("\nOptimizer")
    print(f"status={best.status} | {best.message}")
    print(f"nfev={best.nfev} | cost={best.cost:.8f} | optimality={best.optimality:.3e}")

    flag_bounds(spec, best.x)
    se = report_uncertainty(spec, best)
    if spec.structural_checks is not None:
        spec.structural_checks(parameters)
    diagnostics(best.x, problem, best)

    if floor_sens and objective != "sse":
        floor_sensitivity(spec, best.x, df, S0, q, r_by_maturity, objective)

    # The vega fit keeps the file name used by the plotting scripts.
    out_file = spec.output_file
    if objective != "vega":
        out_file = out_file.replace(".csv", f"_{objective}.csv")
    pd.DataFrame({"value": best.x, "std_err": se}, index=spec.param_names).to_csv(
        out_file
    )
    print(f"\nSaved parameters to {out_file}")
    return parameters, best, problem


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(SPECS), default=MODEL)
    ap.add_argument("--objective", choices=list(OBJECTIVES), default=OBJECTIVE)
    ap.add_argument("--n-starts", type=int, default=20)
    ap.add_argument("--n-refine", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-floor-sensitivity", action="store_true", help="skip floor sensitivity"
    )
    a = ap.parse_args()
    main(
        a.model,
        a.objective,
        a.n_starts,
        a.n_refine,
        a.seed,
        floor_sens=RUN_FLOOR_SENSITIVITY and not a.no_floor_sensitivity,
    )
