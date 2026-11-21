import numpy as np
import pandas as pd
from scipy.optimize import minimize

from option_pricing import (
    calculate_implied_volatility,
    implied_call_price,
    implied_put_price,
)

FILE_PATH = "Data20191113.xlsx"

SHEETS = {
    "SPX_options": {
        "sheet_name": "SPX options",
        "start_row": 6,
        "end_row": 3784,
        "usecols": "A:M",
        "close_cell": "G3",
    },
}


# =====================================================================
# Loading
# =====================================================================
def load_data(
    file_path=FILE_PATH,
    config=SHEETS["SPX_options"],
    clean=True,
    infer_q_r=True,
    NoArb=True,
    verbose=False,
):
    if NoArb and not (clean and infer_q_r):
        raise ValueError("NoArb=True requires clean=True and infer_q_r=True.")

    sheet = config["sheet_name"]

    with pd.ExcelFile(file_path) as workbook:
        df = pd.read_excel(
            workbook,
            sheet_name=sheet,
            usecols=config["usecols"],
            skiprows=config["start_row"] - 1,
            nrows=config["end_row"] - config["start_row"] + 1,
        )

        price_underlying = pd.read_excel(
            workbook,
            sheet_name=sheet,
            usecols=config["close_cell"][0],
            skiprows=int("".join(filter(str.isdigit, config["close_cell"]))) - 1,
            nrows=1,
            header=None,
        ).iat[0, 0]

    df["time_to_maturity"] = (df["exdate"] - df["date"]).dt.days / 365.0

    return_value = {"dataframe": df, "underlying_close": price_underlying}

    c_df = clean_data(df, price_underlying) if clean else df.copy()
    if clean:
        if verbose:
            print(f"Initial number of observations: {len(df)}")
            print(f"Number of observations after cleaning: {len(c_df)}")
        return_value["cleaned_dataframe"] = c_df

    if infer_q_r:
        closest_options = get_closest_options(c_df, price_underlying)
        inferred_r, inferred_q = infer_dividend_yield_and_discount_rate(
            closest_options, price_underlying
        )

        if verbose:
            print("Inferred discount rates by maturity:")
            # sort and print the inferred discount rates by maturity
            for exdate, r in sorted(inferred_r.items()):
                print(f"  {exdate.date()}: {r:.4%}")
            print(f"Inferred dividend yield (single q): {inferred_q:.4%}")
        return_value["r_by_maturity"] = inferred_r
        return_value["q"] = inferred_q

    if NoArb:
        return_value["cleaned_dataframe"] = remove_arbitrage_violations(
            c_df,
            price_underlying,
            return_value["r_by_maturity"],
            return_value["q"],
            verbose=verbose,
        )

    return return_value


# =====================================================================
# No-arbitrage filtering
# =====================================================================
def _clean_maturity_slice(group, S, r, q, T, tol=1e-8):
    """
    Strike-wise conditions for one maturity, with K = 0
    C(0, T) = S exp(-qT) included as a protected node:

        0 <= slope_i <= exp(-rT)            (monotonicity)
        slope_i  - slope_{i+1} >=0          (convexity, incl. protected node)

    where slope_i = (C_i - C_{i+1}) / (K_{i+1} - K_i).
    Removes the observation involved in the most violations until none
    remain. Removing a point from a convex sequence keeps it convex.
    """
    group = group.sort_values("Strike").reset_index(drop=True)
    max_slope = np.exp(-r * T)
    max_mono = S * np.exp(-q * T)
    has_spread = {"best_bid", "best_offer"}.issubset(group.columns)

    n_removed = n_mono = n_conv = 0

    while len(group) >= 1:
        strikes = np.concatenate(([0.0], group["Strike"].to_numpy()))
        prices = np.concatenate(([max_mono], group["Mid"].to_numpy()))

        slopes = (prices[:-1] - prices[1:]) / np.diff(strikes)

        # monotonicity violation: slope < 0 or slope > max_slope
        mono = (slopes < -tol) | (slopes > max_slope + tol)

        # convexity violation: slope_i - slope_{i+1} < 0
        conv = slopes[:-1] - slopes[1:] < -tol

        if not mono.any() and not conv.any():
            break  # no violations remain so break the loop

        # count violations for reporting
        n_mono += int(mono.sum())
        n_conv += int(conv.sum())

        # score each observation by how many violations it is involved in
        scores = np.zeros(len(prices))
        for i in np.flatnonzero(mono):
            scores[i] += 1
            scores[i + 1] += 1
        for i in np.flatnonzero(conv):
            scores[i : i + 3] += 1

        scores[0] = -1  # the max mono is never removed

        candidates = np.flatnonzero(scores == scores.max())
        if len(candidates) > 1 and has_spread:
            spreads = (group["best_offer"] - group["best_bid"]).to_numpy()[
                candidates - 1  # adjust for the added K=0
            ]
            # remove the candidate with the widest spread
            remove_idx = candidates[np.argmax(spreads)] - 1
        else:
            remove_idx = candidates[0] - 1

        group = group.drop(index=remove_idx).reset_index(drop=True)
        n_removed += 1

    return group, n_removed, n_mono, n_conv


def _calendar_drops(slices, S, r_map, q, tol=1e-8, rule="wide"):
    """
    Calendar-spread condition at fixed forward moneyness.

    With F_T = S exp((r-q)T), y = K / F_T and c(y, T) = C / (S exp(-qT)),
    the call price is non-decreasing in maturity iff, for T2 > T1,
        c(y, T2) >= c(y, T1)   for all y.

    c(., T2) is convex in y, so linear interpolation between its nodes
    (including the anchor c(0) = 1) overestimates it. Any T1 point lying
    above that interpolation is therefore a genuine violation.

    Returns a list of (exdate, strike) to drop: for each adjacent pair of
    maturities, the point involved in the largest violation.
    rule: "wide" (drop the wider bid-ask point of the pair),
          "short" (drop the shorter-maturity point) or
          "long" (drop the longer-maturity point).
    """
    mats = sorted(slices)
    info = {}
    for ex in mats:
        g = slices[ex]
        T = g["time_to_maturity"].iloc[0]
        F = S * np.exp((r_map[ex] - q) * T)
        scale = S * np.exp(-q * T)
        if {"best_bid", "best_offer"}.issubset(g.columns):
            rel_spread = (g["best_offer"] - g["best_bid"]).to_numpy() / scale
        else:
            rel_spread = np.zeros(len(g))
        info[ex] = {
            "strike": g["Strike"].to_numpy(),
            "y": g["Strike"].to_numpy() / F,
            "c": g["Mid"].to_numpy() / scale,
            "spread": rel_spread,
        }

    drops = []
    for e1, e2 in zip(mats[:-1], mats[1:]):
        a, b = info[e1], info[e2]
        y2 = np.concatenate(([0.0], b["y"]))
        c2 = np.concatenate(([1.0], b["c"]))

        inside = a["y"] <= y2.max()
        if not inside.any():
            continue

        interp = np.full(len(a["y"]), np.nan)
        interp[inside] = np.interp(a["y"][inside], y2, c2)
        gap = np.where(inside, a["c"] - interp, -np.inf)

        if gap.max() <= tol:
            continue

        i = int(np.argmax(gap))

        # nearest real node of the longer maturity
        pos = int(np.searchsorted(b["y"], a["y"][i]))
        cand = [p for p in (pos - 1, pos) if 0 <= p < len(b["y"])]
        j = min(cand, key=lambda p: abs(b["y"][p] - a["y"][i]))

        if rule == "short":
            drop_short = True
        elif rule == "long":
            drop_short = False
        else:
            drop_short = a["spread"][i] >= b["spread"][j]

        if drop_short:
            drops.append((e1, a["strike"][i]))
        else:
            drops.append((e2, b["strike"][j]))

    return drops


def remove_arbitrage_violations(
    df,
    price_underlying,
    r_by_maturity,
    q,
    verbose=False,
    tol=1e-8,
    calendar_tol=1e-8,
    min_obs=10,
    calendar_rule="wide",
):
    """
    Convert OTM quotes to calls and filter observations violating the
    sufficient no-arbitrage conditions for European call prices of
    Carr & Madan (2005):

        1. Price bounds.
        2. Monotonicity in strike (with anchor C(0,T) = S e^{-qT}).
        3. Convexity in strike (including the anchor).
        4. Calendar spreads: c(y, T) non-decreasing in T, where
           y = K / F_T is forward moneyness.
    """
    S0 = price_underlying
    df = df.copy()
    n_start = len(df)

    # ---------------------------------------------------------
    # 0. Basic validity: positive strike, positive maturity, known r
    # ---------------------------------------------------------
    r_map = {ex: r for ex, r in r_by_maturity.items()}
    df = df[
        (df["Strike"] > 0)
        & (df["time_to_maturity"] > 0)
        & df["exdate"].isin(r_map.keys())
    ].copy()
    n_invalid = n_start - len(df)

    # ---------------------------------------------------------
    # 1. Remove ITM options
    # ---------------------------------------------------------

    is_call = df["cp_flag"] == "C"
    is_put = df["cp_flag"] == "P"
    # OTM condition: calls with K >= S0, puts with K <= S0
    otm = (is_call & (df["Strike"] >= S0)) | (is_put & (df["Strike"] <= S0))

    n_before_itm = len(df)
    df = df[otm].copy()  # keep only OTM options
    n_itm_removed = n_before_itm - len(df)

    # ---------------------------------------------------------
    # 2. Convert puts to calls via parity: C = P + S e^{-qT} - K e^{-rT} and take the average mid of duplicates
    # ---------------------------------------------------------

    n_converted = 0
    puts = df[df["cp_flag"] == "P"].copy()
    calls = df[df["cp_flag"] == "C"].copy()
    if not puts.empty:
        for col in ["Mid", "best_bid", "best_offer"]:
            puts[col] = puts.apply(
                lambda row: implied_call_price(
                    S0,
                    row[col],
                    row["Strike"],
                    row["time_to_maturity"],
                    r_map[row["exdate"]],
                    q,
                ),
                axis=1,
            )
        puts["cp_flag"] = "C"  # change the cp_flag to C for converted puts

        # new df is all the original calls plus the converted puts
        df = pd.concat([calls, puts], ignore_index=True)

        # average quotes of duplicates (same exdate and strike)
        df = df.groupby(["exdate", "Strike"], as_index=False).agg(
            Mid=("Mid", "mean"),
            cp_flag=("cp_flag", "first"),
            time_to_maturity=("time_to_maturity", "first"),
            best_bid=("best_bid", "mean"),
            best_offer=("best_offer", "mean"),
        )
        n_converted = len(puts)
        n_original_calls = len(calls)

    n_total_calls = len(df)

    # ---------------------------------------------------------
    # 3. Individual price bounds
    # ---------------------------------------------------------
    T_row = df["time_to_maturity"]
    r_row = df["exdate"].map(r_map)
    lower = np.maximum(
        0.0, S0 * np.exp(-q * T_row) - df["Strike"] * np.exp(-r_row * T_row)
    )
    # upper = S * np.exp(-q * T_row)
    in_bound = df["Mid"] >= lower - tol  # & (df["Mid"] <= upper + tol)

    n_bounds_removed = int((~in_bound).sum())
    df = df[in_bound].copy()

    # ---------------------------------------------------------
    # 4. Strike-wise shape + calendar spreads, iterated to a fixed point
    # ---------------------------------------------------------
    slices = {ex: g.copy() for ex, g in df.groupby("exdate")}

    n_shape_removed = n_mono_found = n_conv_found = 0
    n_calendar_removed = n_maturity_removed = 0

    # 4a. strike-wise conditions per maturity
    for ex in list(slices):
        g = slices[ex]
        T = g["time_to_maturity"].iloc[0]
        g, n_rem, n_m, n_c = _clean_maturity_slice(g, S0, r_map[ex], q, T, tol)
        n_shape_removed += n_rem
        n_mono_found += n_m
        n_conv_found += n_c
        slices[ex] = g

    # TODO: figure out if we need calendar spread filtering.
    # # 4b. calendar-spread condition across adjacent maturities
    # if len(slices) >= 2:  # need at least 2 maturities
    #     drops = _calendar_drops(slices, S, r_map, q, calendar_tol, calendar_rule)
    #
    #     for ex, strike in drops:
    #         g = slices[ex]
    #         slices[ex] = g[g["Strike"] != strike]
    #         n_calendar_removed += 1

    # 4c. drop maturities that are too thin to define a smile
    for ex in list(slices):
        if len(slices[ex]) < min_obs:
            n_maturity_removed += len(slices[ex])
            del slices[ex]
    if slices:
        call_df = pd.concat([slices[ex] for ex in sorted(slices)], ignore_index=True)
    else:
        call_df = df.iloc[0:0].copy()

    # ---------------------------------------------------------
    # 5. Verbose output
    # ---------------------------------------------------------
    if verbose:
        print("\n--- Option data filtering ---")
        print(f"Initial observations:               {n_start}")
        print(f"Removed (invalid K/T/no r):         {n_invalid}")
        print(f"Removed ITM:                        {n_itm_removed}")
        print(f"Original calls:                     {n_original_calls}")
        print(f"Puts converted to calls:            {n_converted}")
        print(f"Total call observations:            {n_total_calls}")
        print(f"Removed by price bounds:            {n_bounds_removed}")
        print(f"Removed by slope/convexity:         {n_shape_removed}")
        print(f"  Monotonicity violations found:    {n_mono_found}")
        print(f"  Convexity violations found:       {n_conv_found}")
        # print(f"Removed by calendar spread:         {n_calendar_removed}")
        print(f"Removed by maturity filter:         {n_maturity_removed}")
        print("----------------------------------")
        print(f"Remaining observations:             {len(call_df)}")
        print(f"Remaining maturities:               {call_df['exdate'].nunique()}")

    return call_df


# =====================================================================
# Inference of r and q
# =====================================================================
def get_closest_options(df, price_underlying, num_options=10):
    df = df.copy()
    n_each_side = num_options // 2

    df["abs_diff"] = (df["Strike"] - price_underlying).abs()
    df = df.sort_values("abs_diff")

    closest_options = []

    for maturity in df["exdate"].unique():
        df_maturity = df[df["exdate"] == maturity]

        below = df_maturity[df_maturity["Strike"] < price_underlying]
        above = df_maturity[df_maturity["Strike"] > price_underlying]

        if len(below) < n_each_side:
            print(
                f"Warning: Not enough options below the underlying price for maturity {maturity.date()}. "
                f"Requested {n_each_side}, but found {len(below)}."
            )
        if len(above) < n_each_side:
            print(
                f"Warning: Not enough options above the underlying price for maturity {maturity.date()}. "
                f"Requested {n_each_side}, but found {len(above)}."
            )

        selected = pd.concat(
            [
                below.groupby("cp_flag").head(n_each_side),
                above.groupby("cp_flag").head(n_each_side),
            ]
        )

        puts = selected[selected["cp_flag"] == "P"]
        calls = selected[selected["cp_flag"] == "C"]

        row = pd.merge(
            puts,
            calls,
            on=["Strike", "exdate", "date", "time_to_maturity"],
            suffixes=("_put", "_call"),
        )

        closest_options.append(row)

    return (
        pd.concat(closest_options, ignore_index=True)
        if closest_options
        else pd.DataFrame()
    )


def put_call_parity_loss(S, C, P, K, T, r, q):
    P_calculated = implied_put_price(S, C, K, T, r, q)
    loss = np.abs(P_calculated - P)
    return loss


# loss function is put call parity loss for each maturity and strike price
def loss_function(closest_options, S, r_by_maturity, q):
    total_loss = 0

    for _, row in closest_options.iterrows():
        C = row["Mid_call"]
        P = row["Mid_put"]
        K = row["Strike"]
        T = row["time_to_maturity"]

        r = r_by_maturity[row["exdate"]]

        loss = put_call_parity_loss(S, C, P, K, T, r, q) ** 2
        total_loss += loss

    return total_loss


def infer_dividend_yield_and_discount_rate(
    closest_options, S, r_bounds=(0, 0.1), q_bounds=(0, 0.1)
):
    maturities = closest_options["exdate"].unique()
    n_maturities = len(maturities)

    def objective(params):
        q = params[0]
        r_values = params[1:]
        r_by_maturity = dict(zip(maturities, r_values))
        return loss_function(closest_options, S, r_by_maturity, q)

    # Initial guess inside the bounds: common q, one r per maturity
    initial_guess = [0.02] * (n_maturities + 1)
    bounds = [q_bounds] + [r_bounds] * n_maturities

    result = minimize(objective, initial_guess, bounds=bounds)

    if result.success:
        inferred_q = result.x[0]
        inferred_r = dict(zip(maturities, result.x[1:]))
        return inferred_r, inferred_q
    else:
        raise ValueError("Optimization failed: " + result.message)


# =====================================================================
# Cleaning and IV
# =====================================================================
def clean_data(df, price_underlying):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])

    # remove zero bids/asks and crossed quotes
    df = df[
        (df["best_bid"] > 0)
        & (df["best_offer"] > 0)
        & (df["best_offer"] >= df["best_bid"])
    ]
    return df


def add_IV_column(df, price_underlying, r_by_maturity, q):
    df = df.copy()
    df["implied_volatility"] = df.apply(
        lambda row: calculate_implied_volatility(
            price_underlying,
            row["Mid"],
            row["Strike"],
            row["time_to_maturity"],
            r_by_maturity[row["exdate"]],
            q,
            row["cp_flag"],
        ),
        axis=1,
    )
    return df


if __name__ == "__main__":
    d = load_data(verbose=True)
