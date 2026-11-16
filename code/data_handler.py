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


def load_data(
    file_path=FILE_PATH,
    config=SHEETS["SPX_options"],
    clean=True,
    infer_q_r=True,
    NoArb=True,
    verbose=False,
):
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

    if infer_q_r:
        closest_options = get_closest_options(df, price_underlying)
        inferred_r, inferred_q = infer_dividend_yield_and_discount_rate(
            closest_options, price_underlying
        )
        if verbose:
            print("Inferred discount rates by maturity:")
            for exdate, r in inferred_r.items():
                print(f"{exdate.date()}: {r:.4%}")
            print(f"Inferred dividend yield: {inferred_q:.4%}")
        return_value["r_by_maturity"] = inferred_r
        return_value["q"] = inferred_q

    if clean:
        c_df = clean_data(df, price_underlying)
        if verbose:
            print(f"Initial number of observations: {len(df)}")
            print(f"Number of observations after cleaning: {len(c_df)}")
        return_value["cleaned_dataframe"] = c_df

    if NoArb:
        narb_df = remove_arbitrage_violations(
            c_df,
            price_underlying,
            return_value["r_by_maturity"],
            return_value["q"],
            verbose=verbose,
        )

        return_value["cleaned_dataframe"] = narb_df

    return return_value


def remove_arbitrage_violations(
    df,
    price_underlying,
    r_by_maturity,
    q,
    verbose=False,
    tol=1e-8,
):
    """
    Convert puts to calls and filter observations violating
    European call no-arbitrage conditions.

    Conditions:
        1. Discounted lower and upper price bounds.
        2. Monotonicity in strike.
        3. Convexity in strike.

    """

    df = df.copy()
    # ---------------------------------------------------------
    # 0. Remove ITM options
    # ---------------------------------------------------------
    n_before_itm = len(df)
    df = df[
        ~(
            (df["cp_flag"] == "C") & (df["Strike"] < price_underlying)
            | (df["cp_flag"] == "P") & (df["Strike"] > price_underlying)
        )
    ]
    n_itm_removed = n_before_itm - len(df)

    call_df = df[df["cp_flag"] == "C"].copy()
    put_df = df[df["cp_flag"] == "P"]

    # ---------------------------------------------------------
    # 1. Convert puts to calls using put-call parity
    # ---------------------------------------------------------
    for _, row in put_df.iterrows():
        exdate = row["exdate"]
        K = row["Strike"]
        T = row["time_to_maturity"]
        r = r_by_maturity[exdate]

        C_calculated = implied_call_price(price_underlying, row["Mid"], K, T, r, q)

        mask = (call_df["exdate"] == exdate) & (call_df["Strike"] == K)

        if mask.any():
            call_df.loc[mask, "Mid"] = (call_df.loc[mask, "Mid"] + C_calculated) / 2
        else:
            new_row = row.copy()
            new_row["cp_flag"] = "C"
            new_row["Mid"] = C_calculated

            call_df = pd.concat(
                [call_df, pd.DataFrame([new_row])],
                ignore_index=True,
            )

    call_df = call_df.reset_index(drop=True)

    n_converted = len(put_df)
    n_initial = len(call_df)
    n_original_calls = len(df[df["cp_flag"] == "C"])

    # ---------------------------------------------------------
    # 2. Check individual price bounds
    # ---------------------------------------------------------
    valid = pd.Series(True, index=call_df.index)

    for exdate, group in call_df.groupby("exdate"):
        r = r_by_maturity[exdate]
        T = group["time_to_maturity"].iloc[0]

        strikes = group["Strike"].to_numpy()
        prices = group["Mid"].to_numpy()

        lower_bound = np.maximum(
            0,
            price_underlying * np.exp(-q * T) - strikes * np.exp(-r * T),
        )

        upper_bound = price_underlying * np.exp(-q * T)

        group_valid = (prices >= lower_bound - tol) & (prices <= upper_bound + tol)

        valid.loc[group.index] = group_valid

    n_before_bounds = len(call_df)
    call_df = call_df.loc[valid].copy()
    n_bounds_removed = n_before_bounds - len(call_df)

    # ---------------------------------------------------------
    # 3. Check monotonicity and convexity iteratively
    # ---------------------------------------------------------
    cleaned_groups = []
    n_shape_removed = 0
    n_monotonicity_violations = 0
    n_convexity_violations = 0

    for exdate, group in call_df.groupby("exdate"):
        group = group.sort_values("Strike").copy()
        group = group.reset_index(drop=True)

        r = r_by_maturity[exdate]
        T = group["time_to_maturity"].iloc[0]
        max_slope = np.exp(-r * T)

        if group["Strike"].duplicated().any():
            raise ValueError(
                f"Duplicate strikes found for maturity {exdate.date()}. "
                "Consolidate duplicate quotes before filtering."
            )

        while len(group) >= 2:
            strikes = group["Strike"].to_numpy()
            prices = group["Mid"].to_numpy()

            slopes = (prices[:-1] - prices[1:]) / np.diff(strikes)

            # Monotonicity and slope upper bound
            mono_violations = (slopes < -tol) | (slopes > max_slope + tol)

            # Convexity
            convex_violations = (
                (slopes[:-1] < slopes[1:] - tol)
                if len(slopes) >= 2
                else np.array([], dtype=bool)
            )

            if not mono_violations.any() and not convex_violations.any():
                break

            n_monotonicity_violations += int(mono_violations.sum())
            n_convexity_violations += int(convex_violations.sum())

            # Count violations involving each observation
            scores = np.zeros(len(group))

            for i in np.where(mono_violations)[0]:
                scores[i] += 1
                scores[i + 1] += 1

            for i in np.where(convex_violations)[0]:
                scores[i] += 1
                scores[i + 1] += 1
                scores[i + 2] += 1

            # Remove observation involved in most violations
            candidates = np.flatnonzero(scores == scores.max())

            # Break ties using the widest bid-ask spread
            if len(candidates) > 1 and {"best_bid", "best_offer"}.issubset(
                group.columns
            ):

                spreads = (
                    group.loc[candidates, "best_offer"].to_numpy()
                    - group.loc[candidates, "best_bid"].to_numpy()
                )

                remove_idx = candidates[np.argmax(spreads)]
            else:
                remove_idx = candidates[0]

            group = group.drop(index=remove_idx).reset_index(drop=True)
            n_shape_removed += 1

        cleaned_groups.append(group)

    call_df = pd.concat(cleaned_groups, ignore_index=True)

    # ---------------------------------------------------------
    # 5. Remove maturities with fewer than 10 observations
    # ---------------------------------------------------------
    maturity_counts = call_df.groupby("exdate").size()

    valid_maturities = maturity_counts[maturity_counts >= 10].index

    n_before_maturity = len(call_df)

    call_df = call_df[call_df["exdate"].isin(valid_maturities)].copy()

    n_maturity_removed = n_before_maturity - len(call_df)

    # ---------------------------------------------------------
    # 6. Verbose output
    # ---------------------------------------------------------
    if verbose:
        print("\n--- Option data filtering ---")
        print(f"Initial observations:               {n_before_itm}")
        print(f"Removed ITM options:               {n_itm_removed}")
        print(f"Original call observations:        {n_original_calls}")
        print(f"Puts converted to calls:           {n_converted}")
        print(f"Total call observations:           {n_initial}")
        print(f"Removed by price bounds:           {n_bounds_removed}")
        print(f"Removed by slope/convexity:        {n_shape_removed}")
        print(f"  Monotonicity violations found:   {n_monotonicity_violations}")
        print(f"  Convexity violations found:      {n_convexity_violations}")
        print(f"Removed by maturity filter:        {n_maturity_removed}")
        print("----------------------------------")
        print(f"Remaining observations:            {len(call_df)}")
        print(f"Remaining maturities:              {call_df['exdate'].nunique()}")

    return call_df.reset_index(drop=True)


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

    # Define the objective function
    def objective(params):
        q = params[0]
        r_values = params[1:]

        r_by_maturity = dict(zip(maturities, r_values))

        return loss_function(closest_options, S, r_by_maturity, q)

    # Initial guess: common q and one r for each maturity
    initial_guess = [0.2] * (n_maturities + 1)

    # Bounds: one bound for q and one for each maturity-specific r
    bounds = [q_bounds] + [r_bounds] * n_maturities

    # Perform optimization
    result = minimize(objective, initial_guess, bounds=bounds)

    if result.success:
        inferred_q = result.x[0]
        inferred_r = dict(zip(maturities, result.x[1:]))

        return inferred_r, inferred_q
    else:
        raise ValueError("Optimization failed: " + result.message)


def clean_data(df, price_underlying):
    # Convert date columns to datetime
    df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["exdate"] = pd.to_datetime(df["exdate"])

    # remove rows with a 0 bid or ask price
    df = df[(df["best_bid"] > 0) & (df["best_offer"] > 0)]
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
    data = load_data(verbose=True)
    df = data["cleaned_dataframe"]
    price_underlying = data["underlying_close"]
    r_by_maturity = data["r_by_maturity"]
    q = data["q"]
