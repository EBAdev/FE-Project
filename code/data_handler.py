import datetime as dt

import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def load_data(file_path=FILE_PATH, config=SHEETS["SPX_options"]):
    sheet = config["sheet_name"]

    with pd.ExcelFile(file_path) as workbook:
        df = pd.read_excel(
            workbook,
            sheet_name=sheet,
            usecols=config["usecols"],
            skiprows=config["start_row"] - 1,
            nrows=config["end_row"] - config["start_row"] + 1,
        )

        underlying_close = pd.read_excel(
            workbook,
            sheet_name=sheet,
            usecols=config["close_cell"][0],
            skiprows=int("".join(filter(str.isdigit, config["close_cell"]))) - 1,
            nrows=1,
            header=None,
        ).iat[0, 0]

    return df, underlying_close


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
            on=["Strike", "exdate", "date"],
            suffixes=("_put", "_call"),
        )

        closest_options.append(row)

    return (
        pd.concat(closest_options, ignore_index=True)
        if closest_options
        else pd.DataFrame()
    )


# put call parity price
def implied_put_price(S, C, K, T, r, q):
    # Calculate the put price using put-call parity
    P = C + K * np.exp(-r * T) - S * np.exp(-q * T)
    return P


def put_call_parity_loss(S, C, P, K, T, r, q):
    # Calculate the put price using put-call parity
    P_calculated = implied_put_price(S, C, K, T, r, q)
    # Calculate the loss as the absolute difference between the calculated and actual put prices
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
    # remove ITM opions
    df = df[
        (df["cp_flag"] == "C") & (df["Strike"] > price_underlying)
        | (df["cp_flag"] == "P") & (df["Strike"] < price_underlying)
    ]

    return df


def load_and_clean_data(file_path=FILE_PATH, config=SHEETS["SPX_options"]):
    df, underlying_close = load_data(file_path, config)
    df = clean_data(df)
    return df, underlying_close


if __name__ == "__main__":
    # Load the data
    df, underlying_close = load_data()

    # Get the closest options
    closest_options = get_closest_options(df, underlying_close)
    closest_options["time_to_maturity"] = (
        closest_options["exdate"] - closest_options["date"]
    ).dt.days / 365.0

    # Infer dividend yield and discount rates
    inferred_r, inferred_q = infer_dividend_yield_and_discount_rate(
        closest_options, underlying_close
    )

    print(f"Inferred dividend yield (q): {inferred_q:.6f}")
    print("\nInferred discount rates:")

    for maturity, rate in inferred_r.items():
        print(f"{maturity.date()}: {rate:.6f}")

    clean_df = clean_data(df, underlying_close)
    # clean the data and print the removed rows
    removed_rows = df[~df.index.isin(clean_df.index)]

    print(f"\nCleaning removed: {removed_rows.shape[0]} rows")
    print(f"Remaining rows after cleaning: {len(clean_df)}")
    print(f"Final cleaned data:\n{clean_df.head()}")
