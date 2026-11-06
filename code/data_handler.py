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


def load_data(file_path, config):
    sheet = config["sheet_name"]

    with pd.ExcelFile(file_path) as workbook:
        df = pd.read_excel(
            workbook,
            sheet_name=sheet,
            usecols=config["usecols"],
            skiprows=config["start_row"] - 1,
            nrows=config["end_row"] - config["start_row"] + 1,
        )

        close_price = pd.read_excel(
            workbook,
            sheet_name=sheet,
            usecols=config["close_cell"][0],
            skiprows=int("".join(filter(str.isdigit, config["close_cell"]))) - 1,
            nrows=1,
            header=None,
        ).iat[0, 0]

    return df, close_price


# Get the five closest options for each maturity on each side closest to ATM
def get_closest_options(df, price_underlying, num_options=5):
    df = df.copy()  # Create a copy to avoid modifying the original DataFrame
    # exclude options that are not of the specified type
    df["abs_diff"] = (df["Strike"] - price_underlying).abs()
    df.sort_values(by="abs_diff", inplace=True)

    closest_options = pd.DataFrame()
    for maturity in df["exdate"].unique():
        # Filter the DataFrame for the current maturity
        df_maturity = df[df["exdate"] == maturity]
        puts = df_maturity[df_maturity["cp_flag"] == "P"].head(num_options)
        calls = df_maturity[df_maturity["cp_flag"] == "C"].head(num_options)

        # We assume that the puts and calls have the same strike
        row = pd.merge(
            puts, calls, on=["Strike", "exdate", "date"], suffixes=("_put", "_call")
        )
        closest_options = pd.concat([closest_options, row], ignore_index=True)

    return closest_options


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


if __name__ == "__main__":
    # Load the data
    df, close_price = load_data(FILE_PATH, SHEETS["SPX_options"])

    # Get the closest options
    closest_options = get_closest_options(df, close_price)
    print("Closest options:")
    print(closest_options)
    closest_options["time_to_maturity"] = (
        closest_options["exdate"] - closest_options["date"]
    ).dt.days / 365.0

    # Infer dividend yield and discount rates
    inferred_r, inferred_q = infer_dividend_yield_and_discount_rate(
        closest_options, close_price
    )

    print(f"Inferred dividend yield (q): {inferred_q:.6f}")
    print("\nInferred discount rates:")

    for maturity, rate in inferred_r.items():
        print(f"{maturity.date()}: {rate:.6f}")
