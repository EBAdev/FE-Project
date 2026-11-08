import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.stats import norm

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
    ITM=False,
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
        return_value["r_by_maturity"] = inferred_r
        return_value["q"] = inferred_q

    if clean:
        df = clean_data(df, price_underlying)
        return_value["cleaned_dataframe"] = df

    if not ITM:
        # remove ITM opions
        df = df[
            (df["cp_flag"] == "C") & (df["Strike"] > price_underlying)
            | (df["cp_flag"] == "P") & (df["Strike"] < price_underlying)
        ]
        return_value["cleaned_dataframe"] = df

    return return_value


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
            on=["Strike", "exdate", "date", "time_to_maturity"],
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
    return df


def BS_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def BS_put(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def calculate_implied_volatility(
    price_underlying, option_mid, strike, time_to_maturity, r, q, option_type
):
    # Define the function to find the root of
    def objective(sigma):
        if option_type == "C":
            return (
                BS_call(price_underlying, strike, time_to_maturity, r, q, sigma)
                - option_mid
            )
        else:
            return (
                BS_put(price_underlying, strike, time_to_maturity, r, q, sigma)
                - option_mid
            )

    # Use Brent's method to find the implied volatility
    try:
        implied_vol = brentq(objective, 1e-6, 5)
    except ValueError:
        implied_vol = np.nan  # Return NaN if no solution is found

    return implied_vol


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
