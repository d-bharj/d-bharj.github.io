import os
import yfinance as yf
import pandas as pd
import numpy as np


def fetch_company_data(ticker):
    # pull what we need for one ticker yFinance gives a monster dict
    stock = yf.Ticker(ticker)
    info = stock.info

    # yfinance returns None or 0 for missing fields coerce to nan so
    # pivot merges downstream dont silently drop rows
    def safe(key):
        val = info.get(key, np.nan)
        return val if val is not None and val != 0 else np.nan

    # core capital structure line items
    return {
        "Ticker": ticker.upper(),
        "Market_Cap": safe("marketCap"),      # equity value
        "Total_Debt": safe("totalDebt"),      # book value of debt
        "Cash": safe("totalCash"),            # cash & equivalents
        "Revenue": safe("totalRevenue"),      # trailing twelve month
        "EBITDA": safe("ebitda"),             # earnings b4 int tax deprec
        "Net_Income": safe("netIncomeToCommon")  # attributable to common
    }


def fetch_all_data(target_ticker, peer_tickers):
    """target first then every peer all in one frame"""
    all_tickers = [target_ticker] + peer_tickers
    records = [fetch_company_data(t) for t in all_tickers]  # list comp over the tickers
    return pd.DataFrame(records).set_index("Ticker")


def compute_enterprise_values(df):
    # whole company equity we'd buy plus debt we inherit minus cash we keep
    df["Enterprise_Value"] = df["Market_Cap"] + df["Total_Debt"] - df["Cash"]
    return df


def compute_multiples(df):
    # three comps multiples all standard
    df["EV_EBITDA"] = df["Enterprise_Value"] / df["EBITDA"]  # workhorse of most comps
    df["EV_Sales"] = df["Enterprise_Value"] / df["Revenue"]  # matters when ebitda thin
    df["P_E"] = df["Market_Cap"] / df["Net_Income"]          # equity side not firm side
    return df


def compute_peer_stats(df, target_ticker):
    """drop the target out stats on the group thats left"""
    peers = df.drop(index=target_ticker, errors="ignore")
    ev_ebitda = peers["EV_EBITDA"].dropna()  # only clean ebitda rows ignore empties
    # percentile math flaky with fewer than 4 obs so guard it
    return {
        "P25_EBITDA": np.percentile(ev_ebitda, 25) if len(ev_ebitda) >= 4 else np.nan,
        "Median_EBITDA": np.median(ev_ebitda) if len(ev_ebitda) > 0 else np.nan,  # >=1 for median
        "P75_EBITDA": np.percentile(ev_ebitda, 75) if len(ev_ebitda) >= 4 else np.nan,
    }


def compute_implied_valuations(df, stats, target_ticker):
    """whats the target worth if it traded at peer multiple"""
    target_ebitda = df.loc[target_ticker, "EBITDA"]

    # company with no ebitda cant be keyed off a multiple bail
    if pd.isna(target_ebitda) or target_ebitda <= 0:
        return None

    # ttm ebitda x each percentile gives a low mid high EV range
    return {
        "Metric": "Implied Enterprise Value",
        "P25_EBITDA": target_ebitda * stats["P25_EBITDA"],
        "Median_EBITDA": target_ebitda * stats["Median_EBITDA"],
        "P75_EBITDA": target_ebitda * stats["P75_EBITDA"],
    }


def main():
    # change these defaults for a different base case
    target = "MSFT"
    peers = ["ORCL", "SAP", "CRM", "ADBE"]

    # surface current setup ask nicely if theyd like to swap it
    print(f"Current target: {target}")
    print(f"Current peers:  {', '.join(peers)}")
    change = input("Change tickers? (y/n): ").strip().lower()

    if change == "y":
        target = input("Target ticker: ").strip().upper()
        peer_input = input("Peer tickers (comma or space separated): ").strip()
        # split on commas OR spaces strip and uppercase each
        peers = [t.strip().upper() for t in peer_input.replace(",", " ").split() if t.strip()]

    print("\nFetching data...")

    # 1 get the tables
    df = fetch_all_data(target, peers)

    # 2 step through the ev calcs
    df = compute_enterprise_values(df)
    df = compute_multiples(df)
    stats = compute_peer_stats(df, target)
    implied = compute_implied_valuations(df, stats, target)

    # 3 write the whole thing out
    os.makedirs("output", exist_ok=True)
    path = "output/mna_comps_valuation.csv"

    # lee first the per-company table then the benchmark beneath it
    export = df.copy()
    export.index.name = "Ticker"
    export.reset_index(inplace=True)
    export.to_csv(path)

    # dont appending summary blocks at the foot of the csv
    with open(path, "a") as f:
        f.write("\n\n--- Peer EV/EBITDA Statistics ---\n")
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                  f.write(f"{k},{v:.2f}\n")
            else:
                f.write(f"{k},{v}\n")
        if implied:
            f.write(f"\n--- Implied Valuation (Target: {target}) ---\n")
            for k, v in implied.items():
                if isinstance(v, (int, float)):
                    f.write(f"{k},{v:.2f}\n")
                else:
                    f.write(f"{k},{v}\n")

    print(f"\nDone. Exported to {path}")
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()