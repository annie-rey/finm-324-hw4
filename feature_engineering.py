import pandas as pd
import argparse
import os
import sys
from datetime import datetime

#Constants
TIME_COLUMNS = ["sip_timestamp", "order_time", "execution_time"]
DAY = "2025-09-10"


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


def load_data(quotes_path, executions_path):
    """Load quotes and executions CSV files and parse datetime columns."""
    # Load quotes
    try:
        quotes = pd.read_csv(quotes_path, dtype={"ticker": "category"})
    except FileNotFoundError:
        raise DataProcessingError(f"Quotes file not found: {quotes_path}")
    except pd.errors.EmptyDataError:
        raise DataProcessingError(f"Quotes file is empty: {quotes_path}")
    except pd.errors.ParserError as exc:
        raise DataProcessingError(f"Error parsing quotes file {quotes_path}: {exc}")

    # Validate required columns in quotes
    for col in ["ticker", "sip_timestamp"]:
        if col not in quotes.columns:
            raise DataProcessingError(
                f"Quotes file {quotes_path} is missing required column '{col}'. "
                f"Available columns: {list(quotes.columns)}"
            )

    # Parse quotes timestamp
    try:
        quotes["sip_timestamp"] = pd.to_datetime(quotes["sip_timestamp"], unit="ns", errors="raise")
    except (ValueError, TypeError) as exc:
        raise DataProcessingError(
            f"Failed to convert 'sip_timestamp' to datetime in quotes file {quotes_path}: {exc}"
        )

    # Load executions
    try:
        executions = pd.read_csv(executions_path, dtype={"symbol": "category"})
    except FileNotFoundError:
        raise DataProcessingError(f"Executions file not found: {executions_path}")
    except pd.errors.EmptyDataError:
        raise DataProcessingError(f"Executions file is empty: {executions_path}")
    except pd.errors.ParserError as exc:
        raise DataProcessingError(f"Error parsing executions file {executions_path}: {exc}")

    # Validate required columns in executions
    for col in ["symbol", "execution_time", "order_time", "limit_price", "execution_price", "side"]:
        if col not in executions.columns:
            raise DataProcessingError(
                f"Executions file {executions_path} is missing required column '{col}'. "
                f"Available columns: {list(executions.columns)}"
            )

    # Parse execution timestamps
    for time_col in ["execution_time", "order_time"]:
        try:
            executions[time_col] = pd.to_datetime(executions[time_col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise DataProcessingError(
                f"Failed to convert '{time_col}' to datetime in executions file {executions_path}: {exc}"
            )

    return quotes, executions


def match_categories(quotes, executions):
    """Align categorical categories between quotes['ticker'] and executions['symbol']."""
    if "ticker" not in quotes.columns or "symbol" not in executions.columns:
        raise DataProcessingError("Cannot match categories: 'ticker' or 'symbol' column missing.")

    # Ensure both are categorical
    if not pd.api.types.is_categorical_dtype(quotes["ticker"]):
        quotes["ticker"] = quotes["ticker"].astype("category")

    if not pd.api.types.is_categorical_dtype(executions["symbol"]):
        executions["symbol"] = executions["symbol"].astype("category")

    try:
        executions_tickers = set(executions["symbol"].cat.categories)
        quotes_tickers = set(quotes["ticker"].cat.categories)
        all_tickers = executions_tickers.union(quotes_tickers)

        executions["symbol"] = executions["symbol"].cat.set_categories(all_tickers)
        quotes["ticker"] = quotes["ticker"].cat.set_categories(all_tickers)
    except Exception as exc:
        raise DataProcessingError(f"Failed to match ticker categories: {exc}")

    return quotes, executions


def trim_to_market_hours(quotes, executions, time_columns, day):
    """
    Trim quotes and executions to market hours on a given day.

    time_columns: list of timestamp column names to consider.
    day: string date like 'YYYY-MM-DD'.
    """
    # Validate day format
    try:
        _ = datetime.strptime(day, "%Y-%m-%d")
    except ValueError as exc:
        raise DataProcessingError(f"DAY must be in 'YYYY-MM-DD' format, got '{day}': {exc}")

    market_open = f"{day} 09:30"
    market_close = f"{day} 16:00"

    # Quotes
    quotes_time_columns = [col for col in time_columns if col in quotes.columns]
    for col in quotes_time_columns:
        if not pd.api.types.is_datetime64_any_dtype(quotes[col]):
            raise DataProcessingError(
                f"Column '{col}' in quotes must be datetime before trimming to market hours."
            )
        quotes = quotes.loc[(quotes[col] >= market_open) & (quotes[col] <= market_close)]

    # Executions
    executions_time_columns = [col for col in time_columns if col in executions.columns]
    for col in executions_time_columns:
        if not pd.api.types.is_datetime64_any_dtype(executions[col]):
            raise DataProcessingError(
                f"Column '{col}' in executions must be datetime before trimming to market hours."
            )
        executions = executions.loc[(executions[col] >= market_open) & (executions[col] <= market_close)]

    if quotes.empty:
        raise DataProcessingError(
            f"No quote rows remain after trimming to market hours on {day}. "
            "Check that the DAY argument matches the data."
        )

    if executions.empty:
        raise DataProcessingError(
            f"No execution rows remain after trimming to market hours on {day}. "
            "Check that the DAY argument matches the data."
        )

    return quotes, executions

def binarize_order_side(executions):
    """Make 'side' a binary variable with value 1 if side=="BUY" and 0 if side=="SELL" """
    executions["side"] = executions["side"].apply(lambda x:x if x==1 else 0)
    return executions


def merge_data(quotes, executions):
    """Merge executions with the most recent quotes using merge_asof()."""
    required_exec_cols = {"order_time", "symbol"}
    required_quote_cols = {"sip_timestamp", "ticker"}

    if not required_exec_cols.issubset(executions.columns):
        raise DataProcessingError(
            f"Executions DataFrame missing columns for merge: {required_exec_cols - set(executions.columns)}"
        )
    if not required_quote_cols.issubset(quotes.columns):
        raise DataProcessingError(
            f"Quotes DataFrame missing columns for merge: {required_quote_cols - set(quotes.columns)}"
        )

    # Sort before merge_asof
    try:
        executions = executions.sort_values(by=["order_time", "symbol"])
        quotes = quotes.sort_values(by=["sip_timestamp", "ticker"])
    except KeyError as exc:
        raise DataProcessingError(f"Failed to sort before merge: {exc}")

    try:
        merged = pd.merge_asof(
            executions,
            quotes,
            left_on="order_time",
            right_on="sip_timestamp",
            left_by="symbol",
            right_by="ticker",
            direction="backward",
        )
    except Exception as exc:
        raise DataProcessingError(f"Error during merge_asof operation: {exc}")

    if merged.empty:
        raise DataProcessingError("Merged DataFrame is empty after merge_asof().")

    return merged


def clean_merged_data(merged_data):
    """Drop rows with NaNs and reset index."""
    if not isinstance(merged_data, pd.DataFrame):
        raise DataProcessingError("clean_merged_data expects a pandas DataFrame.")

    before = len(merged_data)
    merged_data = merged_data.dropna().reset_index(drop=True)
    after = len(merged_data)

    if after == 0:
        raise DataProcessingError(
            f"All rows were dropped during cleaning (dropna). Input rows: {before}, remaining: {after}."
        )

    return merged_data


def add_price_improvement(merged_data):
    """Add a 'price_improvement' column based on side, limit_price, and execution_price."""
    required_cols = {"limit_price", "execution_price", "side"}
    if not required_cols.issubset(merged_data.columns):
        raise DataProcessingError(
            f"Cannot compute price_improvement. Missing columns: {required_cols - set(merged_data.columns)}"
        )

    # Validate numeric types
    for col in ["limit_price", "execution_price"]:
        if not pd.api.types.is_numeric_dtype(merged_data[col]):
            raise DataProcessingError(f"Column '{col}' must be numeric to compute price improvement.")

    # side should be something we can compare to 1
    if not pd.api.types.is_numeric_dtype(merged_data["side"]):
        raise DataProcessingError("Column 'side' must be numeric (e.g., 1 for buy, 0 or -1 for sell).")

    try:
        merged_data["price_improvement"] = (
            (merged_data["limit_price"] - merged_data["execution_price"]) * (merged_data["side"] == 1)
            + (merged_data["execution_price"] - merged_data["limit_price"]) * (merged_data["side"] != 1)
        )
    except Exception as exc:
        raise DataProcessingError(f"Failed to compute price_improvement: {exc}")

    return merged_data


def export_merged_data(merged_data, export_path):
    """Export merged data to CSV."""
    try:
        merged_data.to_csv(export_path, index=False)
    except OSError as exc:
        raise DataProcessingError(f"Failed to write output CSV to '{export_path}': {exc}")


def parse_args():
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(description="Merge quotes and executions and compute price improvement.")

    # helper functions to raise helpful errors if the arguments are input incorrectly
    def input_quotes_file(path):
        if not path.endswith(".csv.gz"):
            raise argparse.ArgumentTypeError("Input quotes file must end with .csv.gz")
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        return path

    def input_executions_file(path):
        if not path.endswith(".csv"):
            raise argparse.ArgumentTypeError("Input executions file must end with .csv")
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        return path

    def output_csv_file(path):
        if not path.endswith(".csv"):
            raise argparse.ArgumentTypeError("Output file must end with .csv")
        # directory existence check (if directory is given)
        dir_name = os.path.dirname(path)
        if dir_name and not os.path.exists(dir_name):
            raise argparse.ArgumentTypeError(f"Directory does not exist for output path: {dir_name}")
        return path

    # arguments are defined using the helper functions above for helpful and robust error handling
    parser.add_argument(
        "--input_quotes_file",
        required=True,
        type=input_quotes_file,
        help="path to (compressed) quotes csv.gz file",
    )
    parser.add_argument(
        "--input_executions_file",
        required=True,
        type=input_executions_file,
        help="path to executions csv file",
    )
    parser.add_argument(
        "--output_csv_file",
        required=True,
        type=output_csv_file,
        help="path to store the resulting merged csv file under",
    )

    args = parser.parse_args()
    return parser, args


def main():
    """Main script."""
    parser, args = parse_args()

    # Ensure all the arguments needed are present
    missing = []
    if not getattr(args, "input_quotes_file", None):
        missing.append("--input_quotes_file")
    if not getattr(args, "input_executions_file", None):
        missing.append("--input_executions_file")
    if not getattr(args, "output_csv_file", None):
        missing.append("--output_csv_file")

    if len(missing) > 0:
        raise ValueError(
            f"Missing required command-line arguments: {missing}. "
            f"Run with --help to see usage information."
        )

    try:
        print("loading...")
        quotes, executions = load_data(args.input_quotes_file, args.input_executions_file)

        print("matching...")
        quotes, executions = match_categories(quotes, executions)

        print("trimming to market hours...")
        quotes, executions = trim_to_market_hours(quotes, executions, TIME_COLUMNS, DAY)

        print("binarizing order side...")
        executions = binarize_order_side(executions)

        print("merging...")
        merged_data = merge_data(quotes, executions)

        print("cleaning...")
        merged_data = clean_merged_data(merged_data)

        print("adding price improvement...")
        merged_data = add_price_improvement(merged_data)

        print("exporting...")
        export_merged_data(merged_data, args.output_csv_file)

        print("Done.")

    except DataProcessingError as exc:
        # Known, user-facing errors
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        # Unexpected errors – still fail cleanly
        print("An unexpected error occurred:", file=sys.stderr)
        print(repr(exc), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
