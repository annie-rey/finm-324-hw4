import pandas as pd
import argparse
import os

def load_data(quotes_path, executions_path):
    """Docstring"""
    quotes = pd.read_csv(quotes_path, dtype = {"ticker": "category"})
    quotes["sip_timestamp"] = pd.to_datetime(quotes["sip_timestamp"], unit="ns")

    executions = pd.read_csv(executions_path, dtype={"symbol": "category"})
    executions["execution_time"] = pd.to_datetime(executions["execution_time"])
    executions["order_time"] = pd.to_datetime(executions["order_time"])

    return quotes, executions

def match_categories(quotes, executions):
    """Docstring"""
    #setting the categories to the same values prevents errors with merge_asof down the line
    executions_tickers = set(executions["symbol"].cat.categories)
    quotes_tickers = set(quotes["ticker"].cat.categories)
    all_tickers = executions_tickers.union(quotes_tickers)
    executions.symbol = executions.symbol.cat.set_categories(all_tickers)
    quotes.ticker = quotes.ticker.cat.set_categories(all_tickers)

    return quotes, executions

def trim_to_market_hours(quotes, executions, time_columns, day):
    quotes_time_columns = [col for col in time_columns
                           if col in quotes.columns]
    for col in quotes_time_columns:
        #using .loc because .truncate was overloading my computer?
        quotes = quotes.loc[(quotes[col] >= f"{day} 09:30") & (quotes[col] <= f"{day} 16:00")]
    executions_time_columns = [col for col in time_columns
                               if col in executions.columns]
    for col in executions_time_columns:
        executions = executions.loc[(executions[col] >= f"{day} 09:30") & (executions[col] <= f"{day} 16:00")]
        
    return quotes, executions


def merge_data(quotes, executions):
    """Docstring"""
    executions = executions.sort_values(by=["order_time", "symbol"])
    quotes = quotes.sort_values(by=["sip_timestamp", "ticker"])
    
    merged = pd.merge_asof(
        executions,
        quotes,
        left_on='order_time',
        right_on='sip_timestamp',
        left_by='symbol',
        right_by='ticker',
        direction='backward'
    )

    return merged

def clean_merged_data(merged_data):
    """Docstring"""

    merged_data.dropna(inplace=True)
    merged_data.reset_index(inplace=True)
    merged_data.drop(["index"], axis=1, inplace=True)

    return merged_data

def add_price_improvement(merged_data):
    """Docstring"""

    merged_data['price_improvement'] = ((merged_data['limit_price'] - merged_data['execution_price'])*(merged_data['side'] == 1) + 
                                        (merged_data['execution_price'] - merged_data['limit_price'])*(merged_data['side'] != 1))
    
    return merged_data

def export_merged_data(merged_data, export_path):
    merged_data.to_csv(export_path)

def parse_args():
    """Docstring"""
    parser = argparse.ArgumentParser()

    #helper functions to raise helpful errors if the arguments are input incorrectly
    def input_quotes_file(path):
        if not path.endswith('.csv.gz'):
            raise argparse.ArgumentTypeError("Input quotes file must end with .csv.gz")
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        return path 
    
    def input_executions_file(path):
        if not path.endswith('.csv'):
            raise argparse.ArgumentTypeError("Input executions file must end with .csv")
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        return path
    
    def output_csv_file(path):
        if not path.endswith('.csv'):
            raise argparse.ArgumentTypeError("Output file must end with .csv")
        return path 
    
    #arguments are defined using the helper functions above for helpful and robust error handling
    parser.add_argument('--input_quotes_file', type=input_quotes_file,
                        help="path to (compressed) quotes csv file")
    parser.add_argument('--input_executions_file', type=input_executions_file,
                        help="path to executions csv file")
    parser.add_argument('--output_csv_file', type=output_csv_file,
                        help='path to store the resulting merged csv file under')
    args = parser.parse_args()

    return parser, args


def main():
    """Docstring"""

    parser, args = parse_args()
    if (not args.input_quotes_file or 
        not args.input_executions_file or 
        not args.output_csv_file):
        parser.error("All of --input_quotes_file --input_executions_file and --output_csv_file are required.")

    TIME_COLUMNS = ["sip_timestamp", "order_time", "execution_time"]
    DAY = "2025-09-10"

    print("loading...")
    quotes, executions = load_data(args.input_quotes_file, args.input_executions_file)
    print("matching...")
    quotes, executions = match_categories(quotes, executions)
    print("trimming...")
    quotes, executions = trim_to_market_hours(quotes, executions, TIME_COLUMNS, DAY)
    print("merging...")
    merged_data = merge_data(quotes, executions)
    print("cleaning...")
    merged_data = clean_merged_data(merged_data)
    print("adding...")
    merged_data = add_price_improvement(merged_data)
    print("exporting...")
    export_merged_data(merged_data, args.output_csv_file)

if __name__ == "__main__":
    main()
    

