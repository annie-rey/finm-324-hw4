import pandas as pd
import argparse
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer, make_column_transformer

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import root_mean_squared_error, r2_score

from sklearn import pipeline

from joblib import dump

NUMERIC_FEATURES = ["order_qty", "limit_price", "bid_price", "ask_price", "bid_size", "ask_size"]
BINARY_FEATURE = ["side"]

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def get_exchange_data(exchange, data):
    exchange_data = data.loc[data.exchange == f"ID{exchange}"]

    X = exchange_data[NUMERIC_FEATURES + BINARY_FEATURE]
    y = exchange_data["price_improvement"]

    return X, y

def split_exchange_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test

def build_exchange_pipeline(model, power_transform=False, scale_binary=True):

    numeric_pipeline = pipeline.make_pipeline(
        ("power", PowerTransformer()),
        ("scale", StandardScaler())
    )

    numeric_preprocessor = numeric_pipeline if power_transform == True else StandardScaler()
    binary_preprocessor = StandardScaler() if scale_binary == True else "passthrough"

    preprocessor = make_column_transformer(
        (numeric_preprocessor, NUMERIC_FEATURES),
        (binary_preprocessor, BINARY_FEATURE)
    )

    pipe = pipeline.make_pipeline(
        preprocessor,
        model
    )

    return pipe

def train_exchange_pipeline(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    pipe.score(X_test, y_test)

def export_exchange_model(pipe, output_path):
    dump(pipe, output_path)

def parse_args():
    """Docstring"""
    parser = argparse.ArgumentParser()

    #helper functions to raise helpful errors if the arguments are input incorrectly
    def input_csv_file(path):
        if not path.endswith('.csv'):
            raise argparse.ArgumentTypeError("Input file must end with .csv")
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        return path
    
    def output_joblib_file(path):
        if not path.endswith('/'):
            raise argparse.ArgumentTypeError("Output root directory must end with /")
        return path 
    
    #arguments are defined using the helper functions above for helpful and robust error handling
    parser.add_argument('--input_csv_file', type=input_csv_file,
                        help="path to input csv file")
    parser.add_argument('--output_joblib_root_dir', type=output_joblib_file,
                        help='root path to store the resulting models under')
    args = parser.parse_args()

    return parser, args

def main():
    parser, args = parse_args()

    data = load_data(args.input_csv_file)

    EXCHANGES = ["1516", "29608"]
    
    preprocessing = {"1516": (False, False),
                     "29608": (False, True)}
    
    models = {"1516": ExtraTreesRegressor(n_jobs=-1, random_state=731, max_features='sqrt',
                                          min_samples_leaf=1, n_estimators=100),
              "29608": ExtraTreesRegressor(n_jobs=-1, random_state=731, max_features='sqrt',
                                           min_samples_leaf=1, n_estimators=100),
              }
    
    for exchange in EXCHANGES:
        power_transform, scale_binary = preprocessing[exchange]
        model = models[exchange]

        X, y = get_exchange_data(exchange, data)
        X_train, X_test, y_train, y_test = split_exchange_data(X, y)
        pipe = build_exchange_pipeline(model, power_transform, scale_binary)
        train_exchange_pipeline(pipe, X_train, y_train, X_test, y_test)
        export_exchange_model(pipe, args.output_joblib_root_dir + f"model_{exchange}.joblib")
        print(f"Saved model under {args.output_joblib_root_dir + f"model_{exchange}.joblib"}")

if __name__ == "__main__":
    main()
        