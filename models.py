import argparse
import os
from typing import Dict, Tuple, Any

import pandas as pd
from joblib import dump
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import make_column_transformer

# Constants

NUMERIC_FEATURES = ["order_qty", "limit_price", "bid_price", "ask_price", "bid_size", "ask_size"]
BINARY_FEATURE = ["side"]
TARGET_COLUMN = "price_improvement"
EXCHANGE_COLUMN = "exchange"

# Exchanges to train and their preprocessing/model configuration
EXCHANGES = ["1516", "29608"]

PREPROCESSING_CONFIG: Dict[str, Tuple[bool, bool]] = {
    # exchange: (power_transform_numeric, scale_binary)
    "1516": (False, False),
    "29608": (False, True),
}

MODEL_CONFIG: Dict[str, Any] = {
    "1516": ExtraTreesRegressor(
        n_jobs=-1,
        random_state=731,
        max_features="sqrt",
        min_samples_leaf=1,
        n_estimators=100,
    ),
    "29608": ExtraTreesRegressor(
        n_jobs=-1,
        random_state=731,
        max_features="sqrt",
        min_samples_leaf=1,
        n_estimators=100,
    ),
}

RANDOM_STATE_SPLIT = 731


# Data loading & preparation

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load training data from a CSV file.

    Parameters:
    file_path : str
        Path to the input CSV file.

    Returns:
    pd.DataFrame
        Loaded dataset.

    Raises:
    FileNotFoundError
        If the provided file path does not exist.
    ValueError
        If the file cannot be read as CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input CSV file not found: {file_path}")

    try:
        data = pd.read_csv(file_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unable to read CSV file at: {file_path}") from exc

    return data


def validate_required_columns(data: pd.DataFrame) -> None:
    """
    Validate that all required columns are present in the dataset.

    Parameters:
    data : pd.DataFrame
        Input dataset.

    Raises:
    ValueError
        If any required column is missing.
    """
    required_columns = set(NUMERIC_FEATURES + BINARY_FEATURE + [TARGET_COLUMN, EXCHANGE_COLUMN])
    missing = required_columns.difference(data.columns)

    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns in input data: {missing_str}")


def get_exchange_data(exchange: str, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Filter data for a specific exchange and split into features and target.

    Parameters:
    exchange : str
        Exchange code (e.g. "1516"). (Not including "ID")
    data : pd.DataFrame
        Full dataset containing multiple exchanges.

    Returns:
    X : pd.DataFrame
        Feature matrix for this exchange.
    y : pd.Series
        Target vector for this exchange.

    Raises:
    ValueError
        If no rows exist for the requested exchange.
    """
    # In the CSV, the exchange column is expected to match "ID{exchange}"
    exchange_id = f"ID{exchange}"
    exchange_data = data.loc[data[EXCHANGE_COLUMN] == exchange_id]

    if exchange_data.empty:
        raise ValueError(f"No data found for exchange: {exchange_id}")

    X = exchange_data[NUMERIC_FEATURES + BINARY_FEATURE]
    y = exchange_data[TARGET_COLUMN]

    return X, y


def split_exchange_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE_SPLIT,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split features and target into train and test sets.

    Parameters:
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.
    test_size : float, optional
        Proportion of data used for testing, by default 0.2.
    random_state : int, optional
        Random seed for reproducibility, by default RANDOM_STATE_SPLIT.

    Returns:
    X_train, X_test, y_train, y_test : tuple
        Splits of features and target.

    Raises:
    ValueError
        If there are not enough samples to perform the split.
    """
    if len(X) < 2:
        raise ValueError("Not enough samples to split train/test sets.")

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# ---- Pipeline construction & training ----

def build_exchange_pipeline(
    model: ExtraTreesRegressor,
    power_transform_numeric: bool = False,
    scale_binary: bool = True,
) -> Pipeline:
    """
    Build a preprocessing + model pipeline for a single exchange.

    Parameters:
    model : ExtraTreesRegressor
        Regressor to use after preprocessing.
    power_transform_numeric : bool, optional
        Whether to apply a PowerTransformer before scaling numeric features.
    scale_binary : bool, optional
        Whether to standardize the binary feature; if False, passes it through.

    Returns:
    sklearn.pipeline.Pipeline
        Complete pipeline ready for fitting.
    """
    # Numeric preprocessing: optional power transform + scaling
    if power_transform_numeric:
        numeric_preprocessor = make_pipeline(PowerTransformer(), StandardScaler())
    else:
        numeric_preprocessor = StandardScaler()

    # Binary preprocessing: either scale or passthrough
    binary_preprocessor = StandardScaler() if scale_binary else "passthrough"

    # Column-wise preprocessing
    preprocessor = make_column_transformer(
        (numeric_preprocessor, NUMERIC_FEATURES),
        (binary_preprocessor, BINARY_FEATURE),
        remainder="drop",  # ensure no unexpected columns leak through
    )

    # Full pipeline: preprocessing -> model
    pipeline_model = make_pipeline(preprocessor, model)
    return pipeline_model


def train_exchange_pipeline(
    pipeline_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fit the exchange pipeline on training data.

    Parameters:
    pipeline_model : Pipeline
        Preprocessing + model pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns:
    Pipeline
        Fitted pipeline.
    """
    pipeline_model.fit(X_train, y_train)
    return pipeline_model


def evaluate_exchange_pipeline(
    pipeline_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate a fitted pipeline on test data using regression metrics.

    Parameters:
    pipeline_model : Pipeline
        Fitted pipeline.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.

    Returns:
    dict
        Dictionary containing RMSE, R², and pipeline score.
    """
    y_pred = pipeline_model.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pipeline_score = pipeline_model.score(X_test, y_test)

    return {"rmse": rmse, "r2": r2, "score": pipeline_score}


def export_exchange_model(pipeline_model: Pipeline, output_path: str) -> None:
    """
    Serialize a fitted pipeline to disk using joblib.

    Parameters:
    pipeline_model : Pipeline
        Fitted preprocessing + model pipeline.
    output_path : str
        Full path (excluding filename) to store the joblib file.

    Raises:
    OSError
        If saving the model fails.
    """
    try:
        dump(pipeline_model, output_path)
    except Exception as exc:  # noqa: BLE001
        raise OSError(f"Failed to save model to: {output_path}") from exc


# Argument Parsing

def parse_args():
    """
    Parse and validate command line arguments for training.
    """
    parser = argparse.ArgumentParser(
        description="Train exchange-specific price improvement models and export them as joblib files.",
    )

    # Helper functions to validate arguments

    def input_csv_file(path: str) -> str:
        if not path.endswith(".csv"):
            raise argparse.ArgumentTypeError("Input file must end with .csv")
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"File not found: {path}")
        return path

    def output_directory(path: str) -> str:
        # Allow optional trailing slash; normalize to always end with "/"
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"Output directory does not exist: {path}")
        return path if path.endswith("/") else path + "/"

    parser.add_argument(
        "--input_csv_file",
        type=input_csv_file,
        required=True,
        help="Path to input CSV file containing training data.",
    )
    parser.add_argument(
        "--output_joblib_root_dir",
        type=output_directory,
        required=True,
        help="Directory to store the trained exchange models (joblib files).",
    )

    args = parser.parse_args()

    return parser, args


# ---- Main program ----

def main() -> None:
    """
    Main script: load data, train a model per exchange, evaluate, and export.
    """
    parser, args = parse_args()

    # Ensure all the arguments are present
    missing = []
    if not getattr(args, "input_csv_file", None):
        missing.append("--input_csv_file")
    if not getattr(args, "output_joblib_root_dir", None):
        missing.append("--output_joblib_root_dir")

    if len(missing) > 0:
        raise ValueError(
            f"Missing required command-line arguments: {missing}. "
            f"Run with --help to see usage information."
        )

    # Load and validate data
    data = load_data(args.input_csv_file)
    validate_required_columns(data)

    # Loop over each exchange configuration and train/export a model
    for exchange in EXCHANGES:
        print(f"\nTraining model for exchange {exchange}")

        power_transform_numeric, scale_binary = PREPROCESSING_CONFIG[exchange]
        model = MODEL_CONFIG[exchange]

        # Prepare data
        X, y = get_exchange_data(exchange, data)
        X_train, X_test, y_train, y_test = split_exchange_data(X, y)

        # Build, train, and evaluate pipeline
        pipeline_model = build_exchange_pipeline(
            model=model,
            power_transform_numeric=power_transform_numeric,
            scale_binary=scale_binary,
        )
        pipeline_model = train_exchange_pipeline(pipeline_model, X_train, y_train)
        metrics = evaluate_exchange_pipeline(pipeline_model, X_test, y_test)

        # Export trained model
        output_path = os.path.join(args.output_joblib_root_dir, f"model_{exchange}.joblib")
        export_exchange_model(pipeline_model, output_path)

        print(
            f"Saved model to: {output_path}\n"
            f"RMSE: {metrics['rmse']:.4f} | R²: {metrics['r2']:.4f} | Score: {metrics['score']:.4f}",
        )


if __name__ == "__main__":
    main()
