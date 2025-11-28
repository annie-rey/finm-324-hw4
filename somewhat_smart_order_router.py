from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from joblib import load

DEFAULT_EXCHANGES = ["1516", "29608"]
DEFAULT_MODELS_DIR = Path("models")


def load_models(models_dir: Path, exchanges: list[str]) -> Dict[str, Any]:
    """Load one model per exchange from the given directory.

    Models are expected to be named: model_{exchange}.joblib
    exchanges in list should not include "ID" at the front.
    and are keyed as 'ID{exchange}' in the returned dict.

    Raises:
    ValueError
        If no exchanges are provided.
    FileNotFoundError
        If any expected model file is missing.
    """
    exchanges = list(exchanges)
    if not exchanges:
        raise ValueError("At least one exchange must be provided.")

    models: Dict[str, Any] = {}

    for exchange in exchanges:
        model_path = models_dir / f"model_{exchange}.joblib"
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        models[f"ID{exchange}"] = load(model_path)

    return models


def best_price_improvement(
    symbol: str,
    side: str,  # "B" or "S"
    quantity: int,
    limit_price: float,
    bid_price: float,
    ask_price: float,
    bid_size: int,
    ask_size: int,
    exchanges: list[str] = DEFAULT_EXCHANGES,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Tuple[str, float]:
    """Return the exchange with the highest predicted price improvement.

    Raises
    ------
    ValueError
        If inputs are invalid.
    FileNotFoundError
        If model files are missing.
    RuntimeError
        If no models are loaded.
    """
    # Validate the inputs of the function

    # side validation
    if side not in {"B", "S"}:
        raise ValueError("side must be 'B' (buy) or 'S' (sell).")

    # positive integer checks
    int_fields = {
        "quantity": quantity,
        "bid_size": bid_size,
        "ask_size": ask_size,
    }
    for name, value in int_fields.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    # non-negative numeric checks
    price_fields = {
        "limit_price": limit_price,
        "bid_price": bid_price,
        "ask_price": ask_price,
    }
    for name, value in price_fields.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative.")

    # logical price check
    if bid_price > ask_price:
        raise ValueError("bid_price cannot be greater than ask_price.")

    # Construct the features dataframe from the function inputs
    features = pd.DataFrame(
        [
            {
                "side": 1 if side == "B" else 0, #Construct as a binary variable
                "order_qty": quantity,
                "limit_price": limit_price,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "bid_size": bid_size,
                "ask_size": ask_size,
            }
        ]
    )

    # Load the models using the helper function
    models = load_models(models_dir=models_dir, exchanges=exchanges)
    if not models:
        raise RuntimeError("No models were loaded; cannot compute price improvement.")

    # Get the predictions
    predictions = {
        exchange_name: float(model.predict(features)[0])
        for exchange_name, model in models.items()
    }

    # Find which one is best
    prediction_series = pd.Series(predictions, name="price_improvement")
    best_exchange = prediction_series.idxmax()
    best_value = prediction_series.loc[best_exchange]

    return best_exchange, best_value