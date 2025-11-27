from joblib import load
from typing import Any, Tuple
import pandas as pd

EXCHANGES = ["1516", "29608"]
ROOT_PATH = "models/"

def best_price_improvement(
        symbol: str,
        side: str, # must be "B" or "S"
        quantity: int,
        limit_price: float,
        bid_price: float,
        ask_price: float,
        bid_size: int,
        ask_size: int,
    ) -> Tuple[str, float]:
    
    X_new = pd.DataFrame([{
        "side": int(side == "B"),
        "order_qty": quantity,
        "limit_price": limit_price,
        "bid_price": bid_price,
        "ask_price": ask_price,
        "bid_size": bid_size,
        "ask_size": ask_size,
    }])

    def load_models(root_path:str, exchanges:list[str]) -> dict[str, Any]:
        models = {}

        for exchange in exchanges:
            models[f"ID{exchange}"] = load(root_path + f"model_{exchange}.joblib")

        return models
    
    EXCHANGES = ["1516", "29608"]
    ROOT_PATH = "models/"

    models = load_models(ROOT_PATH, EXCHANGES)
    predictions = {}

    for name, model in models.items():
        predictions[name] = model.predict(X_new)

    predictions_df = pd.DataFrame(predictions, columns=["price_improvement"]).T

    best_value = predictions_df.price_improvement.max()
    best_exchange = predictions_df.price_improvement.idxmax()

    return best_exchange, best_value
