from evaluators.joblib_runner import JoblibRunner
from evaluators.pytorch_evaluator import PyTorchEvaluator
from evaluators.tensorflow_evaluator import TensorFlowEvaluator
from method_library import METHODS
from config import CONFIG
import os, pandas as pd, numpy as np

class InstitutionalConfluenceLogger:
    def __init__(self):
        self.joblib_eval = JoblibRunner(model_dir="model/joblib")
        self.torch_eval = PyTorchEvaluator(model_dir="model/pytorch")
        self.tf_eval = TensorFlowEvaluator(model_dir="model/tensorflow")
        self.methods = METHODS
        self.config = CONFIG
        self.logs = []

    def evaluate_methods(self, df):
        results = {}
        for name, fn in self.methods.items():
            # Determine model path
            joblib_path = f"model/joblib/{name}.joblib"
            torch_path = f"model/pytorch/{name}.pt"
            tf_path = f"model/tensorflow/{name}.keras"

            if os.path.exists(joblib_path):
                metrics, trades = self.joblib_eval.run(df, {name: fn}, self.config)
            elif os.path.exists(torch_path):
                metrics, trades = self.torch_eval.run(df, {name: fn}, self.config)
            elif os.path.exists(tf_path):
                metrics, trades = self.tf_eval.run(df, {name: fn}, self.config)
            else:
                continue  # Skip if no model found

            results[name] = metrics
        return results

    def compute_confluence(self, method_results):
        """Aggregate method results into a unified confidence score."""
        if not method_results:
            return {"bias": None, "confidence": 0.0}

        # Simple confluence logic: bias = BUY if majority methods bullish
        buy_count = sum(1 for r in method_results.values() if r.get("win_rate", 0) > 0.5)
        sell_count = sum(1 for r in method_results.values() if r.get("win_rate", 0) <= 0.5)
        total = buy_count + sell_count

        if total == 0:
            return {"bias": None, "confidence": 0.0}

        bias = "BUY" if buy_count > sell_count else "SELL"
        confidence = abs(buy_count - sell_count) / total
        return {"bias": bias, "confidence": confidence}

    def log_cycle(self, df):
        method_results = self.evaluate_methods(df)
        confluence = self.compute_confluence(method_results)
        self.logs.append({
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "confluence": confluence,
            "method_results": method_results
        })
        return confluence
