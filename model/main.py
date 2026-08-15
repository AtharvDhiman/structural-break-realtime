"""
Structural Break: Real-Time — submission entry point.

train(): fit AR+GARCH residual statistics per training series, assemble calibrated
features, and train a LightGBM stacker (deterministic, single-thread).
infer(): stream per-step break probabilities, one score per online observation,
using the incremental scorer in sbrt_core (O(1) per point after a per-series fit).
"""
import os
from typing import List, Tuple, Iterable, Optional

import numpy as np

import sbrt_core as core

MODEL_FILE = "model.txt"  # LightGBM native text format: portable across platforms

LGB_PARAMS = dict(
    objective="binary", n_estimators=200, learning_rate=0.03, num_leaves=31,
    min_child_samples=200, colsample_bytree=1.0, reg_lambda=1.0,
    random_state=0, deterministic=True, force_row_wise=True, num_threads=1,
    verbosity=-1,
)


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
):
    model_path = os.path.join(model_directory_path, MODEL_FILE)
    if os.path.exists(model_path):
        # A pre-trained model is shipped in resources/; skip the (slow) refit.
        # Delete resources/model.txt to force a full retrain from `datasets`.
        return

    import lightgbm as lgb

    Xs, ys = [], []
    for dataset_id, x_hist, x_online, tau in datasets:
        n = len(x_online)
        if n == 0:
            continue
        X = core.series_features(np.asarray(x_hist, dtype=np.float64),
                                 np.asarray(x_online, dtype=np.float64))
        if tau is None or tau < 0:
            y = np.zeros(n, dtype=np.int8)
        else:
            y = (np.arange(n) >= int(tau)).astype(np.int8)
        Xs.append(X)
        ys.append(y)

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    model = lgb.LGBMClassifier(**LGB_PARAMS).fit(X, y, feature_name=core.FEAT_ORDER)
    model.booster_.save_model(os.path.join(model_directory_path, MODEL_FILE))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    import lightgbm as lgb

    booster = lgb.Booster(model_file=os.path.join(model_directory_path, MODEL_FILE))

    yield  # signal readiness to the runner

    for x_historical, x_online in datasets:
        scorer = core.SeriesScorer(booster)
        scorer.prepare(np.asarray(x_historical, dtype=np.float64))
        for point in x_online:
            yield scorer.update(float(point))
