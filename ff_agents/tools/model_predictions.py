"""
Tool functions that wrap predictive models for use by ADK agents.
Each function takes text (article or statement) and returns model scores as dicts.
Models are loaded and cached on first use.
"""

import json
import os
import sys
from pathlib import Path

_TOOL_DIR = Path(__file__).resolve().parent
_FF_AGENTS_DIR = _TOOL_DIR.parent
_PROJECT_ROOT = _FF_AGENTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd

_models_loaded = False
_freq_model = None
_sens_model = None
_malicious_model = None
_naive_model = None


def _ensure_models_loaded() -> None:
    global _models_loaded, _freq_model, _sens_model, _malicious_model, _naive_model
    if _models_loaded:
        return
    from src.predictive_models import (
        load_datasets,
        build_frequency_model,
        build_sensationalism_model,
        build_malicious_account_model,
        build_naive_realism_model,
    )
    data_dir = _PROJECT_ROOT / "data"
    config_path = _PROJECT_ROOT / "src" / "config.json"
    train_path = data_dir / "train_set.csv"
    val_path = data_dir / "val_set.csv"
    test_path = data_dir / "test_set.csv"
    with open(config_path, "r") as f:
        cfg = json.load(f)
    train_df, _, _ = load_datasets(str(train_path), str(val_path), str(test_path))
    _freq_model = build_frequency_model(train_df, **cfg["models"]["frequency"])
    _sens_model = build_sensationalism_model(train_df, **cfg["models"]["sensationalism"])
    _malicious_model = build_malicious_account_model(train_df, **cfg["models"]["malicious_account"])
    _naive_model = build_naive_realism_model(train_df, **cfg["models"]["naive_realism"])
    _models_loaded = True


def get_frequency_heuristic_prediction(text: str) -> dict:
    """
    Get the frequency-heuristic model prediction for the given text.
    Use this to inform your analysis of repetition, origin tracing, and evidence verification.

    Args:
        text: The article or statement text to score.

    Returns:
        dict with status, score (0-2), and confidence (0-1).
        score: 0 = none/minimal, 1 = moderate, 2 = heavy repetition or appeal to consensus.
    """
    _ensure_models_loaded()
    from src.predictive_models import predict_frequency_model
    df = pd.DataFrame({"statement": [text.strip()]})
    m, tfidf, cv, token_dict, buzzwords, le = _freq_model
    out = predict_frequency_model(df, m, tfidf, cv, token_dict, buzzwords, le)
    row = out.iloc[0]
    return {
        "status": "success",
        "score": int(row["predicted_frequency_heuristic"]),
        "confidence": float(row["frequency_heuristic_score"]),
    }


def get_sensationalism_prediction(text: str) -> dict:
    """
    Get the sensationalism model prediction for the given text.
    Use this to inform your analysis of language intensity, shock vs. substance, and tone.

    Args:
        text: The article or statement text to score.

    Returns:
        dict with status, score (0-2), and confidence (0-1).
        score: 0 = neutral/objective, 1 = mildly emotional, 2 = highly sensationalized.
    """
    _ensure_models_loaded()
    from src.predictive_models import predict_sensationalism_model
    df = pd.DataFrame({"statement": [text.strip()]})
    pipeline, num_feats = _sens_model
    out = predict_sensationalism_model(df, pipeline, num_feats)
    row = out.iloc[0]
    return {
        "status": "success",
        "score": int(row["predicted_sensationalism"]),
        "confidence": float(row["sensationalism_score"]),
    }


def get_malicious_account_prediction(text: str) -> dict:
    """
    Get the malicious-account model prediction for the given text.
    Use this to inform your analysis of source credibility and inauthentic-behavior indicators.

    Args:
        text: The article or statement text to score.

    Returns:
        dict with status, score (0-2), and confidence (0-1).
        score: 0 = credible, 1 = slightly suspicious, 2 = clearly deceptive or malicious.
    """
    _ensure_models_loaded()
    from src.predictive_models import predict_malicious_account_model
    df = pd.DataFrame({"statement": [text.strip()]})
    model, tfidf, le = _malicious_model
    out = predict_malicious_account_model(df, model, tfidf, le)
    row = out.iloc[0]
    return {
        "status": "success",
        "score": int(row["predicted_malicious_account"]),
        "confidence": float(row["malicious_account_score"]),
    }


def get_naive_realism_prediction(text: str) -> dict:
    """
    Get the naive-realism model prediction for the given text.
    Use this to inform your analysis of one-sidedness, dismissal of alternatives, and dogmatism.

    Args:
        text: The article or statement text to score.

    Returns:
        dict with status, score (0-2), and confidence (0-1).
        score: 0 = balanced, 1 = somewhat one-sided, 2 = fully dogmatic.
    """
    _ensure_models_loaded()
    from src.predictive_models import predict_naive_realism_model
    df = pd.DataFrame({"statement": [text.strip()]})
    pipeline, num_feats = _naive_model
    out = predict_naive_realism_model(df, pipeline, num_feats)
    row = out.iloc[0]
    return {
        "status": "success",
        "score": int(row["predicted_naive_realism"]),
        "confidence": float(row["naive_realism_score"]),
    }
