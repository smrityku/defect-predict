"""Model-name resolution for DefectInsight experiment scripts."""

from copy import deepcopy

from models.train_compare import MODELS


MODEL_ALIASES = {
    "logistic": "Logistic Regression",
    "logistic_regression": "Logistic Regression",
    "lr": "Logistic Regression",
    "random_forest": "Random Forest",
    "rf": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "gb": "Gradient Boosting",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "hgb": "Hist Gradient Boosting",
    "extra_trees": "Extra Trees",
    "et": "Extra Trees",
    "adaboost": "AdaBoost",
    "ada_boost": "AdaBoost",
    "svm": "SVM",
    "svc": "SVM",
    "naive_bayes": "Naive Bayes",
    "nb": "Naive Bayes",
    "mlp": "MLP",
}


def slugify_model_name(name):
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def canonical_model_name(name):
    slug = slugify_model_name(name)
    if slug in MODEL_ALIASES:
        return MODEL_ALIASES[slug]
    for existing in MODELS:
        if slugify_model_name(existing) == slug:
            return existing
    if slug == "xgboost":
        return "XGBoost"
    return None


def _xgboost_spec():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is not installed. Install the optional dependency or remove xgboost from --models."
        ) from exc

    return {
        "est": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1,
        ),
        "grid": {},
    }


def resolve_model_specs(model_names=None, skip_unavailable=True):
    """Return `{display_name: spec}` for requested model names."""
    if not model_names or model_names == ["all"]:
        requested = list(MODELS.keys())
    else:
        requested = model_names

    specs = {}
    skipped = []
    for raw_name in requested:
        canonical = canonical_model_name(raw_name)
        if canonical is None:
            raise ValueError(f"Unknown model '{raw_name}'. Known models: {', '.join(available_model_slugs())}")

        if canonical == "XGBoost":
            try:
                specs[canonical] = _xgboost_spec()
            except ImportError as exc:
                if skip_unavailable:
                    skipped.append({"model": raw_name, "reason": str(exc)})
                    continue
                raise
        else:
            specs[canonical] = deepcopy(MODELS[canonical])

    return specs, skipped


def available_model_slugs(include_optional=True):
    slugs = sorted({slugify_model_name(name) for name in MODELS} | set(MODEL_ALIASES))
    if include_optional:
        slugs.append("xgboost")
    return slugs


def default_fast_model_names():
    return ["logistic_regression", "random_forest", "svm"]
