"""End-to-end training, experiment tracking, model registry, monitoring, and holdout prediction."""

import hashlib
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from .data import load_raw_data, clean_data, check_all_issues
from .features import ALL_FEATURES, NUMERIC_FEATURES, TARGET
from .model import train_model, evaluate_model, save_model
from .drift import calculate_psi, evaluate_mae_by_group


def run_pipeline(config_path: str = "params.yaml"):
    """Execute full MLOps pipeline for MP053."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Fingerprint raw train data
    train_path = Path(config["data"]["raw_train_path"])
    raw_bytes = train_path.read_bytes()
    data_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    print(f"Data SHA-256: {data_sha256}")

    # 2. Load and inspect issues
    df_train_raw = load_raw_data(train_path)
    df_holdout_raw = load_raw_data(config["data"]["holdout_path"])
    issues = check_all_issues(df_train_raw, df_holdout_raw)
    print("Detected Data Issues:", issues)

    # 3. Clean training data
    df_train_clean = clean_data(df_train_raw, is_train=True)
    print(f"Cleaned train shape: {df_train_clean.shape}")

    # 4. Setup MLflow
    tracking_uri = config["training"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    exp_name = config["training"]["experiment_name"]
    mlflow.set_experiment(exp_name)
    client = MlflowClient()

    # Train / validation split for model exploration
    X = df_train_clean[ALL_FEATURES]
    y = df_train_clean[TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["seed"]
    )

    # 5. Track 3 candidate runs
    candidate_configs = [
        ("linear-baseline", "linear", {}),
        ("rf-n100-d8", "rf", {"n_estimators": 100, "max_depth": 8}),
        ("gb-n100-lr01", "gb", {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3}),
    ]

    run_maes = {}
    for run_name, model_type, kwargs in candidate_configs:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", model_type)
            for k, v in kwargs.items():
                mlflow.log_param(k, v)

            model = train_model(X_tr, y_tr, model_type=model_type, **kwargs)
            val_mae = evaluate_model(model, X_val, y_val)
            mlflow.log_metric("val_mae", val_mae)
            mlflow.log_metric("mae", val_mae)
            run_maes[run_name] = (val_mae, model_type, kwargs, run.info.run_id)
            print(f"Candidate {run_name} MAE: {val_mae:.4f}")

    # Pick winner
    best_candidate = min(run_maes, key=lambda k: run_maes[k][0])
    best_val_mae, best_type, best_kwargs, _ = run_maes[best_candidate]
    print(f"Best candidate: {best_candidate} (MAE: {best_val_mae:.4f})")

    # 6. Fit v1 on full cleaned training set and register
    model_name = config["training"]["registered_model_name"]
    with mlflow.start_run(run_name="champion-v1") as v1_run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", best_type)
        mlflow.log_param("stage", "v1")
        for k, v in best_kwargs.items():
            mlflow.log_param(k, v)

        v1_model = train_model(X, y, model_type=best_type, **best_kwargs)
        train_mae = evaluate_model(v1_model, X, y)
        mlflow.log_metric("train_mae", train_mae)
        mlflow.log_metric("val_mae", best_val_mae)
        mlflow.log_metric("mae", train_mae)

        mlflow.sklearn.log_model(
            sk_model=v1_model,
            artifact_path="model",
            registered_model_name=model_name,
            input_example=X.head(2),
            serialization_format="cloudpickle",
        )
        v1_run_id = v1_run.info.run_id

    # Set alias champion to version 1 initially
    client.set_registered_model_alias(model_name, "champion", "1")
    save_model(v1_model, config["training"]["model_path"])
    print(f"Registered v1 run_id: {v1_run_id} as @champion version 1")

    # 7. Monitoring on later.csv
    later_path = Path(config["data"]["raw_later_path"])
    df_later_raw = load_raw_data(later_path)
    # Sort by date as instructed in PROBLEM.md section 7
    df_later = clean_data(df_later_raw, is_train=False)
    df_later = df_later.sort_values(by=["date", "record_id"]).reset_index(drop=True)

    # Compute PSI for all 6 numeric columns
    psi_dict = {}
    for col in NUMERIC_FEATURES:
        psi_val = calculate_psi(df_train_clean[col], df_later[col], bins=config["monitoring"]["psi_bins"])
        psi_dict[col] = psi_val
    most_drifted = max(psi_dict, key=psi_dict.get)
    print("PSI Values:", psi_dict)
    print(f"Most drifted feature: {most_drifted}")

    # Compute v1's MAE per hostel group on later.csv
    mae_by_group = evaluate_mae_by_group(v1_model, df_later, ALL_FEATURES, group_col="hostel", target_col=TARGET)
    worst_group = max(mae_by_group, key=mae_by_group.get)
    print("MAE by group on later:", mae_by_group)
    print(f"Worst served group: {worst_group}")

    # 8. Retrain v2 on clean train + first 70% of later.csv
    n_split = int(config["monitoring"]["retrain_split"] * len(df_later))
    later_70 = df_later.iloc[:n_split].copy()
    later_30 = df_later.iloc[n_split:].copy()

    df_train_v2 = pd.concat([df_train_clean[ALL_FEATURES + [TARGET]], later_70[ALL_FEATURES + [TARGET]]], ignore_index=True)
    v2_model = train_model(df_train_v2[ALL_FEATURES], df_train_v2[TARGET], model_type=best_type, **best_kwargs)

    # Evaluate on check set (last 30% of later.csv)
    mae_v1_on_check = evaluate_model(v1_model, later_30[ALL_FEATURES], later_30[TARGET])
    mae_v2_on_check = evaluate_model(v2_model, later_30[ALL_FEATURES], later_30[TARGET])
    rel_improvement = (mae_v1_on_check - mae_v2_on_check) / mae_v1_on_check
    print(f"Check Set: v1 MAE = {mae_v1_on_check:.4f}, v2 MAE = {mae_v2_on_check:.4f}, Improvement = {rel_improvement*100:.2f}%")

    decision = "promote" if rel_improvement >= config["monitoring"]["promotion_threshold"] else "keep"
    champion_version = 1

    with mlflow.start_run(run_name="retrain-v2") as v2_run:
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("model_type", best_type)
        mlflow.log_param("stage", "v2")
        mlflow.log_metric("mae_v1_on_check", mae_v1_on_check)
        mlflow.log_metric("mae_v2_on_check", mae_v2_on_check)
        mlflow.log_metric("relative_improvement", rel_improvement)
        mlflow.log_metric("mae", mae_v2_on_check)

        if decision == "promote":
            mlflow.sklearn.log_model(
                sk_model=v2_model,
                artifact_path="model",
                registered_model_name=model_name,
                input_example=X.head(2),
                serialization_format="cloudpickle",
            )
            client.set_registered_model_alias(model_name, "champion", "2")
            save_model(v2_model, config["training"]["model_path"])
            champion_version = 2
            champion_model = v2_model
            print("Decision: PROMOTE! Set @champion alias to version 2.")
        else:
            champion_model = v1_model
            print("Decision: KEEP v1.")

    # 9. Final Holdout Predictions using @champion model
    df_holdout = clean_data(df_holdout_raw, is_train=False)
    holdout_preds = champion_model.predict(df_holdout[ALL_FEATURES])
    
    pred_dir = Path("predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = Path(config["monitoring"]["predictions_path"])
    
    df_out = pd.DataFrame({
        "record_id": df_holdout["record_id"],
        "prediction": np.round(holdout_preds, 2),
    })
    df_out.to_csv(pred_file, index=False)
    print(f"Generated {len(df_out)} predictions to {pred_file}")

    # 10. Update findings.yaml
    findings_data = {
        "group_id": "G212",
        "problem_id": "MP053",
        "data_issues": {
            "leaking_column": {
                "present": issues["leaking_column"]["present"],
                "column": issues["leaking_column"]["column"],
            },
            "unit_mixup": {
                "present": issues["unit_mixup"]["present"],
                "column": issues["unit_mixup"]["column"],
                "group_value": issues["unit_mixup"]["group_value"],
                "factor": issues["unit_mixup"]["factor"],
            },
            "sentinel_value": {
                "present": issues["sentinel_value"]["present"],
                "column": issues["sentinel_value"]["column"],
                "value": issues["sentinel_value"]["value"],
                "count": issues["sentinel_value"]["count"],
            },
            "duplicate_rows": {
                "present": issues["duplicate_rows"]["present"],
                "count": issues["duplicate_rows"]["count"],
            },
            "stuck_sensor": {
                "present": issues["stuck_sensor"]["present"],
                "column": issues["stuck_sensor"]["column"],
                "start_date": str(issues["stuck_sensor"]["start_date"]),
                "end_date": str(issues["stuck_sensor"]["end_date"]),
            },
            "target_typos": {
                "present": issues["target_typos"]["present"],
                "count": issues["target_typos"]["count"],
            },
            "spelling_variants": {
                "present": issues["spelling_variants"]["present"],
                "column": issues["spelling_variants"]["column"],
                "true_levels": issues["spelling_variants"]["true_levels"],
            },
        },
        "tracking": {
            "experiment_name": exp_name,
            "best_run_id": v1_run_id,
            "registered_model_name": model_name,
            "champion_version": champion_version,
        },
        "monitoring": {
            "psi": {col: float(val) for col, val in psi_dict.items()},
            "most_drifted_feature": most_drifted,
            "mae_by_group": {
                "Boys Hostel 1": mae_by_group.get("Boys Hostel 1"),
                "Boys Hostel 2": mae_by_group.get("Boys Hostel 2"),
                "Girls Hostel": mae_by_group.get("Girls Hostel"),
            },
            "worst_served_group": worst_group,
            "mae_v1_on_check": float(round(mae_v1_on_check, 4)),
            "mae_v2_on_check": float(round(mae_v2_on_check, 4)),
            "decision": decision,
        },
    }

    # Write findings to reports/findings.yaml and findings.yaml
    for out_path in ["findings.yaml", "reports/findings.yaml"]:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            yaml.dump(findings_data, f, sort_keys=False)
        print(f"Updated {out_path}")

    print("\nPipeline execution complete successfully!")


if __name__ == "__main__":
    run_pipeline()
