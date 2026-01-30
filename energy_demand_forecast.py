"""
End-to-end: sample tables -> feature engineering pipeline -> time-based validation
-> regression model -> peak-focused evaluation -> forecast table output.

Works as-is (pandas + scikit-learn).
"""

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def make_sample_data(seed: int = 7) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    ts = pd.date_range("2025-07-01 00:00:00", "2025-07-10 23:00:00", freq="h")
    zones = ["NORTH", "SOUTH"]

    # demand(ts, zone_id, demand_mw)
    demand_rows = []
    for zone in zones:
        base = 1200 if zone == "NORTH" else 1000
        for t in ts:
            h = t.hour
            # daily + afternoon peak + noise
            daily = 140 * np.sin((h - 6) / 24 * 2 * np.pi)
            peak = 260 * np.exp(-((h - 17) ** 2) / (2 * 4 ** 2))
            noise = rng.normal(0, 22)
            y = max(100, base + daily + peak + noise)
            demand_rows.append((t, zone, round(float(y), 2)))
    demand = pd.DataFrame(demand_rows, columns=["ts", "zone_id", "demand_mw"])

    # weather(ts, zone_id, temp_f, humidity_pct, wind_mph)
    weather_rows = []
    for zone in zones:
        temp_base = 82 if zone == "NORTH" else 86
        for t in ts:
            h = t.hour
            temp = temp_base + 9 * np.sin((h - 7) / 24 * 2 * np.pi) + rng.normal(0, 0.7)
            hum = 55 + 11 * np.cos(h / 24 * 2 * np.pi) + rng.normal(0, 1.0)
            wind = 6 + 2.0 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 0.4)
            weather_rows.append((t, zone, round(float(temp), 2), round(float(hum), 2), round(float(wind), 2)))
    weather = pd.DataFrame(weather_rows, columns=["ts", "zone_id", "temp_f", "humidity_pct", "wind_mph"])

    # calendar(date, day_of_week, is_weekend, holiday_name, is_holiday)
    dates = pd.date_range(ts.min().normalize(), ts.max().normalize(), freq="D")
    calendar = pd.DataFrame({"date": dates})
    calendar["day_of_week"] = calendar["date"].dt.day_name()
    calendar["is_weekend"] = (calendar["date"].dt.weekday >= 5).astype(int)

    # example: mark one holiday
    calendar["holiday_name"] = None
    calendar.loc[calendar["date"] == pd.Timestamp("2025-07-04"), "holiday_name"] = "Independence Day"
    calendar["is_holiday"] = calendar["holiday_name"].notna().astype(int)

    # events(date, event_name, event_impact_flag)
    events = pd.DataFrame({
        "date": [pd.Timestamp("2025-07-06")],
        "event_name": ["Major Sports Event"],
        "event_impact_flag": [1],
    })

    return {"demand": demand, "weather": weather, "calendar": calendar, "events": events}


# -----------------------------
# 2) FEATURE ENGINEERING
# -----------------------------
@dataclass
class FeatureConfig:
    # lag hours for target
    lags: Tuple[int, ...] = (1, 24, 48, 168)
    # rolling windows in hours
    roll_windows: Tuple[int, ...] = (24, 72, 168)
    # peak quantile for peak-aware evaluation/features
    peak_quantile: float = 0.90


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["ts"].dt.hour
    df["dow"] = df["ts"].dt.dayofweek  # 0=Mon
    df["month"] = df["ts"].dt.month
    df["is_weekend_ts"] = (df["dow"] >= 5).astype(int)
    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cdh"] = (df["temp_f"] - 65).clip(lower=0)
    df["hdh"] = (65 - df["temp_f"]).clip(lower=0)
    df["temp_x_hour"] = df["temp_f"] * df["hour"]
    df["cdh_x_hour"] = df["cdh"] * df["hour"]
    return df


def build_feature_table(
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    calendar: pd.DataFrame,
    events: Optional[pd.DataFrame] = None,
    cfg: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    df = demand.merge(weather, on=["ts", "zone_id"], how="left")
    df["date"] = df["ts"].dt.floor("D")
    df = df.merge(calendar, on="date", how="left")

    if events is not None and not events.empty:
        df = df.merge(events[["date", "event_impact_flag"]], on="date", how="left")
    df["event_impact_flag"] = df["event_impact_flag"].fillna(0).astype(int)
    df = df.sort_values(["zone_id", "ts"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_weather_features(df)
    w = Window.partitionBy("zone_id").orderBy("ts")
    df = (
        df
        .withColumn("lag_1",   F.lag("demand_mw", 1).over(w))
        .withColumn("lag_24",  F.lag("demand_mw", 24).over(w))
        .withColumn("lag_48",  F.lag("demand_mw", 48).over(w))
        .withColumn("lag_168", F.lag("demand_mw", 168).over(w))
    )
    y_shifted = df.groupby("zone_id")["demand_mw"].shift(1)
    for w in cfg.roll_windows:
        df[f"roll_{w}_mean"] = y_shifted.groupby(df["zone_id"]).rolling(w, min_periods=max(3, w // 4)).mean().reset_index(level=0, drop=True)
        df[f"roll_{w}_max"] = y_shifted.groupby(df["zone_id"]).rolling(w, min_periods=max(3, w // 4)).max().reset_index(level=0, drop=True)
        df[f"roll_{w}_std"] = y_shifted.groupby(df["zone_id"]).rolling(w, min_periods=max(3, w // 4)).std().reset_index(level=0, drop=True)
    peak_threshold = df["demand_mw"].quantile(cfg.peak_quantile)
    df["is_peak_hist"] = (df["demand_mw"] >= peak_threshold).astype(int)
    required = ["lag_24", "roll_24_mean"]
    df = df.dropna(subset=required).reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, train_frac: float = 0.75, valid_frac: float = 0.15):
    df = df.sort_values("ts").reset_index(drop=True)
    n = len(df)
    n_train = int(n * train_frac)
    n_valid = int(n * (train_frac + valid_frac))
    train = df.iloc[:n_train].copy()
    valid = df.iloc[n_train:n_valid].copy()
    test = df.iloc[n_valid:].copy()
    return train, valid, test


def make_model(numeric_features: List[str], categorical_features: List[str]) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore")),])
    pre = ColumnTransformer(transformers=[("num", num_pipe, numeric_features),("cat", cat_pipe, categorical_features),],remainder="drop",)
    model = HistGradientBoostingRegressor(learning_rate=0.05,max_depth=6,max_iter=400,random_state=7,)
    return Pipeline(steps=[("prep", pre), ("model", model)])

def evaluate_peak_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_mask: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    if peak_mask.sum() == 0:
        return {"rmse": rmse, "mae": mae, "peak_rmse": np.nan, "peak_mae": np.nan}
    peak_rmse = float(np.sqrt(mean_squared_error(y_true[peak_mask], y_pred[peak_mask])))
    peak_mae = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]))
    return {"rmse": rmse, "mae": mae, "peak_rmse": peak_rmse, "peak_mae": peak_mae}


# -----------------------------
# 4) TRAIN + EVAL + FORECAST OUTPUT
# -----------------------------
def run_end_to_end():
    tables = make_sample_data()
    print(tables)

    cfg = FeatureConfig(
        lags=(1, 24, 48, 168),
        roll_windows=(24, 72, 168),
        peak_quantile=0.90,
    )

    feat = build_feature_table( demand=tables["demand"],weather=tables["weather"],calendar=tables["calendar"],events=tables["events"],cfg=cfg,)
    train, valid, test = time_split(feat, train_frac=0.75, valid_frac=0.15)
    peak_thr = train["demand_mw"].quantile(cfg.peak_quantile)
    target = "demand_mw"
    drop_cols = ["demand_mw", "ts", "date", "holiday_name"] 
    categorical = ["zone_id", "day_of_week"]
    numeric = [c for c in train.columns if c not in drop_cols + categorical]

    X_train, y_train = train[numeric + categorical], train[target].values
    X_valid, y_valid = valid[numeric + categorical], valid[target].values
    X_test, y_test = test[numeric + categorical], test[target].values

    pipe = make_model(numeric_features=numeric, categorical_features=categorical)
    pipe.fit(X_train, y_train)

    # Evaluate
    pred_valid = pipe.predict(X_valid)
    valid_peak_mask = (y_valid >= peak_thr)
    valid_metrics = evaluate_peak_metrics(y_valid, pred_valid, valid_peak_mask)

    pred_test = pipe.predict(X_test)
    test_peak_mask = (y_test >= peak_thr)
    test_metrics = evaluate_peak_metrics(y_test, pred_test, test_peak_mask)
    forecast = test[["ts", "zone_id"]].copy()
    forecast["yhat"] = pred_test
    forecast["model_version"] = "hgb_v1"
    forecast["run_id"] = "run_2025-07-11_0001"
    resid_std = float(np.std(y_valid - pred_valid))
    forecast["yhat_lower"] = forecast["yhat"] - 1.96 * resid_std
    forecast["yhat_upper"] = forecast["yhat"] + 1.96 * resid_std

    return tables, feat, (train, valid, test), pipe, forecast


if __name__ == "__main__":
    run_end_to_end()
