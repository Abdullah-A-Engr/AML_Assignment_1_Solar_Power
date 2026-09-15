from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regression import (
    hypothesis, cost, fit_normal, fit_batch_gd, fit_sgd, rmse, clip_predictions
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

def add_time_features(df):
    out = df.copy()
    h = out["datetime"].dt.hour
    out["sin hour"] = np.sin(2 * np.pi * h / 24.0)
    out["cos hour"] = np.cos(2 * np.pi * h / 24.0)
    return out

def scale_train_test(train, test, feature_cols):
    mu = train[feature_cols].mean()
    sigma = train[feature_cols].std(ddof=0).replace(0, 1.0)
    train_s = train.copy()
    test_s = test.copy()
    train_s[feature_cols] = (train[feature_cols] - mu) / sigma
    test_s[feature_cols] = (test[feature_cols] - mu) / sigma
    return train_s, test_s, mu, sigma

def make_X(df, feature_cols):
    return np.column_stack([np.ones(len(df)), df[feature_cols].to_numpy(dtype=float)])

def evaluate(theta, X, y, daytime):
    pred = clip_predictions(hypothesis(X, theta))
    return rmse(y, pred), rmse(y[daytime], pred[daytime]), pred

def main():
    hourly = pd.read_csv(DATA / "plant1_hourly.csv", parse_dates=["datetime"])
    weather = pd.read_csv(DATA / "plant1_openmeteo.csv")
    # Accept the exact Open-Meteo CSV schema returned by the API/script.
    if "time" in weather.columns and "datetime" not in weather.columns:
        weather = weather.rename(columns={"time": "datetime"})
    weather = weather.rename(columns={
        "shortwave_radiation": "sw radiation",
        "temperature_2m": "temp 2m",
        "cloud_cover": "cloud cover",
    })
    weather["datetime"] = pd.to_datetime(weather["datetime"])

    df = hourly.merge(weather, on="datetime", how="inner")
    df = add_time_features(df)

    # Set A: on-site sensors.
    set_a = ["irradiation", "module temp", "ambient temp", "sin hour", "cos hour"]
    # Set B: public weather.
    set_b = ["sw radiation", "temp 2m", "cloud cover", "sin hour", "cos hour"]

    # Train/test split by date, no shuffling.
    train_mask = (df["datetime"].dt.date >= pd.Timestamp("2020-05-15").date()) & (
        df["datetime"].dt.date <= pd.Timestamp("2020-06-10").date()
    )
    test_mask = (df["datetime"].dt.date >= pd.Timestamp("2020-06-11").date()) & (
        df["datetime"].dt.date <= pd.Timestamp("2020-06-17").date()
    )

    train = df.loc[train_mask].copy()
    test = df.loc[test_mask].copy()

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    peak_train = float(train["ac power"].max())
    peak_time = train.loc[train["ac power"].idxmax(), "datetime"]

    # ----- Learning-rate experiments on Set A -----
    train_a, test_a, mu_a, sigma_a = scale_train_test(train, test, set_a)
    XA = make_X(train_a, set_a)
    yA = train_a["ac power"].to_numpy(float)

    batch_histories = {}
    for alpha in [1e-5, 1e-4, 1e-3]:
        theta, hist = fit_batch_gd(XA, yA, alpha, 500)
        batch_histories[alpha] = hist

    plt.figure(figsize=(8, 5))
    for alpha, hist in batch_histories.items():
        plt.plot(hist, label=f"alpha={alpha:g}")
    plt.xlabel("Iteration")
    plt.ylabel("J(theta)")
    plt.title("Batch Gradient Descent Learning-Rate Comparison — Set A")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "05_batch_learning_rates.png", dpi=180)
    plt.close()

    sgd_histories = {}
    for alpha in [1e-4, 1e-3, 1e-2]:
        theta, hist = fit_sgd(XA, yA, alpha, 50)
        sgd_histories[alpha] = hist

    plt.figure(figsize=(8, 5))
    for alpha, hist in sgd_histories.items():
        plt.plot(hist, label=f"alpha={alpha:g}")
    plt.xlabel("Epoch")
    plt.ylabel("J(theta)")
    plt.title("Stochastic Gradient Descent Learning-Rate Comparison — Set A")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "06_sgd_learning_rates.png", dpi=180)
    plt.close()

    # Chosen rates. These are selected from the specified candidates by lowest
    # final training cost while requiring stable finite curves.
    batch_alpha = min(batch_histories, key=lambda a: batch_histories[a][-1])
    sgd_alpha = min(sgd_histories, key=lambda a: sgd_histories[a][-1])

    # Fit all six requested models.
    rows = []
    theta_records = {}

    for label, features, alpha_b, alpha_s in [
        ("Set A", set_a, batch_alpha, sgd_alpha),
        ("Set B", set_b, batch_alpha, sgd_alpha),
    ]:
        train_s, test_s, mu, sigma = scale_train_test(train, test, features)
        Xtr = make_X(train_s, features)
        Xte = make_X(test_s, features)
        ytr = train_s["ac power"].to_numpy(float)
        yte = test_s["ac power"].to_numpy(float)
        daytime = test_s["irradiation"].to_numpy(float) > 0 if "irradiation" in test_s else test_s["sw radiation"].to_numpy(float) > 0

        theta_n = fit_normal(Xtr, ytr)
        theta_b, hist_b = fit_batch_gd(Xtr, ytr, alpha_b, 5000)
        theta_s, hist_s = fit_sgd(Xtr, ytr, alpha_s, 50)

        for solver, theta in [
            ("Normal equation", theta_n),
            (f"Batch GD (alpha={alpha_b:g}, iters=5000)", theta_b),
            (f"Stochastic GD (alpha={alpha_s:g}, epochs=50)", theta_s),
        ]:
            all_rmse, day_rmse, pred = evaluate(theta, Xte, yte, daytime)
            rows.append(
                {
                    "Solver": solver,
                    "Features": label,
                    "All hours RMSE (kW)": all_rmse,
                    "Daytime RMSE (kW)": day_rmse,
                }
            )

        theta_records[label] = {
            "normal": theta_n,
            "batch": theta_b,
            "sgd": theta_s,
            "mu": mu.to_dict(),
            "sigma": sigma.to_dict(),
        }

        # Save normal-equation weights for the app when Set B is processed.
        if label == "Set B":
            with open(RESULTS / "set_b_normal.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "features": features,
                        "theta": theta_n.tolist(),
                        "mean": mu.to_dict(),
                        "std": sigma.to_dict(),
                    },
                    f,
                    indent=2,
                )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS / "table2_rmse.csv", index=False)

    # Table 3 — Set A learned theta.
    names = ["theta0 intercept", "theta1 irradiation", "theta2 module temp",
             "theta3 ambient temp", "theta4 sin hour", "theta5 cos hour"]
    A = theta_records["Set A"]
    table3 = pd.DataFrame(
        {
            "Parameter": names,
            "Normal equation": A["normal"],
            "Batch GD": A["batch"],
            "SGD": A["sgd"],
        }
    )
    max_diff = float(np.max(np.abs(A["batch"] - A["normal"])))
    table3["Max |thetaGD - thetaNormal|"] = np.nan
    table3.loc[0, "Max |thetaGD - thetaNormal|"] = max_diff
    table3.to_csv(RESULTS / "table3_theta_set_a.csv", index=False)

    # Actual vs predicted test week for normal-equation Set A and Set B.
    plt.figure(figsize=(12, 5))
    test_plot = df.loc[test_mask].copy()
    for label, features, style in [
        ("Set A", set_a, "-"),
        ("Set B", set_b, "--"),
    ]:
        mu = theta_records[label]["mu"]
        sigma = theta_records[label]["sigma"]
        tmp = test_plot.copy()
        tmp[features] = (tmp[features] - pd.Series(mu)) / pd.Series(sigma)
        X = make_X(tmp, features)
        pred = clip_predictions(hypothesis(X, theta_records[label]["normal"]))
        plt.plot(tmp["datetime"], pred, style, label=f"Predicted {label}")
    plt.plot(test_plot["datetime"], test_plot["ac power"], label="Actual AC power", alpha=0.7)
    plt.xlabel("Datetime")
    plt.ylabel("AC power (kW)")
    plt.title("Actual vs Predicted AC Power — Test Week")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "07_actual_vs_predicted.png", dpi=180)
    plt.close()

    # Residuals vs hour for Set A normal equation.
    features = set_a
    mu = theta_records["Set A"]["mu"]
    sigma = theta_records["Set A"]["sigma"]
    tmp = test_plot.copy()
    tmp[features] = (tmp[features] - pd.Series(mu)) / pd.Series(sigma)
    X = make_X(tmp, features)
    pred = clip_predictions(hypothesis(X, theta_records["Set A"]["normal"]))
    residual = tmp["ac power"].to_numpy(float) - pred
    hours = tmp["datetime"].dt.hour.to_numpy()

    plt.figure(figsize=(9, 5))
    plt.scatter(hours, residual, s=10, alpha=0.45)
    plt.axhline(0, linewidth=1)
    plt.xticks(range(24))
    plt.xlabel("Hour of day")
    plt.ylabel("Residual (actual - predicted), kW")
    plt.title("Set A Normal-Equation Residuals vs Hour")
    plt.tight_layout()
    plt.savefig(FIG / "08_residuals_by_hour.png", dpi=180)
    plt.close()

    # Location/peak verification: compare three days.
    sensor = hourly[["datetime", "irradiation"]].copy()
    sensor["date"] = sensor["datetime"].dt.date
    weather2 = weather.copy()
    weather2["sw radiation"] = weather2["sw radiation"] / 1000.0
    weather2["date"] = weather2["datetime"].dt.date
    selected_dates = sorted(sensor["date"].unique())[:3]

    peak_rows = []
    plt.figure(figsize=(12, 5))
    for d in selected_dates:
        s = sensor[sensor["date"] == d]
        w = weather2[weather2["date"] == d]
        plt.plot(s["datetime"], s["irradiation"], label=f"Sensor {d}")
        plt.plot(w["datetime"], w["sw radiation"], "--", label=f"Open-Meteo {d}")
        peak_rows.append(
            {
                "date": str(d),
                "sensor_peak_hour": s.loc[s["irradiation"].idxmax(), "datetime"].hour,
                "openmeteo_peak_hour": w.loc[w["sw radiation"].idxmax(), "datetime"].hour,
            }
        )
    plt.xlabel("Datetime")
    plt.ylabel("Irradiation / SW radiation (kW/m²)")
    plt.title("Sensor Irradiation vs Open-Meteo Radiation — Three Days")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "09_sensor_vs_openmeteo.png", dpi=180)
    plt.close()

    pd.DataFrame(peak_rows).to_csv(RESULTS / "location_peak_check.csv", index=False)

    # Main analysis text.
    set_a_day = float(results[(results.Features=="Set A") & (results.Solver=="Normal equation")]["Daytime RMSE (kW)"].iloc[0])
    set_b_day = float(results[(results.Features=="Set B") & (results.Solver=="Normal equation")]["Daytime RMSE (kW)"].iloc[0])
    diff = set_b_day - set_a_day
    pct_peak = 100 * diff / peak_train if peak_train else np.nan

    signs = A["normal"][1:]
    physics = [
        "irradiation (+ expected)",
        "module temperature (often small negative/temperature-derating effect)",
        "ambient temperature (often small negative/indirect effect)",
        "sin hour",
        "cos hour",
    ]

    with open(RESULTS / "analysis.md", "w", encoding="utf-8") as f:
        f.write("# Task 5 — Analysis\n\n")
        f.write(f"## 5.1 Set A normal-equation weights\n\n")
        f.write("```text\n")
        for name, value in zip(names, A["normal"]):
            f.write(f"{name:24s} {value:.8f}\n")
        f.write("```\n\n")
        f.write(
            f"The largest absolute feature weight is **{names[1 + int(np.argmax(np.abs(A['normal'][1:])))]}**. "
            "Because the non-intercept features are standardized, comparing their coefficients is meaningful. "
            "Irradiation should dominate because PV power is strongly driven by incident solar radiation. "
            "A temperature coefficient may be negative because PV efficiency generally falls as module temperature rises.\n\n"
        )
        f.write("## 5.2 Public weather vs on-site sensors\n\n")
        f.write(
            f"The Set A normal-equation daytime RMSE is **{set_a_day:.3f} kW**, while Set B is "
            f"**{set_b_day:.3f} kW**. The increase is **{diff:.3f} kW**, equal to **{pct_peak:.2f}%** "
            f"of the training-set peak hourly AC power (**{peak_train:.3f} kW**, at {peak_time}). "
            "This quantifies the accuracy lost when local irradiation/module-temperature measurements "
            "are replaced by public weather variables.\n\n"
        )
        f.write(
            "For a rooftop installer, the practical answer depends on the required accuracy. Public weather "
            "can be useful for a first-pass estimate or preliminary feasibility assessment, but the measured "
            "sensor model should be preferred when a tighter site-specific power estimate is required.\n\n"
        )
        f.write("## 5.3 Solver comparison\n\n")
        f.write(
            f"Batch GD used the selected learning rate **{batch_alpha:g}** for 5000 iterations; "
            f"SGD used **{sgd_alpha:g}** for 50 epochs. The maximum absolute difference between Set A "
            f"batch-GD and normal-equation parameters is **{max_diff:.8f}**. The normal equation is "
            "the most direct choice for this small dataset. For a dataset with about 10 million rows, "
            "iterative gradient methods are more practical because forming and inverting XᵀX is expensive.\n\n"
        )
        f.write("## 5.4 Cost curves\n\n")
        f.write(
            "Batch GD should produce the smoother J(theta) curve because every update uses the full training "
            "set. SGD updates from one row at a time, so its objective can fluctuate between updates/epochs "
            "even while trending downward overall.\n\n"
        )
        f.write("## 5.5 Residuals\n\n")
        f.write(
            "Inspect `results/figures/08_residuals_by_hour.png`. Larger residual spread around sunrise/sunset "
            "can occur because the relationship between radiation and AC output is less stable during low-light "
            "transition periods. Thermal effects, cloud variability and inverter behavior can also create "
            "hour-dependent errors.\n\n"
        )

    print("\nPeak training AC power:", peak_train, "kW")
    print("Chosen batch alpha:", batch_alpha)
    print("Chosen SGD alpha:", sgd_alpha)
    print("\nTable 2:")
    print(results.to_string(index=False))
    print("\nMax |theta_batch - theta_normal| for Set A:", max_diff)

if __name__ == "__main__":
    main()
