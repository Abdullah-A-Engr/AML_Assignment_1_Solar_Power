from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(DATA / "plant1_hourly.csv", parse_dates=["datetime"])

    # 2.1 AC power vs irradiation
    plt.figure(figsize=(8, 5))
    plt.scatter(df["irradiation"], df["ac power"], s=8, alpha=0.45)
    plt.xlabel("Irradiation (kW/m²)")
    plt.ylabel("AC power (kW)")
    plt.title("AC Power vs Irradiation")
    plt.tight_layout()
    plt.savefig(FIG / "01_ac_vs_irradiation.png", dpi=180)
    plt.close()

    # 2.2 Module temp vs ambient temp, colored by irradiation
    plt.figure(figsize=(8, 5))
    sc = plt.scatter(
        df["ambient temp"], df["module temp"],
        c=df["irradiation"], s=10, alpha=0.55
    )
    plt.xlabel("Ambient temperature (°C)")
    plt.ylabel("Module temperature (°C)")
    plt.title("Module Temperature vs Ambient Temperature")
    plt.colorbar(sc, label="Irradiation (kW/m²)")
    plt.tight_layout()
    plt.savefig(FIG / "02_module_vs_ambient.png", dpi=180)
    plt.close()

    # 2.3 AC vs DC power and ratio
    plt.figure(figsize=(8, 5))
    plt.scatter(df["dc power"], df["ac power"], s=8, alpha=0.45)
    plt.xlabel("DC power (kW)")
    plt.ylabel("AC power (kW)")
    plt.title("AC Power vs DC Power")
    plt.tight_layout()
    plt.savefig(FIG / "03_ac_vs_dc.png", dpi=180)
    plt.close()

    daytime = df["dc power"] > 0
    ratio = (df.loc[daytime, "ac power"] / df.loc[daytime, "dc power"]).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    print("\nAC/DC ratio on positive-DC rows:")
    print(ratio.describe())

    # 2.4 Average AC power by hour
    by_hour = df.groupby(df["datetime"].dt.hour)["ac power"].mean()
    plt.figure(figsize=(8, 5))
    plt.plot(by_hour.index, by_hour.values, marker="o")
    plt.xticks(range(24))
    plt.xlabel("Hour of day")
    plt.ylabel("Average AC power (kW)")
    plt.title("Average AC Power by Hour of Day")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "04_average_ac_by_hour.png", dpi=180)
    plt.close()

    # Save a compact EDA summary for the report.
    peak_idx = df["ac power"].idxmax()
    peak_row = df.loc[peak_idx]
    with open(ROOT / "results" / "eda_summary.md", "w", encoding="utf-8") as f:
        f.write("# Task 2 — EDA observations\n\n")
        f.write(
            f"- Peak hourly AC power: **{peak_row['ac power']:.3f} kW** at "
            f"{peak_row['datetime']}.\n"
        )
        f.write(
            "- **AC power vs irradiation:** PV output should increase strongly with "
            "irradiation. The scatter should therefore show a positive relationship, "
            "with some spread caused by temperature, clouds, inverter behavior and "
            "measurement variation.\n\n"
        )
        f.write(
            "- **Module vs ambient temperature:** module temperature should generally "
            "rise above ambient temperature under sunlight. Points with higher "
            "irradiation should tend to show a larger temperature difference.\n\n"
        )
        f.write(
            "- **AC vs DC power:** AC output should track DC output approximately "
            "linearly but remain lower because of inverter and conversion losses. "
            "The slope gives a useful indication of the conversion ratio.\n\n"
        )
        f.write(
            "- **Average AC power by hour:** output should be near zero at night, "
            "increase after sunrise, reach its highest values around the middle of "
            "the daylight period, and decline toward sunset.\n"
        )

if __name__ == "__main__":
    main()
