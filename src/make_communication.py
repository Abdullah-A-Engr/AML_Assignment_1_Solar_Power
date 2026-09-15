from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "communication"
OUT.mkdir(exist_ok=True)

def main():
    rmse = pd.read_csv(RESULTS / "table2_rmse.csv")
    prep = pd.read_csv(RESULTS / "table1_data_preparation.csv")
    theta = pd.read_csv(RESULTS / "table3_theta_set_a.csv")
    peaks = pd.read_csv(RESULTS / "location_peak_check.csv")

    a = rmse[(rmse.Features == "Set A") & (rmse.Solver == "Normal equation")].iloc[0]
    b = rmse[(rmse.Features == "Set B") & (rmse.Solver == "Normal equation")].iloc[0]
    peak = prep.loc[prep.metric == "Hourly rows after resampling", "value"].iloc[0]

    # Read peak from EDA output if available.
    eda = (ROOT / "results" / "eda_summary.md").read_text(encoding="utf-8")

    blog = f"""# Predicting Solar Power Plant Output from Weather: What Do We Lose Without On-Site Sensors?

## The practical question

A solar plant's AC power output depends strongly on incoming solar radiation and on the thermal conditions of the photovoltaic modules. Plant operators can measure these variables with on-site sensors, but a rooftop installer or early-stage planner may only have access to public weather data.

This project asks a simple practical question: **how much prediction accuracy is lost when on-site irradiation and temperature measurements are replaced with free public weather data?**

The assignment uses a real Indian solar-power dataset and implements linear regression from first principles rather than relying on a machine-learning library.

## Data and preparation

The dataset contains inverter-level generation measurements and plant-level weather-sensor measurements for Plant 1. The raw generation data are recorded every 15 minutes, while the sensor data provide ambient temperature, module temperature and irradiation.

The raw timestamps were inspected before conversion because the two files use different timestamp formats. Generation timestamps are written in day-month-year format, while the sensor timestamps use an ISO-style year-month-day format. After parsing, inverter-level AC and DC power were summed at each timestamp to obtain plant-level power. The plant-level power was then merged with the weather sensors and resampled to hourly means.

The final hourly dataset contains approximately 34 days of observations, matching the assignment's expected time scale. The cleaned data were saved as `data/plant1_hourly.csv`.

## Exploratory observations

Four plots were produced: AC power versus irradiation, module temperature versus ambient temperature, AC power versus DC power, and average AC power by hour of day.

The physical expectation is straightforward. AC power should increase strongly with irradiation, module temperature should generally rise above ambient temperature under sunlight, AC power should track DC power below it because of conversion losses, and hourly production should be close to zero at night with a broad daytime peak.

These relationships provide an important sanity check before fitting a model. In particular, a large daytime prediction error would be suspicious if the timestamp alignment or unit conversion were incorrect.

## Public weather data

For the public-weather model, historical hourly weather was downloaded programmatically from Open-Meteo for the assignment's Plant 1 coordinates: 14.82° N, 78.28° E, with the Asia/Kolkata time zone. The downloaded variables were shortwave radiation, 2-metre temperature and cloud cover.

Open-Meteo reports shortwave radiation in W/m², while the plant sensor irradiation is expressed in kW/m². Therefore, the public radiation was divided by 1000 before comparison and modelling.

A three-day time-series comparison was also produced. The peak-hour check is important because a one-hour shift would indicate a likely coordinate or time-zone problem.

## Regression by hand

Two feature sets were tested.

**Set A** uses on-site irradiation, module temperature and ambient temperature.  
**Set B** uses public shortwave radiation, 2-metre temperature and cloud cover.

Both sets also include sine and cosine hour-of-day features so that the model can represent daily seasonality.

The data were split by date without shuffling: 15 May through 10 June for training and 11 June through 17 June for testing. Feature scaling statistics were calculated using training rows only and then applied unchanged to the test rows.

Three solvers were implemented manually with NumPy: the normal equation, batch gradient descent and stochastic gradient descent. Predictions were clipped at zero because a physical plant cannot produce negative AC power.

## Results

The normal-equation daytime RMSE was:

- **Set A:** {a['Daytime RMSE (kW)']:.3f} kW
- **Set B:** {b['Daytime RMSE (kW)']:.3f} kW

Therefore, replacing local sensor information with public weather increased daytime RMSE by **{b['Daytime RMSE (kW)'] - a['Daytime RMSE (kW)']:.3f} kW**.

The complete comparison is in `results/table2_rmse.csv`, including all hours and daytime-only RMSE for all six requested fits.

The learned Set A parameters are also reported in `results/table3_theta_set_a.csv`. Because the predictors are standardized, the coefficients can be compared on a common scale. Irradiation should have the strongest positive relationship with AC power, while temperature effects can be smaller and may be negative because photovoltaic efficiency tends to decline as module temperature rises.

## What does this mean for a rooftop installer?

The answer is not simply that public weather is useless. Public weather can be valuable when no site-specific sensors exist, especially for preliminary feasibility analysis and rough power estimation.

However, the experiment shows exactly how much predictive accuracy is sacrificed when local measurements are replaced by public data. If a project needs a tighter site-specific estimate, local irradiation and temperature measurements are preferable. If the goal is an early-stage estimate where a larger error is acceptable, public weather can still be useful and has the major advantage of being free and accessible.

This distinction matters in engineering: the best model is not necessarily the most sophisticated one; it is the model whose accuracy is appropriate for the decision being made.

## Solver choice

For this relatively small dataset, the normal equation is attractive because it directly gives the least-squares solution. Batch gradient descent is useful for demonstrating iterative optimization and converges toward the same solution when the learning rate and iteration count are appropriate. SGD uses much cheaper individual updates but produces a noisier optimization path.

For a dataset with millions of rows, iterative gradient methods become more practical because explicitly forming and inverting XᵀX becomes increasingly expensive.

## Conclusion

The project demonstrates a complete supervised-learning workflow without using a regression library: data cleaning, time alignment, public API data collection, exploratory analysis, feature scaling, regression from first principles, gradient-based optimization, time-ordered evaluation and a small deployment interface.

Most importantly, the experiment answers an engineering question rather than stopping at model metrics: **public weather can provide a useful approximation, but on-site sensor information gives a more site-specific prediction and should be preferred when accuracy requirements are strict.**

Figures are available in `results/figures/`.
"""
    (OUT / "medium_blog.md").write_text(blog, encoding="utf-8")

    linkedin = f"""I completed an Applied Machine Learning project on predicting solar power plant output from weather data.

The project compares two approaches:

• Set A — on-site irradiation, module temperature and ambient temperature
• Set B — public Open-Meteo shortwave radiation, 2 m temperature and cloud cover

I implemented the regression methods from scratch with NumPy:
• Normal equation
• Batch gradient descent
• Stochastic gradient descent

The data were split chronologically, features were scaled using training statistics only, and evaluation used RMSE on the held-out test week.

Normal-equation daytime RMSE:
Set A: {a['Daytime RMSE (kW)']:.3f} kW
Set B: {b['Daytime RMSE (kW)']:.3f} kW

The main engineering question was: how much accuracy is lost when local sensors are replaced by free public weather data?

Blog: [ADD MEDIUM LINK]
Repository: [ADD GITHUB LINK]

#MachineLearning #Python #SolarEnergy #Regression #DataScience #RenewableEnergy
"""
    (OUT / "linkedin_post.md").write_text(linkedin, encoding="utf-8")
    print("Generated communication drafts in communication/")

if __name__ == "__main__":
    main()
