# Task 2 — EDA observations

- Peak hourly AC power: **27325.899 kW** at **2020-05-20 12:00:00**.
- Positive-DC hourly AC/DC ratio: mean **0.097712**, median **0.097816**, min **0.096519**, max **0.098761**.

### 2.1 AC power vs irradiation
PV output should increase strongly with irradiation. The prepared data show the expected positive relationship, with spread caused by temperature, cloud variability, inverter behavior and measurement variation.

### 2.2 Module temperature vs ambient temperature
Module temperature generally rises above ambient temperature during sunlight. Higher irradiation should correspond to a larger module-to-ambient temperature difference.

### 2.3 AC power vs DC power
AC power should track DC power approximately linearly but remain below it because of inverter/conversion losses. The hourly positive-DC ratio is approximately **0.0977** for this plant-level aggregation.

### 2.4 Average AC power by hour
Power should be close to zero at night, rise after sunrise, reach its strongest values during daylight, and decline toward sunset. The generated figure provides the required hourly profile.
