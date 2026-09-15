# Task 5 — Analysis

## 5.1 Set A normal-equation weights

```text
theta0 intercept         6796.26007116
theta1 irradiation       8195.27144474
theta2 module temp       8.92710567
theta3 ambient temp      -28.58232896
theta4 sin hour          -30.23189298
theta5 cos hour          -369.44388566
```

The largest absolute feature weight is **theta1 irradiation**. Because the non-intercept features are standardized, comparing their coefficients is meaningful. Irradiation should dominate because PV power is strongly driven by incident solar radiation. A temperature coefficient may be negative because PV efficiency generally falls as module temperature rises.

## 5.2 Public weather vs on-site sensors

The Set A normal-equation daytime RMSE is **918.875 kW**, while Set B is **3424.878 kW**. The increase is **2506.003 kW**, equal to **9.17%** of the training-set peak hourly AC power (**27325.899 kW**, at 2020-05-20 12:00:00). This quantifies the accuracy lost when local irradiation/module-temperature measurements are replaced by public weather variables.

For a rooftop installer, the practical answer depends on the required accuracy. Public weather can be useful for a first-pass estimate or preliminary feasibility assessment, but the measured sensor model should be preferred when a tighter site-specific power estimate is required.

## 5.3 Solver comparison

Batch GD used the selected learning rate **0.001** for 5000 iterations; SGD used **0.01** for 50 epochs. The maximum absolute difference between Set A batch-GD and normal-equation parameters is **4102.05180617**. The normal equation is the most direct choice for this small dataset. For a dataset with about 10 million rows, iterative gradient methods are more practical because forming and inverting XᵀX is expensive.

## 5.4 Cost curves

Batch GD should produce the smoother J(theta) curve because every update uses the full training set. SGD updates from one row at a time, so its objective can fluctuate between updates/epochs even while trending downward overall.

## 5.5 Residuals

Inspect `results/figures/08_residuals_by_hour.png`. Larger residual spread around sunrise/sunset can occur because the relationship between radiation and AC output is less stable during low-light transition periods. Thermal effects, cloud variability and inverter behavior can also create hour-dependent errors.

