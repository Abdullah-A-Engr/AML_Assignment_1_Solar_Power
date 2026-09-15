import json
from pathlib import Path
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "results" / "set_b_normal.json"

st.set_page_config(page_title="Solar Power Predictor", page_icon="☀️")
st.title("Solar Power Plant AC Output Predictor")
st.write("Set B — prediction using public weather variables and saved normal-equation weights.")

if not WEIGHTS.exists():
    st.error("Saved weights not found. Run `python src/train_eval.py` first.")
    st.stop()

with open(WEIGHTS, "r", encoding="utf-8") as f:
    model = json.load(f)

hour = st.slider("Hour of day", 0, 23, 12)
sw = st.number_input("Shortwave radiation (W/m²)", min_value=0.0, value=500.0, step=10.0)
temp = st.number_input("Temperature at 2 m (°C)", value=30.0, step=0.5)
cloud = st.number_input("Cloud cover (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

sin_h = np.sin(2 * np.pi * hour / 24.0)
cos_h = np.cos(2 * np.pi * hour / 24.0)

raw = {
    "sw radiation": sw,
    "temp 2m": temp,
    "cloud cover": cloud,
    "sin hour": sin_h,
    "cos hour": cos_h,
}

features = model["features"]
x = np.array(
    [1.0] + [
        (raw[f] - model["mean"][f]) / model["std"][f]
        for f in features
    ],
    dtype=float,
)
pred = max(0.0, float(x @ np.array(model["theta"], dtype=float)))

st.metric("Predicted AC Power", f"{pred:.2f} kW")
st.caption("Prediction is clipped at zero as required by the assignment.")
