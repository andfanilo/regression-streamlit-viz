from dataclasses import dataclass
from math import sqrt

import SessionState

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

import altair as alt
import streamlit as st

np.random.seed(0)
X_MIN = 0
X_MAX = 1

state = SessionState.get(min_rmse=999)


@dataclass
class Weight:
    w0: float
    w1: float
    w2: float


@st.cache
def build_dataset(xres):
    X_source = np.linspace(X_MIN, X_MAX, xres)
    y_source = (
        np.polynomial.polynomial.polyval(X_source, [0, 2, 5])
        + np.sin(8 * X_source)
        + 0.5 * np.random.normal(size=xres)
    )

    return pd.DataFrame({"x_source": X_source, "y_source": y_source})


def build_regression(source_df, x_res, w: Weight):
    X_reg = source_df["x_source"].copy()
    y_reg = np.polynomial.polynomial.polyval(X_reg, [0, w.w0, w.w1]) + np.sin(
        w.w2 * X_reg
    )
    return pd.DataFrame({"x_reg": X_reg, "y_reg": y_reg})


def build_error(source_df, res_df):
    y_error = np.abs(source_df["y_source"] - res_df["y_reg"])
    return pd.DataFrame({"x": source_df["x_source"], "y_err": y_error})


def compute_rmse(source_df, res_df):
    rmse = sqrt(mean_squared_error(source_df["y_source"], res_df["y_reg"]))
    return rmse


def update_min_rmse(current_rmse, state):
    if current_rmse < state.min_rmse:
        state.min_rmse = current_rmse


st.title("Regression")
st.markdown("Play with weights in sidebar and see if you can fit the points.")
st.markdown("$$f(x)=w_0 \\times x+w_1 \\times x^2 + sin(w_2 \\times x)$$")

st.sidebar.subheader("Parameters")
xres = st.sidebar.slider("Number of points", 100, 1000, 100, 100)
w0 = st.sidebar.slider("w0", 0.0, 10.0, 1.0, 0.5)
w1 = st.sidebar.slider("w1", 0.0, 10.0, 1.0, 0.5)
w2 = st.sidebar.slider("w2", 0.0, 10.0, 1.0, 0.5)
w = Weight(w0, w1, w2)

source_data = build_dataset(xres)
regression_data = build_regression(source_data, xres, w)
error_data = build_error(source_data, regression_data)
rmse = compute_rmse(source_data, regression_data)
update_min_rmse(rmse, state)

alt_source = (
    alt.Chart(source_data)
    .mark_circle(color="black")
    .encode(alt.X("x_source", title=None), alt.Y("y_source", title=None))
)
alt_reg = (
    alt.Chart(regression_data)
    .mark_line()
    .encode(alt.X("x_reg", title=None), alt.Y("y_reg", title=None))
)
alt_err = (
    alt.Chart(error_data)
    .mark_area(clip=True, opacity=0.6)
    .encode(
        alt.X("x", title=None),
        alt.Y("y_err", title="abs error |data - reg|", scale=alt.Scale(domain=(0, 2))),
    )
    .properties(height=200)
)

chart = (alt_source + alt_reg) & alt_err

st.altair_chart(chart, width=-1)
rmse_text = st.text(f"Current RMSE : {rmse}")
min_rmse_text = st.text(f"Minimum RMSE : {state.min_rmse}")
