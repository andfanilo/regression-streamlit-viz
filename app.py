from dataclasses import dataclass

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st


@dataclass
class Weight:
    w0: float
    w1: float
    w2: float


@st.cache
def build_dataset(xres):
    X_source = np.linspace(0.01, 1, xres)
    y_source = (
        np.polynomial.polynomial.polyval(X_source, [0, 2, 5])
        + np.sin(8 * X_source)
        + 0.5 * np.random.normal(size=xres)
    )

    return pd.DataFrame({"x_source": X_source, "y_source": y_source})


def build_regression(x_res, w: Weight):
    X_reg = np.linspace(0.01, 1, xres).astype(np.float)
    y_reg = np.polynomial.polynomial.polyval(X_reg, [0, w.w0, w.w1]) + np.sin(
        w.w2 * X_reg
    )

    return pd.DataFrame({"x_reg": X_reg, "y_reg": y_reg})


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
regression_data = build_regression(xres, w)

alt_source = (
    alt.Chart(source_data)
    .mark_circle(color="black")
    .encode(alt.X("x_source"), alt.Y("y_source"))
)
alt_reg = alt.Chart(regression_data).mark_line().encode(alt.X("x_reg"), alt.Y("y_reg"))

chart = alt.layer(alt_source, alt_reg)

st.altair_chart(chart, width=-1)
