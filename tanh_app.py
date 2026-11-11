import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Tanh Activation Function", page_icon="📈", layout="centered")
st.title("Hyperbolic Tangent (Tanh) Activation Function")
st.write("The Tanh function maps real-valued inputs to outputs between negative one and positive one.")

def tanh(x):
    return np.tanh(x)

x_min = st.slider("Minimum input value", -20, 0, -10)
x_max = st.slider("Maximum input value", 1, 20, 10)
x = np.linspace(x_min, x_max, 500)
y = tanh(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="Tanh: f(x) = tanh(x)")
ax.axhline(0, linewidth=0.8)
ax.axhline(1, linestyle="--", linewidth=0.8, label="y = ±1")
ax.axhline(-1, linestyle="--", linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Tanh Activation Curve")
ax.set_ylim(-1.1, 1.1)
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.subheader("Try an input value")
value = st.number_input("Enter x", value=0.0, step=0.5)
st.success(f"Tanh({value:.2f}) = {tanh(value):.4f}")
st.latex(r"f(x) = \tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}")
st.caption("Tanh output range: (-1, 1). It is zero-centred and can be used in hidden layers.")
