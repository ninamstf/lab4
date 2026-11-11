import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sigmoid Activation Function", page_icon="📈", layout="centered")
st.title("Sigmoid Activation Function")
st.write("The Sigmoid function maps any real-valued input to an output between zero and one.")

def sigmoid(x):
    # Clipping keeps exp() numerically stable for large input magnitudes.
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

x_min = st.slider("Minimum input value", -20, 0, -10)
x_max = st.slider("Maximum input value", 1, 20, 10)
x = np.linspace(x_min, x_max, 500)
y = sigmoid(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="Sigmoid: f(x) = 1 / (1 + e⁻ˣ)")
ax.axhline(0.5, linestyle="--", linewidth=0.8, label="y = 0.5")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("Sigmoid Activation Curve")
ax.set_ylim(-0.05, 1.05)
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.subheader("Try an input value")
value = st.number_input("Enter x", value=0.0, step=0.5)
st.success(f"Sigmoid({value:.2f}) = {sigmoid(value):.4f}")
st.latex(r"f(x) = \frac{1}{1+e^{-x}}")
st.caption("Sigmoid output range: (0, 1). It is often used in the output layer for binary classification.")
