import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ReLU Activation Function", page_icon="📈", layout="centered")
st.title("ReLU Activation Function")
st.write("The Rectified Linear Unit (ReLU) returns zero for negative inputs and returns the input itself for positive inputs.")

def relu(x):
    return np.maximum(0, x)

x_min = st.slider("Minimum input value", -20, 0, -10)
x_max = st.slider("Maximum input value", 1, 20, 10)
x = np.linspace(x_min, x_max, 500)
y = relu(x)

fig, ax = plt.subplots()
ax.plot(x, y, label="ReLU: f(x) = max(0, x)")
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_title("ReLU Activation Curve")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.subheader("Try an input value")
value = st.number_input("Enter x", value=2.0, step=0.5)
st.success(f"ReLU({value:.2f}) = {relu(value):.2f}")
st.latex(r"f(x) = \max(0,x)")
st.caption("ReLU output range: [0, +∞). It is commonly used in hidden layers of neural networks.")
