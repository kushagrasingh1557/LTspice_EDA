import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. READ THE THREE DATASETS

original = pd.read_csv("S2_11v.csv")
ua_ub_zero = pd.read_csv("S4_uaub.csv")
vsat_x100 = pd.read_csv("S4_vsat.csv")

# 2. CLEAN AND SORT THE DATA

def prepare_data(df):
    df = df.dropna(subset=["Vg", "Id(M1)"]).copy()

    df["Vg"] = pd.to_numeric(df["Vg"])
    df["Id(M1)"] = pd.to_numeric(df["Id(M1)"])

    df = df.sort_values("Vg")

    VGS = df["Vg"].to_numpy()
    ID = np.abs(df["Id(M1)"].to_numpy())

    return VGS, ID


VGS_original, ID_original = prepare_data(original)
VGS_ua, ID_ua = prepare_data(ua_ub_zero)
VGS_vsat, ID_vsat = prepare_data(vsat_x100)

# 3. PLOT ALL THREE CURVES

plt.figure(figsize=(9, 6))

plt.plot(
    VGS_original, ID_original * 1e6,
    label="Original PTM model"
)

plt.plot(
    VGS_ua, ID_ua * 1e6,
    label="UA = UB = 0"
)

plt.plot(
    VGS_vsat, ID_vsat * 1e6,
    label="VSAT x 100"
)

plt.xlabel("VGS (V)")
plt.ylabel("ID (µA)")
plt.title("S4: Effect of Mobility Degradation and Velocity Saturation")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig("S4_linear.png", dpi=300)
plt.show()
