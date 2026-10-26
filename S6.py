import pandas as pd
import matplotlib.pyplot as plt

# Read data
tcad_transfer = pd.read_csv("E50.4.csv")
spice_transfer = pd.read_csv("S6_transfer_final.csv")

tcad_output = pd.read_csv("E6c.csv")
spice_output = pd.read_csv("S6_output_final.csv")

# Remove whitespace from column names
for df in [tcad_transfer, spice_transfer, tcad_output, spice_output]:
    df.columns = df.columns.str.strip()

tcad_transfer = tcad_transfer[
    tcad_transfer["gate OuterVoltage"] <= 1.1
]

# ==============================
# 1. TRANSFER CHARACTERISTIC
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    tcad_transfer["gate OuterVoltage"],
    tcad_transfer["drain TotalCurrent"] * 1e3,
    label="TCAD",
    linewidth=2
)

plt.plot(
    spice_transfer["Vg"],
    spice_transfer["Id(M1)"] * 1e3,
    label="SPICE",
    linestyle="--",
    linewidth=2
)

plt.xlabel("VGS (V)")
plt.ylabel("ID (mA)")
plt.title("S6: Transfer Characteristic (VDS = 0.4 V)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S6_transfer_overlay.png", dpi=300)
plt.show()

# ==============================
# 2. OUTPUT CHARACTERISTIC
# ==============================

plt.figure(figsize=(8, 5))

plt.plot(
    tcad_output["drain OuterVoltage"],
    tcad_output["drain TotalCurrent"] * 1e3,
    label="TCAD",
    linewidth=2
)

plt.plot(
    spice_output["Vd"],
    spice_output["Id(M1)"] * 1e3,
    label="SPICE",
    linestyle="--",
    linewidth=2
)

plt.xlabel("VDS (V)")
plt.ylabel("ID (mA)")
plt.title("S6: Output Characteristic (VGS = 0.9 V)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S6_output_overlay.png", dpi=300)
plt.show()