
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. READ DATA

df = pd.read_csv("S3.csv")

# Convert LTspice stepped parameter values like 300m to volts
def convert_voltage(value):
    value = str(value).strip()
    if value.endswith("m"):
        return float(value[:-1]) * 1e-3
    elif value.endswith("u"):
        return float(value[:-1]) * 1e-6
    elif value.endswith("n"):
        return float(value[:-1]) * 1e-9
    return float(value)

df["VGS"] = df["Parameter_Value"].apply(convert_voltage)

# Absolute drain current, since LTspice may export a negative sign
df["ID"] = np.abs(df["Id(M1)"])


# 2. PLOT ID-VDS CURVES


plt.figure(figsize=(9, 6))

for vgs, group in df.groupby("VGS"):
    group = group.sort_values("Vd")
    plt.plot(
        group["Vd"],
        group["ID"] * 1e3,
        label=f"VGS = {vgs:.1f} V"
    )

plt.xlabel("VDS (V)")
plt.ylabel("ID (mA)")
plt.title("S3: NMOS Output Characteristics")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S3_output_characteristics.png", dpi=300)
plt.show()

# 3. SELECT VGS = 1.5 V

group = df[np.isclose(df["VGS"], 1.5)].sort_values("Vd")

VDS = group["Vd"].to_numpy()
ID = group["ID"].to_numpy()

# 4. CALCULATE gds AND ro

gds = np.gradient(ID, VDS)  # A/V

# Avoid division by zero
ro = np.full_like(gds, np.nan, dtype=float)
valid = np.abs(gds) > 0
ro[valid] = 1 / gds[valid]  # ohms

# Find the point closest to VDS = 1.0 V
idx = np.argmin(np.abs(VDS - 1.0))

ro_ohms = ro[idx]

# W = 1 um, so width-normalized ro has the same numerical
# value in kOhm*um as ro in kOhm.
ro_kohm_um = ro_ohms / 1e3 * 1.0

print("\n===== S3 RESULTS =====")
print(f"VGS = 1.5 V")
print(f"VDS used = {VDS[idx]:.3f} V")
print(f"ID = {ID[idx]*1e3:.6f} mA")
print(f"gds = {gds[idx]:.6e} S")
print(f"ro = {ro_ohms/1e3:.4f} kOhm")
print(f"Width-normalized ro = {ro_kohm_um:.4f} kOhm.um")

# 5. PLOT ro VERSUS VDS FOR VGS = 1.5 V

plt.figure(figsize=(9, 6))

plt.plot(VDS, ro / 1e3)
plt.axvline(1.0, linestyle="--", label="VDS = 1.0 V")

plt.xlabel("VDS (V)")
plt.ylabel("Output Resistance ro (kOhm)")
plt.title("S3: Output Resistance at VGS = 1.5 V")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S3_output_resistance.png", dpi=300)
plt.show()
