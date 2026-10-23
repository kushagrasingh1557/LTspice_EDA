
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. READ LTSPICE DATA
# ==========================================

data_50mV = pd.read_csv("S2_50mv.csv")
data_1p1V = pd.read_csv("S2_11v.csv")

data_50mV = data_50mV.dropna(subset=["Vg", "Id(M1)"])
data_1p1V = data_1p1V.dropna(subset=["Vg", "Id(M1)"])

VGS_50 = data_50mV["Vg"].to_numpy(dtype=float)
IDS_50 = np.abs(data_50mV["Id(M1)"].to_numpy(dtype=float))

VGS_11 = data_1p1V["Vg"].to_numpy(dtype=float)
IDS_11 = np.abs(data_1p1V["Id(M1)"].to_numpy(dtype=float))

# Sort by gate voltage
order_50 = np.argsort(VGS_50)
VGS_50, IDS_50 = VGS_50[order_50], IDS_50[order_50]

order_11 = np.argsort(VGS_11)
VGS_11, IDS_11 = VGS_11[order_11], IDS_11[order_11]


# ==========================================
# 2. TRANSFER CURVE - LINEAR AXES
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(VGS_50, IDS_50 * 1e6, label="VDS = 50 mV")
plt.plot(VGS_11, IDS_11 * 1e6, label="VDS = 1.1 V")

plt.xlabel("VGS (V)")
plt.ylabel("ID (µA)")
plt.title("S2: Transfer Characteristics")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S2_transfer_linear.png", dpi=300)
plt.show()


# ==========================================
# 3. TRANSFER CURVE - LOG Y-AXIS
# ==========================================

plt.figure(figsize=(8, 5))

plt.semilogy(VGS_50, np.maximum(IDS_50, 1e-15),
             label="VDS = 50 mV")
plt.semilogy(VGS_11, np.maximum(IDS_11, 1e-15),
             label="VDS = 1.1 V")

plt.xlabel("VGS (V)")
plt.ylabel("ID (A)")
plt.title("S2: Transfer Characteristics - Log Scale")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig("S2_transfer_log.png", dpi=300)
plt.show()


# ==========================================
# 4. EXTRACT VTH USING MAXIMUM-gm TANGENT
# Use VDS = 50 mV
# ==========================================

gm = np.gradient(IDS_50, VGS_50)

idx = np.argmax(gm)

Vg_maxgm = VGS_50[idx]
Id_maxgm = IDS_50[idx]
gm_max = gm[idx]

# X-axis intercept of tangent
Vth = Vg_maxgm - Id_maxgm / gm_max

# Tangent line extended backward to the X-axis
Vg_tangent = np.linspace(Vth, Vg_maxgm, 100)
Id_tangent = gm_max * (Vg_tangent - Vth)


# ==========================================
# 5. EXTRACT ION AND IOFF
# Use VDS = 1.1 V
# ==========================================

Ion = np.interp(1.1, VGS_11, IDS_11)
Ioff = np.interp(0.0, VGS_11, IDS_11)


# ==========================================
# 6. PLOT VTH TANGENT
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(VGS_50, IDS_50 * 1e6,
         label="ID-VGS (VDS = 50 mV)")

plt.plot(Vg_tangent, Id_tangent * 1e6, "--",
         label="Maximum-gm tangent")

plt.scatter(Vg_maxgm, Id_maxgm * 1e6,
            zorder=5, label="Maximum-gm point")

plt.scatter(Vth, 0, zorder=5,
            label=f"VTH = {Vth:.4f} V")

plt.axhline(0, color="black", linewidth=0.8)

plt.xlabel("VGS (V)")
plt.ylabel("ID (µA)")
plt.title("S2: VTH Extraction")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S2_Vth_extraction.png", dpi=300)
plt.show()


# ==========================================
# 7. PRINT ONLY REQUIRED RESULTS
# ==========================================

print("\n===== S2 RESULTS =====")
print(f"VTH = {Vth:.4f} V")
print(f"ION = {Ion:.6e} A ({Ion*1e6:.4f} µA)")
print(f"IOFF = {Ioff:.6e} A ({Ioff*1e9:.4f} nA)")
