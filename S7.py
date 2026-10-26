
import pandas as pd
import matplotlib.pyplot as plt

# Load the LTspice exported CSV
df = pd.read_csv("S7.csv")

# Check the actual column names if needed
print(df.columns)

# Extract input and output voltages
vin = df["Vin"]
vout = df["V(out)"]

# Plot the inverter transfer characteristic
plt.figure(figsize=(8, 5))
plt.plot(vin, vout, linewidth=2, label="CMOS inverter")

plt.xlabel("$V_{in}$ (V)")
plt.ylabel("$V_{out}$ (V)")
plt.title("S7: CMOS Inverter Transfer Characteristic")
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("S7.png", dpi=300)
plt.show()
