
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Folder containing all extracted S5 CSV files
DATA_FOLDER = Path(".")
# READ AND CLEAN DATA
def read_data(filename, x_column):
    df = pd.read_csv(filename)

    required = [x_column, "Id(M1)", "Parameter_Value"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"Skipping {filename.name}: missing {missing}")
        return None

    df = df.dropna(subset=required).copy()

    df[x_column] = pd.to_numeric(df[x_column], errors="coerce")
    df["Id(M1)"] = pd.to_numeric(df["Id(M1)"], errors="coerce")
    df = df.dropna(subset=[x_column, "Id(M1)"])

    return df

# FIND FILES FOR EACH PARAMETER
files = list(DATA_FOLDER.glob("S5_*.csv"))

idvg_files = {
    f.stem.lower().replace("s5_idvg_", ""): f
    for f in files if "_idvg_" in f.stem.lower()
}

idvd_files = {
    f.stem.lower().replace("s5_idvd_", ""): f
    for f in files if "_idvd_" in f.stem.lower()
}

# Only parameters with BOTH datasets
parameters = sorted(set(idvg_files) & set(idvd_files))

print("Experiments with both datasets:", len(parameters))
print("Parameters:", parameters)

# PLOT SIDE-BY-SIDE FOR EACH PARAMETER
for parameter in parameters:

    vg_file = idvg_files[parameter]
    vd_file = idvd_files[parameter]

    vg_data = read_data(vg_file, "Vg")
    vd_data = read_data(vd_file, "Vd")

    if vg_data is None or vd_data is None:
        continue

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # LEFT: ID-VG

    for value, group in vg_data.groupby("Parameter_Value", sort=False):
        group = group.sort_values("Vg")

        axes[0].plot(
            group["Vg"],
            np.abs(group["Id(M1)"]) * 1e6,
            label=f"{parameter.upper()} = {value}"
        )

    axes[0].set_xlabel("VGS (V)")
    axes[0].set_ylabel("ID (µA)")
    axes[0].set_title("ID-VG at VDS = 0.4 V")
    axes[0].grid(True)
    axes[0].legend(fontsize=8)

    # RIGHT: ID-VD

    for value, group in vd_data.groupby("Parameter_Value", sort=False):
        group = group.sort_values("Vd")

        axes[1].plot(
            group["Vd"],
            np.abs(group["Id(M1)"]) * 1e6,
            label=f"{parameter.upper()} = {value}"
        )

    axes[1].set_xlabel("VDS (V)")
    axes[1].set_ylabel("ID (µA)")
    axes[1].set_title("ID-VD at VGS = 1.1 V")
    axes[1].grid(True)
    axes[1].legend(fontsize=8)

    fig.suptitle(f"S5: Effect of {parameter.upper()}", fontsize=14)
    fig.tight_layout()

    output_file = f"S5_{parameter}_side_by_side.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {output_file}")


print("\nDone! All side-by-side plots generated.")
