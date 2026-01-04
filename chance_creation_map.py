from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_csv_safely(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python")
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", engine="python")
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
    return df


def _to_num(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace('"', "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def style_dark():
    plt.rcParams.update({
        "figure.facecolor": "#0b0b0b",
        "axes.facecolor": "#0b0b0b",
        "savefig.facecolor": "#0b0b0b",
        "text.color": "white",
        "axes.labelcolor": "white",
        "xtick.color": "#bfbfbf",
        "ytick.color": "#bfbfbf",
        "axes.edgecolor": "#333333",
        "grid.color": "#222222",
        "font.family": "DejaVu Sans",
    })


def load_barca_only(base_dir: Path) -> pd.DataFrame:
    candidates = [
        base_dir / "barca.csv",
        base_dir / "barcelona.csv",
        base_dir / "fc_barcelona.csv",
        base_dir / "FC Barcelona.csv",
    ]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise RuntimeError("Could not find barca.csv in the same folder as this .py file.")

    df = _read_csv_safely(csv_path)

    if "player" not in df.columns:
        raise RuntimeError("Your barca.csv must contain a 'player' column.")

    for col in ["min", "goals", "assists", "xG", "xA", "xG90", "xA90"]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    if "min" not in df.columns:
        raise RuntimeError("Your barca.csv must contain a 'min' column.")

    # Build per90s if needed
    if "xA90" not in df.columns:
        if "xA" in df.columns and "min" in df.columns:
            df["xA90"] = (df["xA"] / df["min"]) * 90.0

    if "assists" not in df.columns:
        raise RuntimeError("Your barca.csv must contain an 'assists' column.")
    if "goals" not in df.columns:
        raise RuntimeError("Your barca.csv must contain a 'goals' column.")

    df["Assists/90"] = (df["assists"] / df["min"]) * 90.0
    df["Goals/90"] = (df["goals"] / df["min"]) * 90.0
    df["Attacking Contributions"] = df["goals"].fillna(0) + df["assists"].fillna(0)

    df["team"] = "FC Barcelona"
    return df


def _surname(name: str) -> str:
    name = str(name).strip()
    if not name:
        return name
    parts = name.split()
    return parts[-1] if len(parts) > 1 else parts[0]


def chance_creation_map_barca(
    df: pd.DataFrame,
    out_path: Path,
    min_mins: int = 450,
    min_attacking_contrib: int = 5,
    label_top_n: int = 8,
):
    d = df.copy()
    d = d[(d["min"] >= min_mins) & (d["Attacking Contributions"] >= min_attacking_contrib)].copy()
    d = d.dropna(subset=["xA90", "Assists/90"])

    if d.empty:
        raise RuntimeError("No Barça players match your filters (mins + attacking contributions).")

    # Bubble size (minutes) – small range so it doesn't get messy
    mins = d["min"].to_numpy()
    s = 120 + (mins - mins.min()) / (mins.max() - mins.min() + 1e-9) * 260  # 120..380

    x = d["xA90"].to_numpy()
    y = d["Assists/90"].to_numpy()

    # Choose a small set of labels: top by xA/90 + top by A/90
    top_xa = d.sort_values("xA90", ascending=False).head(label_top_n).index
    top_a = d.sort_values("Assists/90", ascending=False).head(label_top_n).index
    label_idx = list(dict.fromkeys(list(top_xa) + list(top_a)))  # preserve order, unique

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111)

    # main scatter (grey, lighter + smaller like you asked)
    ax.scatter(
        x, y,
        s=s,
        c="#bfbfbf",
        alpha=0.28,
        edgecolors="none",
        zorder=1
    )

    # highlight labelled leaders
    dx = d.loc[label_idx, "xA90"].to_numpy()
    dy = d.loc[label_idx, "Assists/90"].to_numpy()
    ds = (d.loc[label_idx, "min"].to_numpy() - mins.min()) / (mins.max() - mins.min() + 1e-9)
    ds = 220 + ds * 220

    ax.scatter(
        dx, dy,
        s=ds,
        c="#f2d16b",
        alpha=0.95,
        edgecolors="#0b0b0b",
        linewidths=1.0,
        zorder=3
    )

    # diagonal (A/90 = xA/90)
    lim = max(float(np.nanmax(x)), float(np.nanmax(y))) * 1.08
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.4, alpha=0.7, zorder=2)

    # labels (only for selected players)
    for i in label_idx:
        row = d.loc[i]
        lx, ly = float(row["xA90"]), float(row["Assists/90"])
        ax.text(lx + 0.01, ly + 0.01, _surname(row["player"]), fontsize=13, color="white", zorder=4)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.6)

    ax.set_xlabel("Expected Assists per 90 (xA/90)", fontsize=13)
    ax.set_ylabel("Assists per 90 (A/90)", fontsize=13)

    # Title + subtitle (subtitle safely lower)
    fig.suptitle(
        "LaLiga’s Biggest Chance Creators & Finishers",
        fontsize=28, fontweight="bold", color="#f2d16b", y=0.975
    )
    fig.text(
        0.5, 0.935,
        f"Chance Creation Map (xA/90 vs A/90) | FC Barcelona only | min {min_mins}+ mins | goals+assists ≥ {min_attacking_contrib}",
        ha="center", va="center", fontsize=13, color="#cfcfcf"
    )

    # small explainer
    ax.text(
        0.02, 0.95,
        "Above line = more assists than expected\nBelow line = fewer assists than expected",
        transform=ax.transAxes, ha="left", va="top", fontsize=12, color="#cfcfcf"
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    style_dark()
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    df_barca = load_barca_only(base_dir)

    chance_creation_map_barca(
        df_barca,
        out_dir / "barca_chance_creation_map.png",
        min_mins=450,
        min_attacking_contrib=5,
        label_top_n=8
    )

    print("Saved: outputs/barca_chance_creation_map.png")


if __name__ == "__main__":
    main()
