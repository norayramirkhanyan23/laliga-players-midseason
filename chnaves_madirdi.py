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


def _surname(name: str) -> str:
    name = str(name).strip()
    if not name:
        return name
    parts = name.split()
    return parts[-1] if len(parts) > 1 else parts[0]


def load_team_csv(base_dir: Path, candidates: list[str], team_label: str) -> pd.DataFrame:
    csv_path = next((base_dir / c for c in candidates if (base_dir / c).exists()), None)
    if csv_path is None:
        raise RuntimeError(f"Could not find CSV for {team_label}. Tried: {candidates}")

    df = _read_csv_safely(csv_path)
    if "player" not in df.columns:
        raise RuntimeError(f"{csv_path.name} must contain a 'player' column.")

    for col in ["min", "goals", "assists", "xA", "xA90"]:
        if col in df.columns:
            df[col] = _to_num(df[col])

    if "min" not in df.columns or "goals" not in df.columns or "assists" not in df.columns:
        raise RuntimeError(f"{csv_path.name} must contain 'min', 'goals', 'assists' columns.")

    if "xA90" not in df.columns:
        if "xA" in df.columns and "min" in df.columns:
            df["xA90"] = (df["xA"] / df["min"]) * 90.0
        else:
            df["xA90"] = np.nan

    df["Assists/90"] = (df["assists"] / df["min"]) * 90.0
    df["Attacking Contributions"] = df["goals"].fillna(0) + df["assists"].fillna(0)
    df["team"] = team_label
    return df


def chance_creation_map_team(
    df: pd.DataFrame,
    out_path: Path,
    team_label: str,
    dot_color: str,
    min_mins: int = 450,
    min_attacking_contrib: int = 5,
    label_top_n: int = 8,
):
    d = df.copy()
    d = d[(d["min"] >= min_mins) & (d["Attacking Contributions"] >= min_attacking_contrib)].copy()
    d = d.dropna(subset=["xA90", "Assists/90"])

    if d.empty:
        raise RuntimeError(f"No {team_label} players match your filters.")

    mins = d["min"].to_numpy()
    sizes = 120 + (mins - mins.min()) / (mins.max() - mins.min() + 1e-9) * 260

    # Label only a small set: top by xA/90 + top by A/90
    top_xa = d.sort_values("xA90", ascending=False).head(label_top_n).index
    top_a = d.sort_values("Assists/90", ascending=False).head(label_top_n).index
    label_idx = list(dict.fromkeys(list(top_xa) + list(top_a)))

    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111)

    # base points (muted)
    ax.scatter(
        d["xA90"].to_numpy(),
        d["Assists/90"].to_numpy(),
        s=sizes,
        c="#bfbfbf",
        alpha=0.25,
        edgecolors="none",
        zorder=1
    )

    # highlight labelled points
    ax.scatter(
        d.loc[label_idx, "xA90"].to_numpy(),
        d.loc[label_idx, "Assists/90"].to_numpy(),
        s=(sizes[d.index.get_indexer(label_idx)] * 1.15),
        c=dot_color,
        alpha=0.95,
        edgecolors="#0b0b0b",
        linewidths=1.0,
        zorder=3
    )

    # diagonal
    lim = max(float(d["xA90"].max()), float(d["Assists/90"].max())) * 1.08
    ax.plot([0, lim], [0, lim], linestyle="--", linewidth=1.4, alpha=0.7, zorder=2)

    # labels
    for i in label_idx:
        r = d.loc[i]
        ax.text(float(r["xA90"]) + 0.01, float(r["Assists/90"]) + 0.01, _surname(r["player"]),
                fontsize=13, color="white", zorder=4)

    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.grid(True, alpha=0.6)
    ax.set_xlabel("Expected Assists per 90 (xA/90)", fontsize=13)
    ax.set_ylabel("Assists per 90 (A/90)", fontsize=13)

    fig.suptitle("LaLiga’s Biggest Chance Creators & Finishers",
                 fontsize=28, fontweight="bold", color="#f2d16b", y=0.975)
    fig.text(
        0.5, 0.935,
        f"Chance Creation Map (xA/90 vs A/90) | {team_label} only | min {min_mins}+ mins | goals+assists ≥ {min_attacking_contrib}",
        ha="center", va="center", fontsize=13, color="#cfcfcf"
    )

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

    # Real Madrid
    df_real = load_team_csv(
        base_dir,
        candidates=["real.csv", "realmadrid.csv", "real_madrid.csv"],
        team_label="Real Madrid"
    )
    chance_creation_map_team(
        df_real,
        out_dir / "real_chance_creation_map.png",
        team_label="Real Madrid",
        dot_color="#3b82f6",
        min_mins=450,
        min_attacking_contrib=5,
        label_top_n=8
    )

    # Atlético Madrid
    df_atleti = load_team_csv(
        base_dir,
        candidates=["atleti.csv", "atletico.csv", "atletico_madrid.csv"],
        team_label="Atlético de Madrid"
    )
    chance_creation_map_team(
        df_atleti,
        out_dir / "atleti_chance_creation_map.png",
        team_label="Atlético de Madrid",
        dot_color="#a56eff",
        min_mins=450,
        min_attacking_contrib=5,
        label_top_n=8
    )

    print("Saved:")
    print(" - outputs/real_chance_creation_map.png")
    print(" - outputs/atleti_chance_creation_map.png")


if __name__ == "__main__":
    main()
