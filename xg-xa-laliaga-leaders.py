from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TEAM_NAME_MAP = {
    "alaves": "Deportivo Alavés",
    "atleti": "Atlético de Madrid",
    "atletic": "Athletic Club",
    "athletic": "Athletic Club",
    "barca": "FC Barcelona",
    "betis": "Real Betis",
    "celta": "Celta de Vigo",
    "elche": "Elche CF",
    "espanyol": "RCD Espanyol",
    "getafe": "Getafe CF",
    "girona": "Girona FC",
    "levante": "Levante UD",
    "mallorca": "RCD Mallorca",
    "osasuna": "CA Osasuna",
    "oviedo": "Real Oviedo",
    "rayo": "Rayo Vallecano",
    "real": "Real Madrid",
    "real_sociedad": "Real Sociedad",
    "sevilla": "Sevilla FC",
    "valencia": "Valencia CF",
    "villarreal": "Villarreal CF",
    "villareal": "Villarreal CF",
}


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


def load_laliga_pool(data_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(data_dir.glob("*.csv")):
        stem = p.stem.lower().strip()
        team = TEAM_NAME_MAP.get(stem, stem.replace("_", " ").title())

        df = _read_csv_safely(p)
        if "player" not in df.columns:
            continue

        for col in ["min", "goals", "assists", "xG", "xA", "xG90", "xA90"]:
            if col in df.columns:
                df[col] = _to_num(df[col])

        if "xG90" not in df.columns and "xG" in df.columns and "min" in df.columns:
            df["xG90"] = (df["xG"] / df["min"]) * 90.0
        if "xA90" not in df.columns and "xA" in df.columns and "min" in df.columns:
            df["xA90"] = (df["xA"] / df["min"]) * 90.0

        if "min" not in df.columns or "goals" not in df.columns or "assists" not in df.columns:
            continue

        df["Goals/90"] = (df["goals"] / df["min"]) * 90.0
        df["Assists/90"] = (df["assists"] / df["min"]) * 90.0

        df["Expected Goal Involvement/90"] = df["xG90"].fillna(0) + df["xA90"].fillna(0)
        df["Actual Goal Involvement/90"] = df["Goals/90"].fillna(0) + df["Assists/90"].fillna(0)

        df["team"] = team
        rows.append(df)

    if not rows:
        raise RuntimeError("No valid CSVs found. Put the .csv files in the same folder as this .py file.")
    return pd.concat(rows, ignore_index=True)


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


def _short_player_label(full_name: str) -> str:
    n = str(full_name).strip()

    low = n.lower()
    if "mbapp" in low:
        return "Mbappé"
    if "vinícius" in low or "vinicius" in low:
        return "Vinícius"
    if "bellingham" in low:
        return "Bellingham"
    if "lewandowski" in low:
        return "Lewandowski"
    if "ferrán" in low or "ferran" in low:
        return "Ferran"
    if "mikautadze" in low:
        return "Mikautadze"
    if low.strip() == "pepe" or " pepe" in low:
        return "Pepe"
    if "gerard moreno" in low:
        return "Gerard Moreno"

    tokens = n.replace("-", " ").split()
    return tokens[-1] if tokens else n


def top10_expected_vs_actual(df: pd.DataFrame, out_path: Path, min_mins: int = 450):
    d = df[df["min"] >= min_mins].copy()
    d = d.dropna(subset=["xG90", "xA90", "Goals/90", "Assists/90"])
    d["label"] = d["player"].astype(str) + " (" + d["team"].astype(str) + ")"

    d = d.sort_values("Expected Goal Involvement/90", ascending=False).head(10)
    d = d.iloc[::-1]

    y = np.arange(len(d))
    xg = d["xG90"].to_numpy()
    xa = d["xA90"].to_numpy()
    actual = d["Actual Goal Involvement/90"].to_numpy()
    expected_total = xg + xa

    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(111)

    ax.barh(y, xg, height=0.72, label="Expected Goals / 90")
    ax.barh(y, xa, left=xg, height=0.72, label="Expected Assists / 90")
    ax.scatter(actual, y, s=70, c="white", edgecolors="black", linewidths=1.2, zorder=5, label="Actual Goals+Assists / 90")

    for yi, val in zip(y, expected_total):
        ax.text(val + 0.02, yi, f"{val:.2f}", va="center", ha="left", fontsize=11, color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(d["label"].tolist(), fontsize=11)
    ax.set_xlabel("Goal involvement per 90", fontsize=12)
    ax.grid(True, axis="x", alpha=0.6)

    fig.suptitle("LaLiga’s Biggest Chance Creators & Finishers", fontsize=26, fontweight="bold", color="#f2d16b", y=0.98)
    fig.text(
        0.5, 0.85,
        f"Season to date | min {min_mins}+ mins | bars = expected, white dot = actual",
        ha="center", va="center", fontsize=13, color="#cfcfcf"
    )

    ax.legend(loc="lower right", frameon=False, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def top3_teams_top3_players(df: pd.DataFrame, out_path: Path, min_mins: int = 450):
    teams = ["Real Madrid", "FC Barcelona", "Villarreal CF"]

    d = df[df["min"] >= min_mins].copy()
    d = d.dropna(subset=["xG90", "xA90", "Goals/90", "Assists/90"])
    d["Expected Goal Involvement/90"] = d["xG90"] + d["xA90"]
    d["Actual Goal Involvement/90"] = d["Goals/90"] + d["Assists/90"]

    team_blocks = []
    for t in teams:
        td = d[d["team"] == t].sort_values("Expected Goal Involvement/90", ascending=False).head(3).copy()
        team_blocks.append(td)

    max_x = 0.0
    for td in team_blocks:
        if td.empty:
            continue
        max_x = max(
            max_x,
            float((td["xG90"] + td["xA90"]).max()),
            float(td["Actual Goal Involvement/90"].max()),
        )
    max_x = max_x * 1.12 if max_x > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    fig.suptitle(
        "LaLiga’s Biggest Chance Creators & Finishers",
        fontsize=24, fontweight="bold", color="#f2d16b", y=0.985
    )
    fig.text(
        0.5, 0.915,
        f"Top 3 players per Top 3 teams | min {min_mins}+ mins | bars = expected, white dot = actual",
        ha="center", va="center", fontsize=12, color="#cfcfcf"
    )

    for ax, t, td in zip(axes, teams, team_blocks):
        ax.set_title(t, fontsize=16, fontweight="bold", color="#f2d16b", pad=8)

        if td.empty:
            ax.text(0.5, 0.5, "No data found", ha="center", va="center", fontsize=13, color="#cfcfcf", transform=ax.transAxes)
            ax.set_xlim(0, max_x)
            ax.set_yticks([])
            ax.grid(True, axis="x", alpha=0.6)
            continue

        td = td.iloc[::-1]
        y = np.arange(len(td))

        xg = td["xG90"].to_numpy()
        xa = td["xA90"].to_numpy()
        actual = td["Actual Goal Involvement/90"].to_numpy()
        expected_total = xg + xa

        labels_short = [_short_player_label(p) for p in td["player"].astype(str).tolist()]

        ax.barh(y, xg, height=0.62, label="Expected Goals / 90")
        ax.barh(y, xa, left=xg, height=0.62, label="Expected Assists / 90")
        ax.scatter(actual, y, s=55, c="white", edgecolors="black", linewidths=1.1, zorder=5, label="Actual Goals+Assists / 90")

        for yi, val in zip(y, expected_total):
            ax.text(val + 0.02, yi, f"{val:.2f}", va="center", ha="left", fontsize=10.5, color="white")

        ax.set_yticks(y)
        ax.set_yticklabels(labels_short, fontsize=12)
        ax.set_xlim(0, max_x)
        ax.set_xlabel("Goal involvement per 90", fontsize=11)
        ax.grid(True, axis="x", alpha=0.6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, frameon=False, fontsize=11,
        bbox_to_anchor=(0.5, 0.03)
    )

    fig.tight_layout(rect=(0.02, 0.10, 1, 0.87))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    style_dark()
    base_dir = Path(__file__).resolve().parent
    out_dir = base_dir / "outputs"
    out_dir.mkdir(exist_ok=True)

    df = load_laliga_pool(base_dir)

    top10_expected_vs_actual(
        df,
        out_dir / "laliga_top10_expected_vs_actual.png",
        min_mins=450
    )

    top3_teams_top3_players(
        df,
        out_dir / "top3_teams_top3_players_expected_vs_actual.png",
        min_mins=450
    )

    print("Saved:")
    print(" - outputs/laliga_top10_expected_vs_actual.png")
    print(" - outputs/top3_teams_top3_players_expected_vs_actual.png")


if __name__ == "__main__":
    main()
