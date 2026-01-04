from pathlib import Path
import unicodedata

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mplsoccer import PyPizza
from scipy.stats import percentileofscore


MIN_MINUTES = 450
DATA_DIR = Path(".")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PLAYERS = [
    "Pedri",
    "Jude Bellingham",
    "Raphinha",
    "Vinícius Júnior",
    "Kylian Mbappé",
    "Ferran Torres",
]

PIZZA_METRICS = {
    "xg90": "xG / 90",
    "xa90": "xA / 90",
    "sp90m": "Shots / 90",
    "kp90": "Key Passes / 90",
    "g90": "Goals / 90",
    "g_minus_xg90": "(G-xG) / 90",
}

THEME = {
    "bg": "#111111",
    "text": "#ffffff",
    "muted": "#bbbbbb",
    "accent": "#3b82f6",
    "highlight": "#a56eff",
    "gold": "#f4d35e",
}

DISPLAY_NAME_OVERRIDES = {
    "kylian mbappe-lottin": "Kylian Mbappé",
    "kylian mbappe": "Kylian Mbappé",
    "kylian mbappé": "Kylian Mbappé",
}

VALUE_RADIUS_FACTOR = 0.88


def norm_name(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s


def safe_filename(name: str) -> str:
    return norm_name(name).replace(" ", "_")


def pct_int(series: pd.Series, value: float) -> int:
    p = percentileofscore(series.dropna(), value, kind="rank")
    return int(round(p))


def set_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": THEME["bg"],
        "axes.facecolor": THEME["bg"],
        "savefig.facecolor": THEME["bg"],
        "text.color": THEME["text"],
        "axes.labelcolor": THEME["text"],
        "xtick.color": THEME["muted"],
        "ytick.color": THEME["muted"],
    })


def read_understat_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", engine="python", quotechar='"')
    df.columns = [str(c).replace('"', "").strip().lower() for c in df.columns]
    df["team"] = path.stem
    return df


def load_league(data_dir: Path) -> pd.DataFrame:
    csv_files = sorted([p for p in data_dir.glob("*.csv") if p.is_file()])
    if not csv_files:
        raise FileNotFoundError("No .csv files found in the project folder.")

    dfs = [read_understat_csv(p) for p in csv_files]
    league = pd.concat(dfs, ignore_index=True)

    required = ["player", "min", "goals", "xg", "xa", "sp90m", "kp90", "xg90", "xa90"]
    missing = [c for c in required if c not in league.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}. Found: {league.columns.tolist()}")

    for c in ["min", "goals", "xg", "xa", "sp90m", "kp90", "xg90", "xa90"]:
        league[c] = pd.to_numeric(league[c], errors="coerce")

    league = league.dropna(subset=["player", "min"])
    league = league[league["min"] >= MIN_MINUTES].copy()

    league["g90"] = league["goals"] * 90 / league["min"]
    league["g_minus_xg90"] = (league["goals"] - league["xg"]) * 90 / league["min"]
    league["player_norm"] = league["player"].apply(norm_name)

    return league


def find_player_row(df: pd.DataFrame, player_name: str) -> pd.Series | None:
    target = norm_name(player_name)
    hits = df[df["player_norm"] == target]

    if hits.empty and "mbappe" in target:
        hits = df[df["player_norm"].isin(["kylian mbappe", "kylian mbappé", "kylian mbappe-lottin"])]

    if hits.empty:
        hits = df[df["player_norm"].str.contains(target, na=False)]

    if hits.empty:
        return None

    return hits.sort_values("min", ascending=False).iloc[0]


def build_ref(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(PIZZA_METRICS.keys())
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing derived metric columns: {missing}")
    return df[cols].copy()


def display_name(raw_name: str) -> str:
    n = norm_name(raw_name)
    return DISPLAY_NAME_OVERRIDES.get(n, raw_name)


def nudge_value_boxes(ax) -> None:
    target_fc = mcolors.to_rgba(THEME["highlight"])
    for t in ax.texts:
        patch = t.get_bbox_patch()
        if patch is None:
            continue
        fc = patch.get_facecolor()
        if tuple(round(x, 3) for x in fc) != tuple(round(x, 3) for x in target_fc):
            continue
        x, y = t.get_position()
        t.set_position((x, y * VALUE_RADIUS_FACTOR))


def draw_single_pizza(player_row: pd.Series, ref: pd.DataFrame, subtitle: str, save_path: Path) -> None:
    values = [pct_int(ref[m], player_row[m]) for m in PIZZA_METRICS.keys()]

    pizza = PyPizza(
        params=list(PIZZA_METRICS.values()),
        background_color=THEME["bg"],
        straight_line_color=THEME["muted"],
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    fig, ax = pizza.make_pizza(
        values,
        figsize=(7, 7),
        slice_colors=[THEME["accent"]] * len(values),
        value_colors=[THEME["text"]] * len(values),
        value_bck_colors=[THEME["highlight"]] * len(values),
        blank_alpha=0.35,
        kwargs_slices=dict(edgecolor=THEME["text"], linewidth=1),
        kwargs_params=dict(color=THEME["text"], fontsize=10),
        kwargs_values=dict(
            color=THEME["text"],
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.22", fc=THEME["highlight"], ec="none", alpha=0.92),
        ),
    )

    nudge_value_boxes(ax)

    fig.text(0.5, 0.975, display_name(player_row["player"]), ha="center",
             fontsize=18, color=THEME["gold"], weight="bold")
    fig.text(0.5, 0.935, subtitle, ha="center", fontsize=11, color=THEME["muted"])

    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_compare(p1: str, p2: str, df: pd.DataFrame, ref: pd.DataFrame, subtitle: str, out_path: Path) -> None:
    r1 = find_player_row(df, p1)
    r2 = find_player_row(df, p2)
    if r1 is None or r2 is None:
        print(f"Missing one of: {p1}, {p2}")
        return

    v1 = [pct_int(ref[m], r1[m]) for m in PIZZA_METRICS.keys()]
    v2 = [pct_int(ref[m], r2[m]) for m in PIZZA_METRICS.keys()]

    pizza = PyPizza(
        params=list(PIZZA_METRICS.values()),
        background_color=THEME["bg"],
        straight_line_color=THEME["muted"],
        straight_line_lw=1,
        last_circle_lw=1,
        other_circle_lw=0,
        inner_circle_size=20,
    )

    fig = plt.figure(figsize=(14, 7), facecolor=THEME["bg"])
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    pizza.make_pizza(
        v1, ax=ax1,
        slice_colors=[THEME["accent"]] * len(v1),
        value_colors=[THEME["text"]] * len(v1),
        value_bck_colors=[THEME["highlight"]] * len(v1),
        blank_alpha=0.35,
        kwargs_slices=dict(edgecolor=THEME["text"], linewidth=1),
        kwargs_params=dict(color=THEME["text"], fontsize=10),
        kwargs_values=dict(
            color=THEME["text"],
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.22", fc=THEME["highlight"], ec="none", alpha=0.92),
        ),
    )

    pizza.make_pizza(
        v2, ax=ax2,
        slice_colors=[THEME["accent"]] * len(v2),
        value_colors=[THEME["text"]] * len(v2),
        value_bck_colors=[THEME["highlight"]] * len(v2),
        blank_alpha=0.35,
        kwargs_slices=dict(edgecolor=THEME["text"], linewidth=1),
        kwargs_params=dict(color=THEME["text"], fontsize=10),
        kwargs_values=dict(
            color=THEME["text"],
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.22", fc=THEME["highlight"], ec="none", alpha=0.92),
        ),
    )

    nudge_value_boxes(ax1)
    nudge_value_boxes(ax2)

    fig.text(0.25, 0.965, display_name(r1["player"]), ha="center",
             fontsize=18, color=THEME["gold"], weight="bold")
    fig.text(0.75, 0.965, display_name(r2["player"]), ha="center",
             fontsize=18, color=THEME["gold"], weight="bold")
    fig.text(0.5, 0.93, subtitle, ha="center", fontsize=11, color=THEME["muted"])

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    set_theme()
    league = load_league(DATA_DIR)
    ref = build_ref(league)

    subtitle = f"Percentiles vs LaLiga pool | min {MIN_MINUTES} mins"

    for player in PLAYERS:
        row = find_player_row(league, player)
        if row is None:
            print(f"Not found: {player}")
            continue
        out = OUT_DIR / f"pizza_{safe_filename(display_name(row['player']))}.png"
        draw_single_pizza(row, ref, subtitle, out)
        print(f"Saved {out}")

    draw_compare("Pedri", "Jude Bellingham", league, ref, subtitle, OUT_DIR / "compare_pedri_vs_bellingham.png")
    draw_compare("Vinícius Júnior", "Kylian Mbappé", league, ref, subtitle, OUT_DIR / "compare_vini_vs_mbappe.png")
    print("DONE")


if __name__ == "__main__":
    main()
