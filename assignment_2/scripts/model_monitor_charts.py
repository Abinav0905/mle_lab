import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.chdir(PROJECT_DIR)

import matplotlib
matplotlib.use("Agg")  # no screen inside the container
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import utils.model_utils as mu

# to call this script: python scripts/model_monitor_charts.py --snapshotdate "2024-12-01"
# Draws the monitoring dashboard charts from the model_monitoring gold table (all months up to the snapshot date).

SURFACE = "#fcfcfb"
INK = "#1f2430"
INK_SECONDARY = "#52514e"
GRID = "#e4e2dc"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # one colour per model version, in training order
STATUS = {"GREEN": "#0ca30c", "AMBER": "#fab219", "RED": "#d03b3b"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def model_colours(versions):
    ordered = sorted(v for v in set(versions) if isinstance(v, str))
    return {v: SERIES[i % len(SERIES)] for i, v in enumerate(ordered)}


def short_name(version):
    # credit_model_2024_08_01 -> "model trained 2024-08"
    return "model trained " + version.replace("credit_model_", "")[:7].replace("_", "-")


def new_chart(title, subtitle):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.14)
    fig.text(0.08, 0.93, title, fontsize=15, fontweight="bold", color=INK)
    fig.text(0.08, 0.875, subtitle, fontsize=11, color=INK_SECONDARY)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    return fig, ax


def line_label(ax, y, label):
    # label sits just above its line, at the right edge of the chart
    ax.annotate(label, xy=(1.0, y), xycoords=("axes fraction", "data"), xytext=(-4, 3), textcoords="offset points",
                ha="right", va="bottom", fontsize=10, color=INK_SECONDARY)


def threshold_line(ax, y, label, status, x_right=None):
    ax.axhline(y, color=STATUS[status], linewidth=1.5, linestyle="--", zorder=1)
    line_label(ax, y, label)


def plot_by_model(ax, df, x, y, colours, label_points=False, fmt="{:.2f}"):
    for version, part in df.groupby("version_for_colour"):
        part = part.sort_values(x)
        ax.plot(part[x], part[y], color=colours[version], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=short_name(version), zorder=3)
        if label_points:
            for _, r in part.iterrows():
                ax.annotate(fmt.format(r[y]), (r[x], r[y]), textcoords="offset points", xytext=(0, 10),
                            ha="center", fontsize=9.5, color=INK_SECONDARY)


def save(fig, name):
    os.makedirs(mu.CHARTS_DIR, exist_ok=True)
    path = os.path.join(mu.CHARTS_DIR, name)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("saved chart:", path)


def chart_score_stability(m, colours):
    df = m.assign(version_for_colour=m["model_version"], month=pd.to_datetime(m["snapshot_date"]))
    fig, ax = new_chart("Score stability: PSI of this month's scores vs the model's reference",
                        "Below 0.10 = stable  ·  0.10 to 0.25 = watch  ·  above 0.25 = act")
    plot_by_model(ax, df, "month", "score_psi", colours)
    x_right = df["month"].max() + pd.Timedelta(days=12)
    threshold_line(ax, mu.PSI_AMBER, "0.10 watch", "AMBER", x_right)
    threshold_line(ax, mu.PSI_RED, "0.25 act", "RED", x_right)
    for _, r in df[df["stability_status"] == "RED"].iterrows():
        ax.annotate("RED: " + str(r["n_features_mostly_missing"]) + " model inputs missing\n(" + r["max_feature_psi_name"] + " PSI " +
                    "{:.1f}".format(r["max_feature_psi"]) + ")", (r["month"], r["score_psi"]),
                    textcoords="offset points", xytext=(-14, -4), ha="right", va="top", fontsize=10, color=INK)
    first_month = df.groupby("model_version")["month"].min().sort_index()
    for version, month in list(first_month.items())[1:]:
        r = df[df["month"] == month].iloc[0]
        ax.annotate("Retrained: " + short_name(version), (month, r["score_psi"]), textcoords="offset points",
                    xytext=(0, 34), ha="center", fontsize=10, color=INK,
                    arrowprops=dict(arrowstyle="-", color=INK_SECONDARY, linewidth=1))
    ax.set_ylabel("Population Stability Index (PSI)")
    ax.set_ylim(0, max(0.35, df["score_psi"].max() * 1.15))
    ax.set_xlim(df["month"].min() - pd.Timedelta(days=20), df["month"].max() + pd.Timedelta(days=20))
    ax.legend(loc="upper left")
    save(fig, "1_score_stability_psi.png")


def chart_mean_score(m, colours):
    df = m.assign(version_for_colour=m["model_version"], month=pd.to_datetime(m["snapshot_date"]))
    fig, ax = new_chart("Average predicted default probability of new applicants, by month",
                        "A sudden jump with no change in the applicants usually means an input problem, not real risk")
    plot_by_model(ax, df, "month", "mean_predicted_probability", colours, label_points=True, fmt="{:.0%}")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "{:.0%}".format(v)))
    ax.set_ylabel("Mean predicted probability")
    ax.set_ylim(0, max(0.45, df["mean_predicted_probability"].max() * 1.2))
    ax.set_xlim(df["month"].min() - pd.Timedelta(days=20), df["month"].max() + pd.Timedelta(days=20))
    ax.legend(loc="lower left")
    save(fig, "2_mean_predicted_probability.png")


def chart_performance(m, colours):
    perf = m.dropna(subset=["gini"]).copy()
    fig, ax = new_chart("Model performance on matured loans (Gini, by application month)",
                        "Outcome known 6 months after application  ·  dashed lines = the model's out-of-time Gini, minus 0.05 (watch) and 0.10 (act)")
    if perf.empty:
        ax.text(0.5, 0.5, "No matured outcomes yet", transform=ax.transAxes, ha="center", fontsize=13, color=INK_SECONDARY)
        save(fig, "3_performance_gini.png")
        return
    perf["version_for_colour"] = perf["performance_model_version"]
    perf["cohort"] = pd.to_datetime(perf["performance_cohort_date"])
    plot_by_model(ax, perf, "cohort", "gini", colours, label_points=True)
    x_right = perf["cohort"].max() + pd.Timedelta(days=12)
    for version, part in perf.groupby("performance_model_version"):
        ref = part["reference_gini_oot"].iloc[0]
        ax.axhline(ref, color=colours[version], linewidth=1.5, linestyle=":", zorder=1)
        line_label(ax, ref, "expected (out-of-time Gini " + "{:.2f}".format(ref) + ")")
        threshold_line(ax, ref - mu.GINI_DROP_AMBER, "watch (-0.05)", "AMBER")
        threshold_line(ax, ref - mu.GINI_DROP_RED, "act (-0.10)", "RED")
    ax.set_ylabel("Gini (2 x AUC - 1)")
    ax.set_ylim(0.4, 0.8)
    ax.set_xlim(perf["cohort"].min() - pd.Timedelta(days=20), perf["cohort"].max() + pd.Timedelta(days=60))
    ax.legend(loc="lower left")
    save(fig, "3_performance_gini.png")


def chart_calibration(m):
    perf = m.dropna(subset=["actual_default_rate"]).copy()
    fig, ax = new_chart("Predicted vs actual default rate (by application month)",
                        "Average predicted probability compared with the share of loans that actually defaulted")
    if perf.empty:
        ax.text(0.5, 0.5, "No matured outcomes yet", transform=ax.transAxes, ha="center", fontsize=13, color=INK_SECONDARY)
        save(fig, "4_predicted_vs_actual.png")
        return
    perf["cohort"] = pd.to_datetime(perf["performance_cohort_date"])
    perf = perf.sort_values("cohort")
    for col, colour, label in [("actual_default_rate", "#2a78d6", "Actual default rate"),
                               ("cohort_mean_predicted_probability", "#eb6834", "Mean predicted probability")]:
        ax.plot(perf["cohort"], perf[col], color=colour, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
        last = perf.iloc[-1]
        ax.annotate(label + " " + "{:.0%}".format(last[col]), (last["cohort"], last[col]), textcoords="offset points",
                    xytext=(10, 0), va="center", fontsize=10, color=INK_SECONDARY)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "{:.0%}".format(v)))
    ax.set_ylim(0, 0.5)
    ax.set_xlim(perf["cohort"].min() - pd.Timedelta(days=20), perf["cohort"].max() + pd.Timedelta(days=110))
    ax.set_ylabel("Default rate")
    ax.legend(loc="lower left")
    save(fig, "4_predicted_vs_actual.png")


def main(snapshotdate):
    print("\n\n---starting job---\n\n")
    snapshot_date = mu.to_date(snapshotdate)
    m = mu.read_partitions(mu.MONITORING_DIR, "model_monitoring_", end=snapshot_date)
    if m.empty:
        print("no monitoring results up to", snapshot_date, "- skipping charts")
        sys.exit(mu.SKIP_EXIT_CODE)

    colours = model_colours(list(m["model_version"]) + list(m["performance_model_version"].dropna()))
    chart_score_stability(m, colours)
    chart_mean_score(m, colours)
    chart_performance(m, colours)
    chart_calibration(m)

    # also export the monitoring table as CSV, easy to open in Excel
    m.sort_values("snapshot_date").to_csv(os.path.join(mu.CHARTS_DIR, "model_monitoring_summary.csv"), index=False)
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="draw model monitoring charts")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
