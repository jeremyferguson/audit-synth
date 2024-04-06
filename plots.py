import pandas as pd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sns.set_theme(style="whitegrid")
title_size = 50
ax_size = 36
tick_size = 30

FIGSIZE = (25, 10)
MARKERSIZE = 15
CAPSIZE = 0  # 15
CAPTHICK = 0  # 5
MARKERS = ["o", "s", "^", "X"]
COLORS = list(mcolors.TABLEAU_COLORS)
LABELPAD = 20
LINEWIDTH = 3


def create_hist(df, params_dict, param_str, style):
    plt.figure(figsize=(10, 6))
    predicates_correct = list(df["predicates_correct"])
    bins = range(min(predicates_correct), max(predicates_correct) + 2)
    if style == "stacked":
        sns.histplot(
            data=df,
            x="predicates_correct",
            hue="expected_value",
            bins=bins,
            multiple="stack",
            palette="viridis",
        )
    elif style == "facet":
        g = sns.FacetGrid(df, row="expected_value", height=6, aspect=2)
        g.map(plt.hist, "predicates_correct", bins=bins, color="skyblue", alpha=0.7)

        # Customize plot
        g.set_axis_labels("Predicates Correct", "Frequency")
        g.set_titles(row_template="Ground Truth: {row_name}")
    else:
        raise ValueError(f"Invalid histogram style: {style}")
    bins_str = "bins" if params_dict["use_bins"] else "no bins"
    mi_str = (
        f"mutual info with {params_dict['mi_pool']*params_dict['preds_per_round']} preds"
        if params_dict["use_mi"]
        else "no mutual info"
    )
    plt.title(
        f"Histogram for Parameter Combination: Thresholds {params_dict['low']} - {params_dict['high']}, "
        + f"{params_dict['num_rounds']} x {params_dict['preds_per_round']} preds,{bins_str}, depth {params_dict['depth']}, {mi_str}"
    )
    plt.xlabel("Predicates Correct")
    plt.ylabel("Frequency")
    plt.legend(title="Expected Value", labels=["False", "True"])
    plt.savefig(param_str)
    plt.close()


def create_pareto_plot(results, param_dict, param_str, scatterplot=True):
    columns = results.groupby("num_examples")
    means = []
    stds = []
    timeouts = []
    xs = []
    all_vals = []
    timeout_nos = []
    complete_nos = []
    all_xs = []
    for num_examples, column in columns:
        all_xs.append(num_examples)
        runs = column.groupby("run_no")
        min_values_run_no = []
        timeout = 0
        complete = 0
        for _, run in runs:
            run_filtered = run[run["f1_above_thresh"]]
            if run_filtered.empty:
                timeouts.append((num_examples, run["num_preds"].max()))
                timeout += 1
            else:
                complete += 1
                min_values_run_no.append(run_filtered["num_preds"].min())
                all_vals.append((num_examples, run_filtered["num_preds"].min()))
        timeout_nos.append(timeout)
        complete_nos.append(complete)
        if min_values_run_no:
            means.append(np.mean(min_values_run_no))
            stds.append(np.std(min_values_run_no))
            xs.append(num_examples)
    percentage_complete = [
        complete / (complete + timeout)
        for complete, timeout in zip(complete_nos, timeout_nos)
    ]

    print(xs)
    print(percentage_complete)
    print(means)
    all_vals = np.array(all_vals)
    timeouts_df = pd.DataFrame(columns=["col", "y"], data=timeouts)
    plt.figure(figsize=(10, 6))
    plt.xticks(range(0, 22, 2))
    plt.yticks(range(0, 80, 20))
    plt.xlim(min(all_xs) - 1, max(all_xs) + 1)
    # sns.lineplot(x=xs, y=means,color='b',label = "Successful Runs")
    if scatterplot and len(all_vals) > 0:
        sns.plot(x=all_vals[:, :1], y=all_vals[:, 1:])
    plt.errorbar(x=xs, y=means, yerr=stds, fmt="b", capsize=5, label="Successful Runs")
    # plot = sns.stripplot(data=timeouts_df,x='col',y='y',color='r',jitter=1.0,marker='^',label="Timeouts")
    plt.ylim(0, 80)
    plt.legend()
    plt.xlabel("Number of Examples")
    plt.ylabel(f"Number of Decisions")
    bins_str = "bins" if param_dict["use_bins"] else "no bins"
    mi_str = (
        f"mutual info with {param_dict['mi_pool']}k preds"
        if param_dict["use_mi"]
        else "no mutual info"
    )
    # plt.title(f"Decisions vs Number of Examples ")
    plt.grid(True)
    plt.savefig(f"{param_str}_pareto.png")
    plt.clf()
    sns.lineplot(x=all_xs, y=percentage_complete, label="Complete Runs", marker="o")
    plt.ylim(-0.05, 1.05)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    plt.xticks(range(0, 22, 2))
    # plt.title(f"Synthesis Completion Percentage vs Number of Examples")
    plt.ylabel("Fraction of Complete Runs")
    plt.xlabel("Number of Examples")
    plt.savefig(f"{param_str}_bar.png")
    plt.close()


def create_highk_plot(results, param_dict, param_str, scatterplot=True):
    columns = results.groupby("num_examples")
    means = []
    stds = []
    timeouts = []
    xs = []
    all_vals = []
    timeout_nos = []
    complete_nos = []
    all_xs = []
    for num_examples, column in columns:
        all_xs.append(num_examples)
        runs = column.groupby("run_no")
        min_values_run_no = []
        timeout = 0
        complete = 0
        for _, run in runs:
            run_filtered = run[run["f1_above_thresh"]]
            if run_filtered.empty:
                timeouts.append((num_examples, run["num_preds"].max()))
                timeout += 1
            else:
                complete += 1
                min_values_run_no.append(run_filtered["num_preds"].min())
                all_vals.append((num_examples, run_filtered["num_preds"].min()))
        timeout_nos.append(timeout)
        complete_nos.append(complete)
        if min_values_run_no:
            means.append(np.mean(min_values_run_no))
            stds.append(np.std(min_values_run_no))
            xs.append(num_examples)
    percentage_complete = [
        complete / (complete + timeout)
        for complete, timeout in zip(complete_nos, timeout_nos)
    ]
    print(xs)
    print(percentage_complete)
    print(means)
    all_vals = np.array(all_vals)
    timeouts_df = pd.DataFrame(columns=["col", "y"], data=timeouts)
    plt.figure(figsize=FIGSIZE)
    plt.xticks(range(0, 301, 20), font={"size": tick_size})
    plt.yticks(range(0, 26, 5), font={"size": tick_size})
    plt.xlim(0, 310)
    # sns.lineplot(x=xs, y=means,color='b',label = "Successful Runs")
    if scatterplot and len(all_vals) > 0:
        plt.scatter(
            all_vals[:, :1],
            all_vals[:, 1:],
            linewidth=LINEWIDTH,
            color=COLORS[0],
        )
    plt.errorbar(
        x=xs,
        y=means,
        yerr=stds,
        capsize=CAPSIZE,
        capthick=CAPTHICK,
        linewidth=LINEWIDTH,
        marker="o",
        markersize=MARKERSIZE,
        color=COLORS[0],
    )
    # plot = sns.stripplot(data=timeouts_df,x='col',y='y',color='r',jitter=1.0,marker='^',label="Timeouts")
    plt.ylim(0, 25)
    # plt.legend()
    plt.xlabel(
        "Number of Examples",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.ylabel(
        f"Number of Decisions",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    bins_str = "bins" if param_dict["use_bins"] else "no bins"
    mi_str = (
        f"mutual info with {param_dict['mi_pool']}k preds"
        if param_dict["use_mi"]
        else "no mutual info"
    )
    # plt.title(f"Decisions vs Number of Examples ", fontdict={"size": title_size})
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{param_str}_pareto.png")
    plt.clf()
    plt.figure(figsize=FIGSIZE)
    plt.plot(
        all_xs,
        percentage_complete,
        marker="o",
        markersize=MARKERSIZE,
        linewidth=LINEWIDTH,
        color=COLORS[0],
    )
    plt.ylim(-0.05, 1.05)
    plt.xlim(0, 310)
    plt.yticks([0.0, 0.25, 0.5, 0.75, 1.0], font={"size": tick_size})
    plt.xticks(range(0, 301, 20), font={"size": tick_size})
    # plt.title(
    #     f"Fraction of Runs that Reach 0.8 F1 Score", fontdict={"size": title_size}
    # )
    plt.ylabel(
        "Fraction Reaching 0.8 F1 Score",
        fontdict={"size": ax_size},
        labelpad=20,
        fontweight="bold",
    )
    plt.xlabel(
        "Number of Examples",
        fontdict={"size": ax_size},
        labelpad=20,
        fontweight="bold",
    )
    # plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{param_str}_bar.png")
    plt.close()


def make_baseline_plots(
    baseline_scores, synth_scores: pd.DataFrame, params_dict, fname
):
    grouped = synth_scores.groupby("num_preds")
    plt.figure(figsize=FIGSIZE)
    labels = []
    max_baseline_score = max(baseline_scores["f1"])
    baseline_scores["f1"] = max_baseline_score
    merged_df = pd.merge(synth_scores, baseline_scores, on="num_examples", how="left")
    merged_df["f1"] = merged_df["f1"].fillna(max_baseline_score)
    for i, (num_preds, score_group) in enumerate(grouped):
        # sns.lineplot(
        #     data=score_group,
        #     x="num_examples",
        #     y="f1_mean",
        #     color=COLORS[i],
        #     linewidth=LINEWIDTH,
        # )
        plt.errorbar(
            data=score_group,
            x="num_examples",
            y="f1_mean",
            yerr="f1_std",
            color=COLORS[i],
            label=f"Synthesis with {num_preds} decisions",
            linewidth=LINEWIDTH,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            marker=MARKERS[i],
            markersize=MARKERSIZE,
        )
    plt.plot(
        # data=merged_df,
        # x="num_examples",
        # y="f1",
        merged_df["num_examples"],
        merged_df["f1"],
        label="Baseline (LLM)",
        color="r",
        linewidth=LINEWIDTH,
    )
    labels.append("Baseline")
    plt.ylim(0, 1)
    plt.xlim(0, 31)
    plt.yticks(np.linspace(0.0, 1.0, 5), font={"size": tick_size})
    plt.xticks(range(0, 32, 5), font={"size": tick_size})
    # plt.title(f"Synthesis Performance vs Baseline", fontdict={"size": title_size})
    plt.legend(loc="lower right", fontsize=34)
    plt.ylabel(
        "F1 Score",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.xlabel(
        "Number of Examples",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def make_ablation_plots(synth_scores: pd.DataFrame, fname):
    runs = synth_scores.groupby("label")
    plt.figure(figsize=FIGSIZE)
    labels = []
    for i, (label, run) in enumerate(runs):
        # sns.lineplot(
        #     data=run,
        #     x="num_examples",
        #     y="f1_mean",
        #     color=colors[i % 4],
        #     label=label,
        #     linewidth=LINEWIDTH,
        # )
        plt.errorbar(
            data=run,
            x="num_examples",
            y="f1_mean",
            yerr="f1_std",
            # fmt=colors[i % 4],
            color=COLORS[i],
            label=label,
            linewidth=LINEWIDTH,
            marker=MARKERS[i],
            markersize=MARKERSIZE,
        )
    plt.ylim(0, 1)
    plt.xlim(-1, 21)
    plt.yticks(np.linspace(0.0, 1.0, 5), font={"size": tick_size})
    plt.xticks(range(0, 22, 2), font={"size": tick_size})
    # plt.title(
    #     f"Synthesis Performance for Ablated Parameters",
    #     fontdict={"size": title_size},
    #     y=1.04,
    # )
    plt.legend(loc="lower right", fontsize=tick_size)
    plt.ylabel(
        "F1 Score",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.xlabel(
        "Number of Examples",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def make_performance_plot(results: pd.DataFrame, fname, xs):
    plt.figure(figsize=FIGSIZE)
    means = [results[results["prog_length"] == x].iloc[0]["time_mean"] for x in xs]
    stds = [results[results["prog_length"] == x].iloc[0]["time_std"] for x in xs]
    plt.xlim(min(xs), max(xs) + 1)
    # sns.lineplot(x=xs, y=means, linewidth=LINEWIDTH, color="b")
    plt.errorbar(
        x=xs,
        y=means,
        yerr=stds,
        linewidth=LINEWIDTH,
        capsize=CAPSIZE,
        color=COLORS[0],
        capthick=CAPTHICK,
        marker="o",
        markersize=MARKERSIZE,
    )
    # plt.title(f"Execution Time vs Program Size", fontdict={"size": title_size})
    # plt.legend(loc = 'lower right')
    plt.ylabel(
        "Execution Time (seconds)",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.xlabel(
        "Program Size (# of predicates)",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.xticks(range(0, 25, 5), font={"size": tick_size})
    plt.yticks(np.arange(0, 4.1, 0.5), font={"size": tick_size})
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def make_heuristic_plots(synth_scores: pd.DataFrame, fname):
    plt.figure(figsize=FIGSIZE)
    runs = synth_scores.groupby("label")
    for i, (label, run) in enumerate(runs):
        # plt.scatter(
        #    data=run,
        #    x="num_preds",
        #    y="f1_mean",
        #    color=colors[i % 4],
        #    label=label,
        # )
        plt.errorbar(
            data=run,
            x="num_preds",
            y="f1_mean",
            yerr="f1_std",
            fmt=COLORS[i],
            label=label,
            marker=MARKERS[i],
            markersize=MARKERSIZE,
            capsize=CAPSIZE,
            capthick=CAPTHICK,
            linewidth=LINEWIDTH,
        )
    plt.ylim(0, 1)
    plt.xlim(0, 105)
    plt.yticks(
        np.linspace(0.0, 1.0, 5),
        font={"size": tick_size},
    )
    plt.xticks(
        range(0, 110, 10),
        font={"size": tick_size},
    )
    # plt.title(
    #     f"Synthesized Program Performance, by Predicate Selection Approach",
    #     fontdict={"size": title_size},
    #     y=1.04,
    # )
    plt.legend(
        fontsize=30,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.53),
    )
    plt.ylabel(
        "F1 Score",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.xlabel(
        "Number of Decisions",
        fontdict={"size": ax_size},
        fontweight="bold",
        labelpad=LABELPAD,
    )
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
