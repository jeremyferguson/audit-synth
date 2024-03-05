import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def create_hist(df, params_dict, param_str,style):
    plt.figure(figsize=(10, 6))
    predicates_correct = list(df['predicates_correct'])
    bins=range(min(predicates_correct), max(predicates_correct) + 2)
    if style == "stacked":
        sns.histplot(data=df, x='predicates_correct', hue='expected_value',
                bins=bins,
                multiple="stack", palette="viridis")
    elif style == 'facet':
        g = sns.FacetGrid(df, row='expected_value', height=6, aspect=2)
        g.map(plt.hist, 'predicates_correct', bins=bins, color="skyblue", alpha=0.7)

        # Customize plot
        g.set_axis_labels("Predicates Correct", "Frequency")
        g.set_titles(row_template="Ground Truth: {row_name}")
    else:
        raise ValueError(f"Invalid histogram style: {style}")
    bins_str = "bins" if params_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {params_dict['mi_pool']*params_dict['preds_per_round']} preds" if params_dict['use_mi'] else "no mutual info"
    plt.title(f"Histogram for Parameter Combination: Thresholds {params_dict['low']} - {params_dict['high']}, " + \
              f"{params_dict['num_rounds']} x {params_dict['preds_per_round']} preds,{bins_str}, depth {params_dict['depth']}, {mi_str}")
    plt.xlabel("Predicates Correct")
    plt.ylabel("Frequency")
    plt.legend(title="Expected Value", labels=["False", "True"])
    plt.savefig(param_str)
    plt.close()

def create_pareto_plot(results,param_dict,param_str,scatterplot=False):
    columns = results.groupby('num_examples')
    examples = []
    means = []
    stds = []
    mins = []
    for num_examples, column in columns:
        examples.append(num_examples)
        runs = column.groupby('run_no')
        min_values_run_no = []

        for _, run in runs:
            run_filtered = run[run['f1_above_thresh']]
            if run_filtered.empty:
                min_values_run_no.append(run['num_preds'].max())
            else:
                min_values_run_no.append(run_filtered['num_preds'].min())
        coords = [(num_examples, val) for val in min_values_run_no]
        mins.extend(coords)
        means.append(np.mean(min_values_run_no))
        stds.append(np.std(min_values_run_no))
        xs = [pair[0] for pair in mins]
        ys = [pair[1] for pair in mins]
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=examples, y=means)
    plt.errorbar(x=examples, y=means, yerr=stds, fmt='o', capsize=5)
    if scatterplot:
        plt.scatter(xs,ys)
    plt.xlabel('Number of examples')
    plt.ylabel(f'Programmer decisions to reach {param_dict["f1_thresh"]} F1 score')
    bins_str = "bins" if param_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {param_dict['mi_pool']}k preds" if param_dict['use_mi'] else "no mutual info"
    plt.title(f"Programmer decisions vs examples {param_dict['low']} - {param_dict['high']}, " + \
            f"{bins_str}, depth {param_dict['depth']}, {mi_str}")
    plt.grid(True)
    plt.savefig(f"{param_str}_pareto.png")
    plt.close()

def make_baseline_plots(baseline_scores,synth_scores,params_dict,fname):
    plt.plot(range(1,synth_scores.shape[0]+1), np.mean(synth_scores,axis=1), label='Synth')
    plt.plot(range(len(baseline_scores)), baseline_scores, label='Baseline')
    plt.xlabel('Number of examples')
    plt.errorbar(range(1,synth_scores.shape[0]+1), y=np.mean(synth_scores,axis=1), yerr=np.std(synth_scores,axis=1), fmt='o', capsize=5)
    bins_str = "bins" if params_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {params_dict['mi_pool']}k preds" if params_dict['use_mi'] else "no mutual info"
    plt.title(f"Max F1 Scores for Baseline vs Synth: {params_dict['low']} - {params_dict['high']}, " + \
            f"{bins_str}, depth {params_dict['depth']}, {mi_str}")
    plt.ylabel('Max F1 Score')
    plt.legend()
    plt.savefig(fname)




