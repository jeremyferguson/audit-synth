import pandas as pd
import maptlotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_facet_histogram(params_dict, results,param_str):
    # Extract relevant data from results
    filenames, expected_values, predicates_correct, total_predicates = zip(*results)

    # Convert expected_values to numeric for coloring
    expected_values_numeric = [1 if val else 0 for val in expected_values]

    # Combine data into a DataFrame
    data = {
        'Predicates Correct': predicates_correct,
        'Expected Value': expected_values,
    }
    df = pd.DataFrame(data)

    # Create a facet grid with histograms
    g = sns.FacetGrid(df, row='Expected Value', height=6, aspect=2)
    g.map(plt.hist, 'Predicates Correct', bins=range(min(predicates_correct), max(predicates_correct) + 2), color="skyblue", alpha=0.7)

    # Customize plot
    g.set_axis_labels("Predicates Correct", "Frequency")
    g.set_titles(row_template="Ground Truth: {row_name}")
    bins_str = "bins" if params_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {params_dict['mi_pool']*params_dict['preds_per_round']} preds" if params_dict['use_mi'] else "no mutual info"
    
    plt.title(f"Histogram for Parameter Combination: Thresholds {params_dict['low']} - {params_dict['high']}, " + \
              f"{params_dict['num_rounds']} x {params_dict['preds_per_round']} preds,{bins_str}, depth {params_dict['depth']}, {mi_str}")
    # Save the figure to a file (change the filename as needed)
    g.savefig(f"{img_dir}/{hist_dir}/facet_histogram_{param_str}.png")
    plt.close()

def create_stacked_histogram(params_dict, results,param_str):
    plt.figure(figsize=(10, 6))

    # Extract relevant data from results
    filenames, expected_values, predicates_correct, total_predicates = zip(*results)

    # Combine data into a DataFrame
    data = {
        'Predicates Correct': predicates_correct,
        'Expected Value': expected_values,
    }
    df = pd.DataFrame(data)
    # Create stacked histogram
    sns.histplot(data=df, x='Predicates Correct', hue='Expected Value',
                 bins=range(min(predicates_correct), max(predicates_correct) + 2),
                 multiple="stack", palette="viridis")

    # Customize plot
    bins_str = "bins" if params_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {params_dict['mi_pool']*params_dict['preds_per_round']} preds" if params_dict['use_mi'] else "no mutual info"
    plt.title(f"Histogram for Parameter Combination: Thresholds {params_dict['low']} - {params_dict['high']}, " + \
              f"{params_dict['num_rounds']} x {params_dict['preds_per_round']} preds,{bins_str}, depth {params_dict['depth']}, {mi_str}")
    plt.xlabel("Predicates Correct")
    plt.ylabel("Frequency")
    plt.legend(title="Expected Value", labels=["False", "True"])
    plt.savefig(f"{img_dir}/{hist_dir}/histogram_{param_str}.png")
    plt.close()

def create_pareto_plots(full_results,scatterplot=False):
    df = pd.DataFrame(full_results, columns=['low_threshold', 'high_threshold', 'num_feature_selection_rounds',
                                     'predicates_per_round', 'use_bins', 'depth', 'use_mutual_information',
                                     'mutual_info_pool_size', 'num_examples', 'precision', 'recall', 'program',
                                     'run_no', 'f1', 'f1_above_thresh', 'subset'])

    df['num_preds'] = df['num_feature_selection_rounds'] * df['predicates_per_round']

    grouped = df.groupby(['low_threshold', 'high_threshold', 'use_bins', 'depth',
                          'use_mutual_information', 'mutual_info_pool_size'])

    for name, group in grouped:
        nested_grouped = group.groupby('num_examples')
        examples = []
        means = []
        stds = []
        mins = []
        for num_examples, nested_group in nested_grouped:
            examples.append(num_examples)
            nested_group_run_no = nested_group.groupby('run_no')
            
            min_values_run_no = []

            for run_no, nested_group_run in nested_group_run_no:
                nested_group_filtered = nested_group_run[nested_group_run['f1_above_thresh']]

                if nested_group_filtered.empty:
                    min_values_run_no.append(nested_group_run['num_preds'].max())
                else:
                    min_values_run_no.append(nested_group_filtered['num_preds'].min())
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
        plt.ylabel(f'Programmer decisions to reach {f1_thresh} F1 score')
        print(group)
        fname_str = f"{group['low_threshold'].iloc[0]}-{group['high_threshold'].iloc[0]}_" + \
                f"preds_{group['use_bins'].iloc[0]}bins_{group['depth'].iloc[0]}depth_{group['use_mutual_information'].iloc[0]}" + \
                f"mi_{group['mutual_info_pool_size'].iloc[0]}midepth"
        bins_str = "bins" if group['use_bins'].iloc[0] else "no bins"
        mi_str = f"mutual info with {group['mutual_info_pool_size'].iloc[0]}k preds" if group['use_mutual_information'].iloc[0] else "no mutual info"
        plt.title(f"Programmer decisions vs examples {group['low_threshold'].iloc[0]} - {group['high_threshold'].iloc[0]}, " + \
              f"{bins_str}, depth {group['depth'].iloc[0]}, {mi_str}")
        plt.grid(True)
        plt.savefig(f"{img_dir}/{pareto_dir}/{fname_str}_{f1_thresh}_{num_samples}_pareto.png")
        plt.close()

def make_baseline_plots(baseline_scores,synth_scores):
    plt.plot(range(1,synth_scores.shape[0]+1), np.mean(synth_scores,axis=1), label='Synth')
    plt.plot(range(len(baseline_scores)), baseline_scores, label='Baseline')
    plt.xlabel('Number of examples')
    plt.errorbar(range(1,synth_scores.shape[0]+1), y=np.mean(synth_scores,axis=1), yerr=np.std(synth_scores,axis=1), fmt='o', capsize=5)
    plt.ylabel('Max F1 Score')
    plt.title('Max F1 Scores for Baseline vs Synth')
    plt.legend()
    plt.savefig(f"{img_dir}/{baseline_dir}/{baseline_plot_fname}")
