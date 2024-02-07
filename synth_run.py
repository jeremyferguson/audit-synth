import itertools
import csv
import numpy as np
import seaborn as sns
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from synth import App, AppConfig

# Define fixed values
output_threshold_value = 0.5
features_fname_value = "extracted_features_detr_500.json"
#examples_fname_value = "partial_labeled_coco.csv"
all_examples_csv_value = "partial_labeled_sports.csv"
prog_name_value = "prog_sports.txt"
manual_value = False
eval_value = True
debug_value = False
samples_per_point = 10

# Define lists of parameters
low_threshold_values = [0.1]
high_threshold_values = [0.9]
num_feature_selection_rounds_values = range(1,6)
predicates_per_round_values = [1,5,10,15,20,25,30]
use_bins_values = [True]
depth_values = [1]
use_mutual_information_values = [True]
mutual_info_pool_size_values = [2]
num_examples_values = range(1,21)

#overall parameters
full_csv_filename = 'synth_results_full.csv'
num_samples = 10
f1_thresh = 0.8
pred_threshold = 0.1
img_dir = "plots"
hist_dir = "hists"
pareto_dir = "pareto"
csv_dir = "csv_out"
hist_type = "stacked"


# Generate all combinations of parameters
parameter_combinations = itertools.product(
    low_threshold_values,
    high_threshold_values,
    #num_feature_selection_rounds_values,
    #predicates_per_round_values,
    use_bins_values,
    depth_values,
    use_mutual_information_values,
    mutual_info_pool_size_values,
    num_examples_values
)
pred_size_combinations = list(itertools.product(num_feature_selection_rounds_values,predicates_per_round_values))
pred_size_combinations.sort(key=lambda x: x[0]*x[1])
# Header for the CSV file
ind_csv_header = [
    'low_threshold', 'high_threshold', 'num_feature_selection_rounds',
    'predicates_per_round', 'use_bins', 'depth', 'use_mutual_information',
    'mutual_info_pool_size', 'num_examples', 'filename', 'expected_value',
    'predicates_correct', 'total_predicates', 'run_number','program'
]

full_csv_header = [
    'low_threshold', 'high_threshold', 'num_feature_selection_rounds',
    'predicates_per_round', 'use_bins', 'depth', 'use_mutual_information',
    'mutual_info_pool_size', 'num_examples', 'precision','recall','program','run_no','f1','f1_above_thresh','subset'
]
def run_synth(params_dict):
    config = AppConfig(
                    low_threshold=params_dict['low'],
                    high_threshold=params_dict['high'],
                    num_feature_selection_rounds=params_dict['num_rounds'],
                    predicates_per_round=params_dict['preds_per_round'],
                    use_bins=params_dict['use_bins'],
                    depth=params_dict['depth'],
                    use_mutual_information=params_dict['use_mi'],
                    mutual_info_pool_size=params_dict['mi_pool'],
                    features_fname=features_fname_value,
                    num_examples=params_dict['num_examples'],
                    full_csv=all_examples_csv_value,
                    manual=manual_value,
                    eval=eval_value,
                    debug=debug_value,
                    prog_fname=prog_name_value,
                    examples=params_dict['examples']
                )
    # Run the app and get evaluation results (results is now an array of tuples)
    app = App(config)
    prog = app.run()
    results = app.eval()
    precision, recall = compute_metrics(results)
    return results, (precision, recall), prog

def run():
    all_examples = pd.read_csv(all_examples_csv_value)
    all_example_fnames = list(all_examples[all_examples['val']==True]['fname'])
    full_results = []
    # Iterate through parameter combinations
    for params in parameter_combinations:
        print("Params: ",params)
        ind_results = []
        params_dict = {'low':params[0],'high':params[1],
                            'use_bins':params[2],'depth':params[3],'use_mi':params[4],'mi_pool':params[5],'num_examples':params[6],
                            }
        param_str = f"{params_dict['low']}-{params_dict['high']}_" + \
                f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}" + \
                f"mi_{params_dict['mi_pool']}midepth_{params_dict['num_examples']}_examples"
        ind_csv_filename = f"{param_str}_results.csv" 
        for run_no in range(num_samples):
            print("Run: ",run_no)
            user_examples = set(random.choices(all_example_fnames,k=params[6]))
            params_dict['examples'] = user_examples
            for num_rounds, predicates_per_round in pred_size_combinations:
                print("Predicates: ",num_rounds * predicates_per_round)
                params_dict['num_rounds'] = num_rounds
                params_dict['preds_per_round'] = predicates_per_round
                results, (precision, recall), prog = run_synth(params_dict)
                if precision + recall < 1e-5:
                    f1 = 0
                else:
                    f1 = (2*precision*recall) / (precision + recall)
                f1_above_thresh = f1 > f1_thresh
                full_results.append([params_dict['low'],params_dict['high'],params_dict['num_rounds'],params_dict['preds_per_round'],
                                     params_dict['use_bins'],params_dict['depth'],params_dict['use_mi'],params_dict['mi_pool'],
                                     params_dict['num_examples'],precision,recall,prog,run_no,f1,f1_above_thresh,user_examples])
                for result in results:
                    fname = result[0]
                    expected_val = result[1]
                    preds_correct = result[2]
                    preds_total = result[3]
                    ind_results.append([params_dict['low'],params_dict['high'],params_dict['num_rounds'],params_dict['preds_per_round'],
                                     params_dict['use_bins'],params_dict['depth'],params_dict['use_mi'],params_dict['mi_pool'],
                                     params_dict['num_examples'],fname,expected_val,preds_correct,preds_total,run_no,prog])
                hist_param_str = param_str + f"_{params_dict['num_rounds']}x{params_dict['preds_per_round']}_{run_no}"
                if hist_type == "stacked":
                    create_stacked_histogram(params_dict,results,hist_param_str)
                elif hist_type == "facet":
                    create_facet_histogram(params_dict,results,hist_param_str)
                else:
                    raise ValueError(f"Invalid histogram type: {hist_type}")
                
                if f1_above_thresh:
                    break
        df = pd.DataFrame(columns = ind_csv_header,data=ind_results)
        df.to_csv(f"{csv_dir}/{ind_csv_filename}")
    df = pd.DataFrame(columns = full_csv_header,data=full_results)
    df.to_csv(f"{csv_dir}/{full_csv_filename}")
    param_str = f"{params_dict['low']}-{params_dict['high']}_" + \
                f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}" + \
                f"mi_{params_dict['mi_pool']}midepth_{params_dict['num_examples']}_examples"
    create_pareto_plots(full_results,params_dict,param_str)
    print(f"Results written to {csv_dir}/{full_csv_filename}")

def compute_metrics(results):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    for row in results:
        if row[3] == 0:
            correct_fraction = 0
        else:
            correct_fraction = row[2] / row[3]
        if row[1] and correct_fraction > pred_threshold:
            true_pos += 1
        elif row[1] and correct_fraction <= pred_threshold:
            false_neg += 1
        elif not row[1] and correct_fraction > pred_threshold:
            false_pos += 1
        elif not row[1] and correct_fraction <= pred_threshold:
            true_neg += 1
    if true_pos + false_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos + false_neg == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)
    return precision, recall

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

def create_pareto_plots(full_results,params_dict,param_str):
        # Convert the array to a pandas DataFrame
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

            means.append(np.mean(min_values_run_no))
            stds.append(np.std(min_values_run_no))

        plt.figure(figsize=(10, 6))
        sns.lineplot(x=examples, y=means)
        plt.errorbar(x=examples, y=means, yerr=stds, fmt='o', capsize=5)
        plt.xlabel('Number of examples')
        plt.ylabel(f'Predicates to reach {f1_thresh} F1 score')
        
        bins_str = "bins" if params_dict['use_bins'] else "no bins"
        mi_str = f"mutual info with {params_dict['mi_pool']*params_dict['preds_per_round']} preds" if params_dict['use_mi'] else "no mutual info"
        plt.title(f"Predicates vs examples {params_dict['low']} - {params_dict['high']}, " + \
              f"{params_dict['num_rounds']} x {params_dict['preds_per_round']} preds,{bins_str}, depth {params_dict['depth']}, {mi_str}")
        plt.grid(True)
        plt.savefig(f"{img_dir}/{pareto_dir}/{param_str}_{f1_thresh}_pareto.png")
        plt.close()

if __name__ == "__main__":
    run()