import itertools
import csv
import seaborn as sns
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from synth import App, AppConfig

# Define fixed values
output_threshold_value = 0.5
features_fname_value = "extracted_features_detr_500.json"
#examples_fname_value = "partial_labeled_coco.csv"
full_csv_value = "partial_labeled_sports.csv"
prog_name_value = "prog_sports.txt"
manual_value = False
eval_value = True
debug_value = False
samples_per_point = 10

# Define lists of parameters
low_threshold_values = [0.1]
high_threshold_values = [0.9]
num_feature_selection_rounds_values = range(1,10)
predicates_per_round_values = range(1,30,5)
use_bins_values = [True]
depth_values = [1]
use_mutual_information_values = [True]
mutual_info_pool_size_values = [2]
num_examples_values = range(1,21)

#overall parameters
full_csv_filename = 'synth_results_full.csv'
num_samples = 100
f1_thresh = 0.8
pred_threshold = 0.2
img_dir = "plots"


# Generate all combinations of parameters
parameter_combinations = itertools.product(
    low_threshold_values,
    high_threshold_values,
    num_feature_selection_rounds_values,
    predicates_per_round_values,
    use_bins_values,
    depth_values,
    use_mutual_information_values,
    mutual_info_pool_size_values,
    num_examples_values
)

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
    'mutual_info_pool_size', 'num_examples', 'precision','recall','program'
]
def run():
    # Open CSV file for writing
    with open(full_csv_filename, 'w', newline='') as csvfile:
        # Create CSV writer
        csv_writer = csv.writer(csvfile)
        
        # Write header to CSV file
        csv_writer.writerow(full_csv_header)
        
    # Iterate through parameter combinations
    for params in parameter_combinations:
        params_dict = {'low':params[0],'high':params[1],'num_rounds':params[2],'preds_per_round':params[3],
                       'use_bins':params[4],'depth':params[5],'use_mi':params[6],'mi_pool':params[7],'examples':params[8]}
        # Instantiate AppConfig with the current parameter values
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
            num_examples=params_dict['examples'],
            full_csv=full_csv_value,
            manual=manual_value,
            eval=eval_value,
            debug=debug_value,
            prog_fname=prog_name_value
        )
        df = pd.DataFrame(columns=ind_csv_header)
        for run_no in range(num_samples):
            # Run the app and get evaluation results (results is now an array of tuples)
            app = App(config)
            prog = app.run()
            results = app.eval()
            precision, recall = compute_metrics(results)
            if precision + recall < 1e-5:
                f1 = 0
            else:
                f1 = (2*precision*recall) / (precision + recall)
            # Write rows to the CSV file with parameters and evaluation results
            for result in results:
                csv_writer.writerow(list(params_dict.values()) + result + [str(prog.preds)] )
        param_str = f"{params_dict['low']}-{params_dict['high']}_{params_dict['num_rounds']}x{params_dict['preds_per_round']}" + \
        f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}mi_{params_dict['mi_pool']*params_dict['preds_per_round']}midepth"    
        create_stacked_histogram(params_dict,results,param_str)

    print(f"Results written to {full_csv_filename}")

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
    precision = true_pos / (true_pos + false_pos)
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
    g.savefig(f"{img_dir}/facet_histogram_{param_str}.png")
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
    plt.savefig(f"{img_dir}/histogram_{param_str}.png")
    plt.close()


if __name__ == "__main__":
    run()