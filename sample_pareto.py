import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def process_and_plot_data(data):
    # Convert the array to a pandas DataFrame
    df = pd.DataFrame(data, columns=['low_threshold', 'high_threshold', 'num_feature_selection_rounds',
                                     'predicates_per_round', 'use_bins', 'depth', 'use_mutual_information',
                                     'mutual_info_pool_size', 'num_examples', 'precision', 'recall', 'program',
                                     'run_no', 'f1', 'f1_above_thresh', 'subset'])
    cols_to_convert = ['low_threshold', 'high_threshold', 'num_feature_selection_rounds', 'predicates_per_round',
                   'depth', 'mutual_info_pool_size', 'num_examples', 'run_no']
    df[cols_to_convert] = df[cols_to_convert].astype(int)
    df['use_bins'] = df['use_bins'].astype(bool)
    df['use_mutual_information'] = df['use_mutual_information'].astype(bool)
    df['f1_above_thresh'] = df['f1_above_thresh'].astype(bool)

    # Create a new column for 'num_preds'
    df['num_preds'] = df['num_feature_selection_rounds'] * df['predicates_per_round']

    # Group by specified columns excluding 'num_feature_selection_rounds' and 'predicates_per_round'
    grouped = df.groupby(['low_threshold', 'high_threshold', 'use_bins', 'depth',
                          'use_mutual_information', 'mutual_info_pool_size'])

    # Initialize lists to store mean and standard deviation values for plotting
    

    # Iterate over each group
    for name, group in grouped:
        # Group by 'num_examples'
        nested_grouped = group.groupby('num_examples')
        examples = []
        means = []
        stds = []
        # Iterate over each 'num_examples' group
        for num_examples, nested_group in nested_grouped:
            #print(nested_group['num_examples'],nested_group['run_no'],nested_group['num_preds'])
            examples.append(num_examples)
            # Group by 'run_no'
            nested_group_run_no = nested_group.groupby('run_no')
            
            # Initialize list to store min values for each 'run_no'
            min_values_run_no = []

            # Iterate over each 'run_no' group
            for run_no, nested_group_run in nested_group_run_no:
                # Filter nested group by 'f1_above_thresh'
                nested_group_filtered = nested_group_run[nested_group_run['f1_above_thresh']]

                # If there are no values above f1 threshold, set min value to max 'num_preds' value
                if nested_group_filtered.empty:
                    min_values_run_no.append(nested_group_run['num_preds'].max())
                else:
                    # Get minimum value for 'num_preds'
                    min_values_run_no.append(nested_group_filtered['num_preds'].min())

            means.append(np.mean(min_values_run_no))
            stds.append(np.std(min_values_run_no))

        # Plotting
        plt.figure(figsize=(10, 6))
        sns.lineplot(x=examples, y=means)
        plt.errorbar(x=examples, y=means, yerr=stds, fmt='o', capsize=5)
        plt.xlabel('num_examples')
        plt.ylabel('Pareto optimal num_preds')
        plt.title(f'Pareto optimal num_preds vs num_examples for Group: {name}')
        plt.grid(True)
        plt.savefig(f"group_{name}_pareto.png")
        plt.close()


columns=['low_threshold', 'high_threshold', 'num_feature_selection_rounds','predicates_per_round', 
        'use_bins', 'depth', 'use_mutual_information',
        'mutual_info_pool_size', 'num_examples', 'precision', 'recall', 'program',
        'run_no', 'f1', 'f1_above_thresh', 'subset']

# Example usage:
data = [
    [1, 2, 3, 4, True, 6, True, 8, 10, 0.9, 0.8, 'program1', 1, 0.85, True, 'subset1'],
    [2, 3, 4, 5, False, 7, False, 9, 10, 0.88, 0.7, 'program2', 1, 0.82, True, 'subset2'],
    [1, 2, 3, 4, True, 6, True, 8, 20, 0.85, 0.75, 'program1', 2, 0.82, True, 'subset1'],
    [2, 3, 4, 5, False, 7, False, 9, 20, 0.87, 0.72, 'program2', 2, 0.84, False, 'subset2'],
]

import numpy as np
import pandas as pd

# Generate synthetic data
np.random.seed(42)

# Define the number of groups and data points per group
num_groups = 50
data_per_group = 20

# Generate random values for each group
groups_data = []
for _ in range(num_groups):
    low_threshold = np.random.randint(1, 10)
    high_threshold = np.random.randint(11, 20)
    use_bins = np.random.choice([True, False])
    depth = np.random.randint(5, 15)
    use_mutual_information = np.random.choice([True, False])
    mutual_info_pool_size = np.random.randint(5, 15)
    
    group_data = []
    for _ in range(data_per_group):
        num_feature_selection_rounds = np.random.randint(1, 5)
        predicates_per_round = np.random.randint(5, 10)
        num_examples = np.random.randint(1,5)
        precision = np.random.rand()
        recall = np.random.rand()
        program = np.random.choice(['program1', 'program2'])
        run_no = np.random.randint(1, 4)
        f1 = np.random.rand()
        f1_above_thresh = np.random.choice([True, False])
        subset = np.random.choice(['subset1', 'subset2'])
        
        group_data.append([low_threshold, high_threshold, num_feature_selection_rounds, predicates_per_round,
                           use_bins, depth, use_mutual_information, mutual_info_pool_size, num_examples,
                           precision, recall, program, run_no, f1, f1_above_thresh, subset])
    
    groups_data.extend(group_data)
#print(groups_data)



process_and_plot_data(groups_data)
