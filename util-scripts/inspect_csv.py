import pandas as pd
import os
import json

features_fname_value = "extracted_features_detr_500.json"
all_examples_csv_value = "partial_labeled_sports.csv"
prog_name_value = "prog_sports.txt"

full_csv_filename = 'synth_results_full.csv'

img_dir = "plots"
hist_dir = "hists"
pareto_dir = "pareto"
csv_dir = "csv_out"
hist_type = "stacked"
with open(features_fname_value,'r') as f:
    features = json.load(f)
    print(features)
params_dict = {'low':0.1,'high':0.9,'use_bins':True,'depth':1,'use_mi':True,'mi_pool':2,'num_examples':3}
def query(params_dict,run_no=-1,num_rounds=-1,preds_per_round=-1):
    param_str = f"{params_dict['low']}-{params_dict['high']}_" + \
                f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}" + \
                f"mi_{params_dict['mi_pool']}midepth_{params_dict['num_examples']}_examples"
    ind_csv_filename = f"{param_str}_results.csv" 
    df = pd.read_csv(f"{csv_dir}/{ind_csv_filename}")
    if run_no != 0:
        df = df.query(f'run_number == {run_no}')
    if num_rounds != 0:
        df = df.query(f'num_feature_selection_rounds == {num_rounds}')
    if preds_per_round != 0:
        df = df.query(f'predicates_per_round == {preds_per_round}')
    nonzero_preds = df.query('total_predicates != 0')
    df['pred_score'] = nonzero_preds['predicates_correct']/nonzero_preds['total_predicates']
    false_pos = df.query('pred_score > 0.0 and expected_value == False')
    false_neg = df.query('pred_score == 0.0 and expected_value == True')
    true_pos = df.query('pred_score > 0.0 and expected_value == True')
    true_neg = df.query('pred_score == 0.0 and expected_value == False')
    print(false_neg)
    for row in false_neg.iterrows():
        print(features[row[1]['filename']])

query(params_dict,run_no=1)
