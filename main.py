import itertools
import numpy as np
import seaborn as sns
import matplotlib
import pandas as pd
import random
from synth import Runner, RunConfig
from utils import *
import argparse

class REPLQuitException(Exception):
    pass
class App:
    def __init__(self,features_fname="extracted_features_detr_500.json",
                 examples_csv_fname="partial_labeled_sports.csv",prog_fname="prog_sports.txt",manual_value=False,
                 eval_value=True,debug_value=True,full_out_csv_filename = 'synth_results_full.csv',
                 baseline_csv_fname="predicted_labels_gpt.csv",synth_params = None, pp_params=None, baseline_params=None,
                 baseline_plot_fname="baseline",ind_csv_header = None, full_csv_header = None):
        
        self.features_fname = features_fname
        self.examples_csv_fname = examples_csv_fname
        self.prog_fname = prog_fname
        self.baseline_csv_fname = f"{self.baseline_dir}/{baseline_csv_fname}"
        self.full_out_csv_filename = full_out_csv_filename
        self.baseline_plot_fname = baseline_plot_fname

        self.manual_value = manual_value
        self.eval_value = eval_value
        self.debug_value = debug_value

        self.img_dir = "plots"
        self.hist_dir = "hists"
        self.pareto_dir = "pareto"
        self.csv_dir = "csv_out"
        self.baseline_dir = "baseline"
        self.default_synth_params = {"low":[0.1],"high":[0.9],"num_rounds":range(1,6),"preds_per_round":[1,5,10,15,20,25,30],
                                 "bins":[True, False],"depth":[1],"mi":[True],"pool":[2],"examples":range(1,21)}
        self.synth_params = synth_params or self.default_synth_params
        self.default_pp_params = {"num_samples":50, "pred_thresh":0.5,"f1_thresh":0.8,"hist_type":"stacked",
                                  }
        self.pp_params = pp_params or self.default_pp_params
        self.default_baseline_params = {"baseline_num_rounds":5,"baseline_ppr":15,"f1_thresholds":np.linspace(0.0,1.0,20),
                                        'low':[0.1],'high':[0.9],'num_rounds':[7],'preds_per_round':[10],'use_bins':[True],
                                      'depth':[1],'use_mi':[True],'mi_pool':[2],'k_vals':range(1,20)}
        self.baseline_params = baseline_params or self.default_baseline_params
        self.ind_csv_header = ind_csv_header or [
            'low_threshold', 'high_threshold', 'num_feature_selection_rounds',
            'predicates_per_round', 'use_bins', 'depth', 'use_mutual_information',
            'mutual_info_pool_size', 'num_examples', 'filename', 'expected_value',
            'predicates_correct', 'total_predicates', 'run_number','program'
        ]

        self.full_csv_header = full_csv_header or [
            'low_threshold', 'high_threshold', 'num_feature_selection_rounds',
            'predicates_per_round', 'use_bins', 'depth', 'use_mutual_information',
            'mutual_info_pool_size', 'num_examples', 'precision','recall','program','run_no','f1','f1_above_thresh','subset'
        ]

    def run_once(self,params_dict):
        config = RunConfig(
                        low_threshold=params_dict['low'],
                        high_threshold=params_dict['high'],
                        num_feature_selection_rounds=params_dict['num_rounds'],
                        predicates_per_round=params_dict['preds_per_round'],
                        use_bins=params_dict['use_bins'],
                        depth=params_dict['depth'],
                        use_mutual_information=params_dict['use_mi'],
                        mutual_info_pool_size=params_dict['mi_pool'],
                        features_fname=self.features_fname,
                        num_examples=params_dict['num_examples'],
                        full_csv=self.examples_csv_fname,
                        manual=self.manual_value,
                        eval=self.eval_value,
                        debug=self.debug_value,
                        prog_fname=self.prog_fname,
                        examples=params_dict['examples']
                    )
        # Run the app and get evaluation results (results is now an array of tuples)
        runner = Runner(config)
        prog = runner.run()
        results = runner.eval()
        return results, prog

    def compute_baseline_scores(self):
        ground_truth_labels = pd.read_csv(self.full_examples_csv_fname)
        predicted_labels = pd.read_csv(self.baseline_csv_filename)
        grouped = predicted_labels.group_by(['k'])
        f1_scores = []
        for group in grouped:
            merged_df = pd.merge(group, ground_truth_labels, on='fname', suffixes=('_pred', '_ground'))
            pred_vals = merged_df['val_pred'].tolist()
            ground_vals = merged_df['val_ground'].tolist()
            _, _, f1 = compute_metrics_baseline(zip(ground_vals,pred_vals))
            f1_scores.append(f1)
        return f1_scores
    
    def extract_baseline_exp_scores(self):
        all_results = {}
        for params_dict in self.synth_param_iter(self.baseline_params):
            params_results = {}
            for num_rounds, preds_per_round in self.synth_num_pred_iter(self.baseline_params):
                fname = self.ind_csv_fname(params_dict)
                results = pd.read_csv(fname)
                preds_results = results[results['predicates_per_round']==preds_per_round and results['num_feature_selection_rounds']==num_rounds]
                runs = preds_results.groupby('run_number')
                f1_scores = []
                for group in runs:
                    run = group[0]
                    run_results = zip(list(run['filename']),list(run['expected_value']),list(run['predicates_correct']),list(run('total_predicates')))
                    f1 = compute_max_f1_scores(run_results)
                    f1_scores.append(f1)
                params_results[(num_rounds,preds_per_round)] = f1_scores
            all_results[self.param_str_top_level(params_dict)] = params_results
        return all_results
    
    def param_str_top_level(self,params_dict):
        param_str = f"{params_dict['low']}-{params_dict['high']}_" + \
                        f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}" + \
                        f"mi_{params_dict['mi_pool']}midepth_{params_dict['num_examples']}_examples"
        return param_str
    
    def param_str_hist(self,params_dict,):
        return self.param_str_top_level(params_dict) + \
                f"_{params_dict['num_rounds']}x{params_dict['preds_per_round']}_{params_dict['run_no']}"
    
    def ind_csv_fname(self,params_dict):
        param_str = self.param_str_top_level(params_dict)
        ind_csv_filename = f"{param_str}_results.csv"
        return f"{self.csv_dir}/{ind_csv_filename}"
    
    def synth_param_iter(self,params):
        # Generate all combinations of parameters
        parameter_combinations = itertools.product(
            params['low'],
            params['high'],
            params['bins'],
            params['depth'],
            params['mi'],
            params['pool'],
            params['examples']
        )
        for params in parameter_combinations:
            params_dict = {'low':params[0],'high':params[1],'use_bins':params[2],'depth':params[3],
                           'use_mi':params[4],'mi_pool':params[5],'num_examples':params[6]}
            yield params_dict

    def synth_num_pred_iter(self,params):
        pred_size_combinations = list(itertools.product(self.params['num_rounds'],self.params['preds_per_round']))
        pred_size_combinations.sort(key=lambda x: x[0]*x[1])
        return pred_size_combinations
    
    def run_all(self):
        all_examples = pd.read_csv(self.all_examples_csv_value)
        all_example_fnames = list(all_examples[all_examples['val']==True]['fname'])
        full_results = []
        # Iterate through parameter combinations
        for params_dict in self.synth_param_iter(self.synth_params):
            ind_results = []
            for run_no in range(self.pp_params['num_samples']):
                print("Run: ",run_no)
                user_examples = set(random.choices(all_example_fnames,k=params_dict['num_examples']))
                params_dict['examples'] = user_examples
                params_dict['run_no'] = run_no
                for num_rounds, predicates_per_round in self.synth_num_pred_iter(self.synth_params):
                    print("Predicates: ",num_rounds * predicates_per_round)
                    params_dict['num_rounds'] = num_rounds
                    params_dict['preds_per_round'] = predicates_per_round
                    results, prog = self.run_once(params_dict)
                    precision, recall, f1 = compute_metrics(result,self.pp_params["pred_thresh"])
                    f1_above_thresh = f1 > self.pp_params['f1_thresh']
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
                    
                    if f1_above_thresh:
                        break
            df = pd.DataFrame(columns = self.ind_csv_header,data=ind_results)
            df.to_csv(self.ind_csv_fname(params_dict))
        df = pd.DataFrame(columns = self.full_csv_header,data=full_results)
        df.to_csv(f"{self.csv_dir}/{self.full_csv_filename}")
        print(f"Results written to {self.csv_dir}/{self.full_csv_filename}")

    def parse_prog(self,fname):
        with open(fname,'r') as f:
            text = f.read()
        prog = text.split('\n')
        for line in prog:
            self.parse_cmd(line)

    def run_repl(self):
        print("Synthesis Eval Shell")
        while True:
            cmd = input(">>> ")
            self.parse_cmd(cmd)

    def parse_cmd(self,cmd):
        terms = cmd.split(' ')
        head = terms[0]
        match head:
            case "q":
                raise REPLQuitException()
            case "set":
                self.parse_set(terms[1:])
            case "compute":
                self.parse_compute(terms[1:])
            case "plot":
                self.parse_plot(terms[1:])
    
    #syntax: set param1 expr1 ... paramN exprN
    def parse_set(self,terms):
        for i in range(0,len(terms),2):
            if param1 == 
    def parse_compute(self,terms):
        if not terms:
            self.run_all()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Synthesis CLI',
        description='CLI for running synthesis')
    parser.add_argument('-f','--fname',required=False)
    args = parser.parse_args()
    app = App()
    if not args.fname:
        app.run_repl() 
    else:
        app.parse_prog(args.fname)
