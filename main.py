import itertools
import numpy as np
import pandas as pd
import random
from synth import Runner, RunConfig
from utils import *
import argparse
import sys
import copy
from plots import create_pareto_plot, create_hist, make_baseline_plots
from util_scripts.img_viewer_csv import launch_app
from lang import Lang, MusicLang, ImgLang

class REPLQuitException(Exception):
    pass

class REPLParseError(Exception):
    pass

class App:
    def __init__(self,features_fname="extracted_features_detr_500.json",
                 examples_csv_fname="partial_labeled_sports.csv",prog_fname="prog_sports.txt",manual_value=False,
                 eval_value=True,debug_value=True,full_out_csv_filename = 'synth_results_full.csv',
                 baseline_csv_fname="predicted_labels_gemini.csv",synth_params = None, pp_params=None, baseline_params=None,
                 pareto_params=None,baseline_plot_fname="baseline",ind_csv_header = None, full_csv_header = None,input_doc_dir = "/home/jmfergie/coco_imgs",
                 baseline_csv_dir="csv_baseline",
                 task="image",img_dir = "plots",hist_dir = "hists",pareto_dir = "pareto",baseline_dir = "baseline",csv_dir = "csv_out"):
        
        self.manual_value = manual_value
        self.eval_value = eval_value
        self.debug_value = debug_value
        if task == "image":
            self.lang : Lang = ImgLang()
        elif task == "music":
            self.lang : Lang = MusicLang()
        else:
            raise Exception(f"Invalid task: {task}")
        self.img_dir = img_dir
        self.hist_dir = hist_dir
        self.pareto_dir = pareto_dir
        self.csv_dir = csv_dir
        self.baseline_dir = baseline_dir
        self.baseline_csv_dir = baseline_csv_dir
        self.input_img_dir = input_doc_dir

        self.baseline_csv_fname = f"{self.baseline_dir}/{baseline_csv_fname}"
        self.full_out_csv_filename = full_out_csv_filename
        self.features_fname = features_fname
        self.examples_csv_fname = examples_csv_fname
        self.prog_fname = prog_fname

        self.default_synth_params = {"low":[0.1],"high":[0.9],"num_rounds":range(1,6),"preds_per_round":[1,5,10,15,20,25,30],
                                 "bins":[True,False],"depth":[1],"mi":[True,False],"pool":[2],"examples":range(1,21)}
        self.synth_params = synth_params or self.default_synth_params
        self.default_pp_params = {"num_samples":50, "pred_thresh":0.1,"f1_thresh":0.8,"hist_type":"stacked",
                                  "pareto_scatterplot":False
                                  }
        self.pp_params = pp_params or self.default_pp_params
        self.default_baseline_params = {"f1_thresholds":np.linspace(0.0,1.0,20),
                                        'low':[0.1],'high':[0.9],'num_rounds':[5],'preds_per_round':[15],'bins':[True],
                                      'depth':[1],'mi':[True],'pool':[2],'k_vals':[10],'examples':range(1,21),'baseline_k':1,
                                      'baseline_diff_thresh':0.5}
        self.baseline_params = baseline_params or self.default_baseline_params
        self.default_pareto_params = {"low":[0.1],"high":[0.9],"bins":[True,False],"depth":[1],"mi":[True,False],"pool":[2],"examples":range(1,21)}
        self.pareto_params = pareto_params or self.default_pareto_params
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

    def run_once(self,params_dict: dict):
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
                        examples=params_dict['examples'],
                        lang=self.lang
                    )
        # Run the app and get evaluation results (results is now an array of tuples)
        runner = Runner(config)
        prog = runner.run()
        results = runner.eval()
        return results, prog

    def compute_baseline_scores(self) -> list[float]:
        ground_truth_labels = pd.read_csv(self.examples_csv_fname)
        predicted_labels = pd.read_csv(self.baseline_csv_fname)
        columns = predicted_labels.groupby(['k'])
        f1_scores = []
        for _, column in columns:
            merged_df = pd.merge(column, ground_truth_labels, on='fname', suffixes=('_pred', '_ground'))
            pred_vals = merged_df['val_pred'].tolist()
            ground_vals = merged_df['val_ground'].tolist()
            #print(pred_vals)
            #print(ground_vals)
            _, _, f1 = compute_metrics_baseline(list(zip(ground_vals,pred_vals)))
            f1_scores.append(f1)
        return f1_scores
    
    def compute_baseline_synth_scores(self):
        all_examples = pd.read_csv(self.examples_csv_fname)
        all_example_fnames = list(all_examples[all_examples['val']==True]['fname'])
        # Iterate through parameter combinations
        for params_dict in self.synth_param_iter(self.baseline_params):
            ind_results = []
            for run_no in range(self.pp_params['num_samples']):
                print("Run: ",run_no)
                user_examples = set(random.choices(all_example_fnames,k=params_dict['num_examples']))
                params_dict['examples'] = user_examples
                params_dict['run_no'] = run_no
                for num_rounds, predicates_per_round in self.synth_num_pred_iter(self.baseline_params):
                    print("Predicates: ",num_rounds * predicates_per_round)
                    params_dict['num_rounds'] = num_rounds
                    params_dict['preds_per_round'] = predicates_per_round
                    results, prog = self.run_once(params_dict)
                    for result in results:
                        fname = result[0]
                        expected_val = result[1]
                        preds_correct = result[2]
                        preds_total = result[3]
                        ind_results.append([params_dict['low'],params_dict['high'],params_dict['num_rounds'],params_dict['preds_per_round'],
                                        params_dict['use_bins'],params_dict['depth'],params_dict['use_mi'],params_dict['mi_pool'],
                                        params_dict['num_examples'],fname,expected_val,preds_correct,preds_total,run_no,prog])
            df = pd.DataFrame(columns = self.ind_csv_header,data=ind_results)
            df.to_csv(self.ind_csv_fname(params_dict,dir=self.baseline_csv_dir))

    def extract_baseline_exp_examples(self):
        all_results = {}
        for params_dict in self.synth_param_iter(self.baseline_params):
            params_results = {}
            for num_rounds, preds_per_round in self.synth_num_pred_iter(self.baseline_params):
                fname = self.ind_csv_fname(params_dict,dir=self.baseline_csv_dir)
                results = pd.read_csv(fname)
                preds_results:pd.DataFrame = results[(results['predicates_per_round']==preds_per_round) & (results['num_feature_selection_rounds']==num_rounds)]
                runs = preds_results.groupby('run_number')
                f1_scores = []
                for _,run in runs:
                    run_results = list(zip(list(run['filename']),list(run['expected_value']),list(run['predicates_correct']),list(run['total_predicates'])))
                    f1 = compute_max_f1_scores(run_results,self.baseline_params['f1_thresholds'])
                    f1_scores.append(f1)
                params_results[(num_rounds,preds_per_round)] = f1_scores
            all_results[self.param_str_top_level(params_dict)] = params_results
        return all_results
    
    def param_str_top_level(self,params_dict):
        param_str = f"{params_dict['low']}-{params_dict['high']}_" + \
                        f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}" + \
                        f"mi_{params_dict['mi_pool']}midepth_{params_dict['num_examples']}_examples"
        return param_str
    
    def param_str_hist(self,params_dict):
        return self.param_str_top_level(params_dict) + \
                f"_{params_dict['num_rounds']}x{params_dict['preds_per_round']}_{params_dict['run_no']}"
    
    def param_str_noex(self,params_dict):
        param_str = f"{params_dict['low']}-{params_dict['high']}_" + \
                        f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}" + \
                        f"mi_{params_dict['mi_pool']}midepth"
        return param_str
    
    def ind_csv_fname(self,params_dict,dir):
        param_str = self.param_str_top_level(params_dict)
        ind_csv_filename = f"{param_str}_results.csv"
        return f"{dir}/{ind_csv_filename}"
    
    def synth_param_iter(self,params):
        for param_dict in self.param_iter_noex(params):
            for num_examples in params['examples']:
                param_dict = copy.copy(param_dict)
                param_dict['num_examples'] = num_examples
                yield param_dict

    def param_iter_noex(self,params):
        parameter_combinations = itertools.product(
            params['low'],
            params['high'],
            params['bins'],
            params['depth'],
            params['mi'],
            params['pool']
        )
        for params in parameter_combinations:
            params_dict = {'low':params[0],'high':params[1],'use_bins':params[2],'depth':params[3],
                           'use_mi':params[4],'mi_pool':params[5]}
            yield params_dict

    def synth_num_pred_iter(self,params):
        pred_size_combinations = list(itertools.product(params['num_rounds'],params['preds_per_round']))
        pred_size_combinations.sort(key=lambda x: x[0]*x[1])
        return pred_size_combinations
    
    def run_all(self):
        all_examples = pd.read_csv(self.examples_csv_fname)
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
                    precision, recall, f1 = compute_metrics_synth(results,self.pp_params["pred_thresh"])
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
            df.to_csv(self.ind_csv_fname(params_dict,dir=self.csv_dir))
        df = pd.DataFrame(columns = self.full_csv_header,data=full_results)
        df.to_csv(f"{self.csv_dir}/{self.full_out_csv_filename}")
        print(f"Results written to {self.csv_dir}/{self.full_out_csv_filename}")

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
            try:
                self.parse_cmd(cmd)
            except REPLQuitException:
                print("Exiting")
                sys.exit(0)
            except REPLParseError as e:
                print(e)

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
            case "select":
                self.parse_select(terms[1:])
            case _:
                print("Invalid command: ",head)
    
    #syntax: set ctx1 param1 expr1 ... ctxN paramN exprN
    #warning: will evaluate arbitrary code. Don't use this for secure stuff
    def parse_set(self,terms):
        if len(terms) % 3 != 0:
            raise REPLParseError("Invalid SET expression. Please use the form set ctx1 param1 expr1 ... ctxN paramN exprN")
        for i in range(0,len(terms)-1,2):
            ctx = terms[i]
            param = terms[i+1]
            expr = terms[i+2]
            match ctx:
                case "pp":
                    params = self.pp_params
                case "synth":
                    params = self.synth_params
                case "baseline":
                    params = self.baseline_params
                case "pareto":
                    params = self.pareto_params
                case _:
                    raise REPLParseError("Invalid context: ",ctx)
            if param not in params:
                raise REPLParseError("Invalid parameter: ", param)
            params[param] = eval(expr)

    def parse_compute(self,terms):
        match terms[0]:
            case "pareto":
                self.run_all()
            case "baseline":
                self.compute_baseline_synth_scores()
            case _:
                raise REPLParseError(f"Invalid compute argument: {terms[0]}")

    def parse_plot(self,terms):
        match terms[0]:
            case "hist":
                self.plot_hists()
            case "pareto":
                self.plot_pareto()
            case "baseline":
                self.plot_baseline()
            case _:
                raise REPLParseError(f"Invalid plot argument: {terms[0]}")
        
    def parse_select(self,terms):
        def parse_params(terms, params):
            if len(terms) % 2 == 1:
                raise REPLParseError("Invalid select statement")
            for i in range(0,len(terms),2):
                params[terms[i]] = eval(terms[i+1])
        match terms[0]:
            case "baseline":
                params = copy.copy(self.baseline_params)
                parse_params(terms[1:],params)
                self.extract_baseline_examples(params)
            case "synth":
                params = copy.copy(self.synth_params)
                parse_params(terms[1:],params)
                self.extract_synth_diff(params)
            case "pareto":
                params = copy.copy(self.pareto_params)
                parse_params(terms[1:],params)
                self.extract_pareto_examples(params)
    
    def extract_baseline_examples(self,params):
        
        ground_truth_labels = pd.read_csv(self.examples_csv_fname)
        predicted_labels = pd.read_csv(self.baseline_csv_fname)
        baseline_k = predicted_labels[predicted_labels['k'] == self.baseline_params['baseline_k']]
        for params_dict in self.param_iter_noex(params):
            for num_rounds, preds_per_round in self.synth_num_pred_iter(params):
                for k in params['k_vals']:
                    params_dict['num_examples'] = k
                    csv_fname = self.ind_csv_fname(params_dict)
                    synth_vals = pd.read_csv(csv_fname)
                    synth_k = synth_vals[(synth_vals['num_feature_selection_rounds'] == num_rounds) & (synth_vals['predicates_per_round'] == preds_per_round)]
                    synth_k['pred'] = synth_k['predicates_correct'] >= 1
                    frac_df = synth_k.groupby('filename')['pred'].mean().reset_index()
                    frac_df.rename(columns={'pred': 'frac','filename':'fname'}, inplace=True)
                    
                    merged_synth_baseline = pd.merge(frac_df,baseline_k,on='fname', suffixes=('_synth', '_baseline'))
                    merged_synth_baseline_ground = pd.merge(merged_synth_baseline,ground_truth_labels,on='fname',suffixes=('','_ground'))
                    diff = merged_synth_baseline[(merged_synth_baseline_ground['frac'] < self.baseline_params['baseline_diff_thresh']) & (merged_synth_baseline_ground['val']) & (merged_synth_baseline_ground['val_ground'])]
                    diff_fnames = diff['fname'].tolist()
                    print(diff)
                    launch_app(None,self.input_doc_dir,self.features_fname,"Baseline comparison",diff_fnames)
                    

    def plot_pareto(self):
        full_results = pd.read_csv(f"{self.csv_dir}/{self.full_out_csv_filename}")
        full_results['num_preds'] = full_results['num_feature_selection_rounds'] * full_results['predicates_per_round']
        for param_dict in self.param_iter_noex(self.pareto_params):
            param_results = full_results[(full_results['low_threshold'] == param_dict['low']) & (full_results['high_threshold'] == param_dict['high']) & \
                (full_results['depth'] == param_dict['depth']) & (full_results['use_bins'] == param_dict['use_bins']) & \
                (full_results['use_mutual_information'] == param_dict['use_mi']) & (full_results['mutual_info_pool_size'] == param_dict['mi_pool'])]
            param_dict['f1_thresh'] = self.pp_params['f1_thresh']
            param_str = self.param_str_noex(param_dict)
            param_str = f"{self.img_dir}/{self.pareto_dir}/{param_str}_{self.pp_params['f1_thresh']}_{self.pp_params['num_samples']}"
            create_pareto_plot(param_results,param_dict,param_str,self.pp_params['pareto_scatterplot'])

    def plot_hists(self):
        for params_dict in self.synth_param_iter(self.synth_params):
            data = pd.read_csv(self.ind_csv_fname(params_dict))
            for num_rounds, preds_per_round in self.synth_num_pred_iter(self.synth_params):
                params_dict['num_rounds'] = num_rounds
                params_dict['preds_per_round'] = preds_per_round
                data_filtered = data[(data['num_feature_selection_rounds'] == num_rounds) & (data['predicates_per_round'] == preds_per_round)]
                if self.debug_value:
                    print(params_dict)
                runs = data_filtered.groupby('run_number')
                for run_no, run in runs:
                    if run_no % 10 == 0 and self.debug_value:
                        print(run_no)
                    params_dict['run_no'] = run_no
                    results = run.filter(['expected_value','predicates_correct'], axis=1)
                    param_str = f"{self.img_dir}/{self.hist_dir}/histogram_{self.param_str_hist(params_dict)}.png"
                    create_hist(results,params_dict,param_str,self.pp_params["hist_type"])
    
    def plot_baseline(self):
        all_synth_scores = np.array(1)#self.extract_baseline_exp_examples()
        baseline_scores = self.compute_baseline_scores()
        #baseline_scores = []
        for params_dict in self.param_iter_noex(self.baseline_params):
            for num_rounds, preds_per_round in self.synth_num_pred_iter(self.baseline_params):
                params_dict['num_rounds'] = num_rounds
                params_dict['preds_per_round'] = preds_per_round
                synth_scores_agg = []
                for num_examples in self.baseline_params['examples']:
                    params_dict['num_examples'] = num_examples
                    param_str = self.param_str_top_level(params_dict)
                    baseline_fname = f"{self.img_dir}/{self.baseline_dir}/{param_str}_{num_rounds}x{preds_per_round}_baseline.png"
                    synth_scores = np.array(1)#all_synth_scores[param_str][(num_rounds, preds_per_round)]
                    synth_scores_agg.append(synth_scores)
                make_baseline_plots(baseline_scores,np.array(synth_scores_agg),params_dict,baseline_fname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Synthesis CLI',
        description='CLI for running synthesis')
    parser.add_argument('-f','--fname',required=False)
    args = parser.parse_args()
    app = App(debug_value=False,task='image')
    if not args.fname:
        app.run_repl() 
    else:
        app.parse_prog(args.fname)
