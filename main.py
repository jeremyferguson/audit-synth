import itertools
import numpy as np
import pandas as pd
import random
import time
from synth import Runner, RunConfig
from utils import *
import argparse
import sys
import os
import copy
import json
from plots import (
    create_pareto_plot,
    make_baseline_plots,
    make_ablation_plots,
    make_performance_plot,
    make_heuristic_plots,
)
from util_scripts.img_viewer_csv import launch_app
from lang import Lang, MusicLang, ImgLang


class REPLQuitException(Exception):
    pass


class REPLParseError(Exception):
    pass


class App:
    @property
    def out_dir(self):
        return self.task_params["out_dir"]
    
    @out_dir.setter
    def out_dir(self,out_dir):
        self.task_params["out_dir"] = out_dir

    @property
    def task(self):
        return self.task_params["task"]
    
    @task.setter
    def task(self,task):
        self.task_params["task"] = task

    @property
    def features_fname(self):
        return self.task_params["features_fname"]
    
    @features_fname.setter
    def features_fname(self,features_fname):
        self.task_params["features_fname"] = features_fname

    @property
    def examples_csv_fname(self):
        return self.task_params["examples_csv_fname"]
    
    @examples_csv_fname.setter
    def examples_csv_fname(self,examples_csv_fname):
        self.task_params["examples_csv_fname"] = examples_csv_fname

    @property
    def prog_fname(self):
        return self.task_params["prog_fname"]
    
    @prog_fname.setter
    def prog_fname(self,prog_fname):
        self.task_params["prog_fname"] = prog_fname

    @property
    def full_out_csv_filename(self):
        return self.task_params["full_out_csv_filename"]
    
    @full_out_csv_filename.setter
    def full_out_csv_filename(self,full_out_csv_filename):
        self.task_params["full_out_csv_filename"] = full_out_csv_filename

    @property
    def baseline_labels_fname(self):
        return self.task_params["baseline_labels_fname"]
    
    @baseline_labels_fname.setter
    def baseline_labels_fname(self,baseline_labels_fname):
        self.task_params["baseline_labels_fname"] = baseline_labels_fname
    

    def __init__(
        self,
        manual_value=False,
        eval_value=True,
        debug_value=False,
        synth_params=None,
        pp_params=None,
        baseline_params=None,
        pareto_params=None,
        task_params=None,
        ind_csv_header=None,
        full_csv_header=None,
        perf_csv_header=None,
        heuristic_params=None,
        ablation_params=None,
        perf_params=None,
        baseline_csv_dir="csv_baseline",
        img_dir="plots",
        heuristic_dir="heuristic",
        ablation_dir="ablation",
        pareto_dir="pareto",
        perf_dir="performance",
        baseline_dir="baseline",
        csv_dir="csv_out",
    ):
        self.manual_value = manual_value
        self.eval_value = eval_value
        self.debug_value = debug_value
        
        self.default_task_params = {
            "out_dir":"sports",
            "task": "image",
            "full_out_csv_filename":"synth_results_full.csv",
            "baseline_labels_fname":"baseline/predicted_labels_gemini.csv",
            "features_fname":"jsons/extracted_features_detr_500.json",
            "examples_csv_fname":"partial_labeled_sports.csv",
            "prog_fname":"prog_sports.txt"
        }
        self.task_params = task_params or self.default_task_params
        self.default_synth_params = {
            "low": [0.1],
            "high": [0.9],
            "num_rounds": range(1, 6),
            "preds_per_round": [1, 5, 10, 15, 20, 25, 30],
            "bins": [True],
            "depth": [1],
            "mi": [True],
            "pool": [2],
            "examples": range(1, 21),
        }
        self.synth_params = synth_params or self.default_synth_params
        self.default_pp_params = {
            "num_samples": 50,
            "pred_thresh": 0.1,
            "f1_thresh": 0.8,
            "hist_type": "stacked",
            "pareto_scatterplot": False,
        }
        self.pp_params = pp_params or self.default_pp_params
        self.default_baseline_params = {
            "f1_thresholds": np.linspace(0.0, 1.0, 20),
            "low": [0.1],
            "high": [0.9],
            "num_rounds": [5],
            "preds_per_round": [15],
            "bins": [True],
            "depth": [1],
            "mi": [True],
            "pool": [2],
            "k_vals": [10],
            "examples": range(1, 31),
            "baseline_k": 1,
            "baseline_diff_thresh": 0.5,
        }
        self.baseline_params = baseline_params or self.default_baseline_params
        self.default_pareto_params = {
            "low": [0.1],
            "high": [0.9],
            "bins": [True],
            "depth": [1],
            "mi": [True],
            "pool": [2],
            "examples": range(1, 21),
        }
        self.pareto_params = pareto_params or self.default_pareto_params
        self.default_pareto_params = {
            "low": [0.1],
            "high": [0.9],
            "bins": [True],
            "depth": [1],
            "mi": [True],
            "pool": [2],
            "examples": [1, 20, 40, 60, 80],
            "num_rounds": range(1, 6),
            "preds_per_round": [1, 5, 10, 15, 20, 25, 30],
        }
        self.pareto_params = pareto_params or self.default_pareto_params
        self.default_ablation_params = {
            "low": [0.1],
            "high": [0.9],
            "bins": [True, False],
            "depth": [1],
            "mi": [True, False],
            "pool": [2],
            "examples": range(1, 21),
            "num_rounds": [5],
            "preds_per_round": [15],
        }
        self.ablation_params = ablation_params or self.default_ablation_params
        self.default_perf_params = {
            "low": [0.1],
            "high": [0.9],
            "bins": [True],
            "depth": [1],
            "mi": [True],
            "pool": [2],
            "examples": [0],
            "prog_size": range(1, 21),
            "num_rounds": [53],
            "preds_per_round": [10],
        }
        self.perf_params = perf_params or self.default_perf_params
        self.default_heuristic_params = {
            "low": [0.1],
            "high": [0.9],
            "bins": [True],
            "depth": [1],
            "mi": [True],
            "pool": [2],
            "examples": [10, 20, 30],
            "num_rounds": range(1, 21),
            "preds_per_round": [5],
            "heuristics": ["rand", "freq"],
        }
        self.heuristic_params = heuristic_params or self.default_heuristic_params
        self.ind_csv_header = ind_csv_header or [
            "low_threshold",
            "high_threshold",
            "num_feature_selection_rounds",
            "predicates_per_round",
            "use_bins",
            "depth",
            "use_mutual_information",
            "mutual_info_pool_size",
            "num_examples",
            "filename",
            "expected_value",
            "predicates_correct",
            "total_predicates",
            "run_number",
            "program",
        ]

        self.full_csv_header = full_csv_header or [
            "low_threshold",
            "high_threshold",
            "num_feature_selection_rounds",
            "predicates_per_round",
            "use_bins",
            "depth",
            "use_mutual_information",
            "mutual_info_pool_size",
            "num_examples",
            "precision",
            "recall",
            "program",
            "run_no",
            "f1",
            "f1_above_thresh",
            "subset",
        ]

        self.perf_csv_header = perf_csv_header or [
            "low_threshold",
            "high_threshold",
            "num_feature_selection_rounds",
            "predicates_per_round",
            "use_bins",
            "depth",
            "use_mutual_information",
            "mutual_info_pool_size",
            "num_examples",
            "run_number",
            "time",
            "prog_length",
            "round"
        ]

        self.img_dir = img_dir
        self.csv_dir = csv_dir
        self.baseline_csv_dir = baseline_csv_dir
    
        self.pareto_dir = pareto_dir
        self.baseline_dir = baseline_dir
        self.ablation_dir = ablation_dir
        self.perf_dir = perf_dir
        self.heuristic_dir = heuristic_dir
        self.make_dirs()
        self.construct_lang()

    def construct_lang(self):
        if self.task == "image":
            self.lang: Lang = ImgLang()
        elif self.task == "music":
            self.lang: Lang = MusicLang()
        else:
            raise Exception(f"Invalid task: {self.task}")
        
    def make_dirs(self):
        os.makedirs(os.path.join(self.out_dir,self.csv_dir),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,self.baseline_csv_dir),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,self.img_dir,self.pareto_dir),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,self.img_dir,self.baseline_dir),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,self.img_dir,self.ablation_dir),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,self.img_dir,self.perf_dir),exist_ok=True)
        os.makedirs(os.path.join(self.out_dir,self.img_dir,self.heuristic_dir),exist_ok=True)

    def run_once(self, params_dict: dict, prog_fname=None):
        config = RunConfig(
            low_threshold=params_dict["low"],
            high_threshold=params_dict["high"],
            num_feature_selection_rounds=params_dict["num_rounds"],
            predicates_per_round=params_dict["preds_per_round"],
            use_bins=params_dict["use_bins"],
            depth=params_dict["depth"],
            use_mutual_information=params_dict["use_mi"],
            mutual_info_pool_size=params_dict["mi_pool"],
            features_fname=self.features_fname,
            num_examples=params_dict["num_examples"],
            full_csv=self.examples_csv_fname,
            manual=self.manual_value,
            eval=self.eval_value,
            debug=self.debug_value,
            prog_fname=prog_fname or self.prog_fname,
            examples=params_dict["examples"],
            lang=self.lang,
            heuristic=params_dict["heuristic"]
            if "heuristic" in params_dict
            else "freq",
            use_not=params_dict["use_not"]
            if "use_not" in params_dict
            else True,
        )
        # Run the app and get evaluation results (results is now an array of tuples)
        runner = Runner(config)
        prog = runner.run()
        results = runner.eval()
        return results, prog

    def compute_baseline_scores(self) -> list[float]:
        ground_truth_labels = pd.read_csv(self.examples_csv_fname)
        predicted_labels = pd.read_csv(self.baseline_labels_fname)
        columns = predicted_labels.groupby(["k"])
        f1_scores = []
        for k, column in columns:
            merged_df = pd.merge(
                column, ground_truth_labels, on="fname", suffixes=("_pred", "_ground")
            )
            pred_vals = merged_df["val_pred"].tolist()
            pred_vals = [val == "True" or val == True for val in pred_vals]
            ground_vals = merged_df["val_ground"].tolist()

            _, _, f1 = compute_metrics_baseline(list(zip(ground_vals, pred_vals)))
            f1_scores.append([k[0], f1])
        return pd.DataFrame(data=f1_scores, columns=["num_examples", "f1"])

    def compute_baseline_synth_scores(self):
        all_examples = pd.read_csv(self.examples_csv_fname)
        all_example_fnames = list(all_examples[all_examples["val"] == True]["fname"])
        # Iterate through parameter combinations
        for params_dict in self.synth_param_iter(self.baseline_params,"low","high","mi","pool","depth","bins"):
            ind_results = []
            if self.debug_value:
                print(params_dict)
            for run_no in range(self.pp_params["num_samples"]):
                if self.debug_value:
                    print("Run: ", run_no)
                for num_rounds, predicates_per_round in self.synth_num_pred_iter(
                    self.baseline_params
                ):
                    if self.debug_value:
                        print("Predicates: ", num_rounds * predicates_per_round)
                    params_dict["num_rounds"] = num_rounds
                    params_dict["preds_per_round"] = predicates_per_round
                    for num_examples in self.baseline_params["examples"]:
                        user_examples = set(
                            random.choices(all_example_fnames, k=num_examples)
                        )
                        params_dict["examples"] = user_examples
                        params_dict["num_examples"] = num_examples
                        params_dict["run_no"] = run_no
                        results, prog = self.run_once(params_dict)
                        for result in results:
                            fname = result[0]
                            expected_val = result[1]
                            preds_correct = result[2]
                            preds_total = result[3]
                            ind_results.append(
                                [
                                    params_dict["low"],
                                    params_dict["high"],
                                    params_dict["num_rounds"],
                                    params_dict["preds_per_round"],
                                    params_dict["use_bins"],
                                    params_dict["depth"],
                                    params_dict["use_mi"],
                                    params_dict["mi_pool"],
                                    num_examples,
                                    fname,
                                    expected_val,
                                    preds_correct,
                                    preds_total,
                                    run_no,
                                    prog,
                                ]
                            )
            df = pd.DataFrame(columns=self.ind_csv_header, data=ind_results)
            param_str = self.param_str_noex(params_dict)
            ind_csv_filename = f"{param_str}_results.csv"
            df.to_csv(os.path.join(self.out_dir,self.baseline_csv_dir,ind_csv_filename))

    def extract_baseline_exp_examples(self):
        all_results = {}
        for params_dict in self.synth_param_iter(self.baseline_params,"low","high","bins","depth","pool","mi"):
            params_results = {}
            param_str = self.param_str_noex(params_dict)
            ind_csv_filename = f"{param_str}_results.csv"
            fname = os.path.join(self.out_dir,self.baseline_csv_dir,ind_csv_filename)
            results = pd.read_csv(fname)
            for num_rounds, preds_per_round in self.synth_num_pred_iter(
                self.baseline_params
            ):
                if self.debug_value:
                    print(params_dict)
                preds_results: pd.DataFrame = results[
                    (results["predicates_per_round"] == preds_per_round)
                    & (results["num_feature_selection_rounds"] == num_rounds)
                ]
                runs = preds_results.groupby("run_number")
                f1_scores = []
                for _, run in runs:
                    run_scores = []
                    for k in self.baseline_params["examples"]:
                        column = run[run["num_examples"] == k]
                        run_results = list(
                            zip(
                                list(column["filename"]),
                                list(column["expected_value"]),
                                list(column["predicates_correct"]),
                                list(column["total_predicates"]),
                            )
                        )
                        f1 = compute_max_f1_scores(run_results)
                        run_scores.append(f1)
                    f1_scores.append(run_scores)
                params_results[(num_rounds, preds_per_round)] = f1_scores
            all_results[param_str] = params_results
        return all_results

    def extract_ablation_results(self):
        all_results = {}
        for params_dict in self.synth_param_iter(self.ablation_params,"low","high","bins","mi","pool","depth"):
            params_results = {}
            param_str = self.param_str_noex(params_dict)
            ind_csv_filename = f"{param_str}_ablation_results.csv"
            fname = os.path.join(self.out_dir,self.csv_dir,ind_csv_filename)
            results = pd.read_csv(fname)
            for num_rounds, preds_per_round in self.synth_num_pred_iter(
                self.ablation_params
            ):
                if self.debug_value:
                    print(params_dict)
                preds_results: pd.DataFrame = results[
                    (results["predicates_per_round"] == preds_per_round)
                    & (results["num_feature_selection_rounds"] == num_rounds)
                ]
                runs = preds_results.groupby("run_number")
                f1_scores = []
                for _, run in runs:
                    run_scores = []
                    for k in self.ablation_params["examples"]:
                        column = run[run["num_examples"] == k]
                        run_results = list(
                            zip(
                                list(column["filename"]),
                                list(column["expected_value"]),
                                list(column["predicates_correct"]),
                                list(column["total_predicates"]),
                            )
                        )
                        f1 = compute_max_f1_scores(run_results)
                        run_scores.append(f1)
                    f1_scores.append(run_scores)
                params_results[(num_rounds, preds_per_round)] = f1_scores
            all_results[param_str] = params_results
        return all_results

    def extract_heuristic_results(self):
        all_results = {}
        for params_dict in self.synth_param_iter(self.heuristic_params,"low","high","bins","mi","pool","depth"):
            params_results = {}
            param_str = self.param_str_noex(params_dict)
            ind_csv_filename = f"{param_str}_heuristic_results.csv"
            fname = os.path.join(self.out_dir,self.csv_dir,ind_csv_filename)
            results = pd.read_csv(fname)
            freq_results = results[results["heuristic"] == "freq"]
            rand_results = results[results["heuristic"] == "rand"]
            for k in self.heuristic_params["examples"]:
                if self.debug_value:
                    print(params_dict)
                preds_results: pd.DataFrame = freq_results[
                    (freq_results["num_examples"] == k)
                ]
                runs = preds_results.groupby("run_number")
                f1_scores = []
                for _, run in runs:
                    run_scores = []
                    for num_rounds, preds_per_round in self.synth_num_pred_iter(
                        self.heuristic_params
                    ):
                        column = run[
                            (run["predicates_per_round"] == preds_per_round)
                            & (run["num_feature_selection_rounds"] == num_rounds)
                        ]
                        run_results = list(
                            zip(
                                list(column["filename"]),
                                list(column["expected_value"]),
                                list(column["predicates_correct"]),
                                list(column["total_predicates"]),
                            )
                        )
                        f1 = compute_max_f1_scores(run_results)
                        run_scores.append(f1)
                    f1_scores.append(run_scores)
                params_results[k] = f1_scores
            runs = rand_results.groupby("run_number")
            f1_scores = []
            for _, run in runs:
                run_scores = []
                for num_rounds, preds_per_round in self.synth_num_pred_iter(
                    self.heuristic_params
                ):
                    column = run[
                        (run["predicates_per_round"] == preds_per_round)
                        & (run["num_feature_selection_rounds"] == num_rounds)
                    ]
                    run_results = list(
                        zip(
                            list(column["filename"]),
                            list(column["expected_value"]),
                            list(column["predicates_correct"]),
                            list(column["total_predicates"]),
                        )
                    )
                    f1 = compute_max_f1_scores(run_results)
                    run_scores.append(f1)
                f1_scores.append(run_scores)
            params_results["rand"] = f1_scores
            all_results[param_str] = params_results
        return all_results

    def param_str_top_level(self, params_dict):
        param_str = (
            f"{params_dict['low']}-{params_dict['high']}_"
            + f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}"
            + f"mi_{params_dict['mi_pool']}midepth_{params_dict['num_examples']}_examples"
        )
        return param_str

    def param_str_hist(self, params_dict):
        return (
            self.param_str_top_level(params_dict)
            + f"_{params_dict['num_rounds']}x{params_dict['preds_per_round']}_{params_dict['run_no']}"
        )

    def param_str_noex(self, params_dict):
        param_str = (
            f"{params_dict['low']}-{params_dict['high']}_"
            + f"preds_{params_dict['use_bins']}bins_{params_dict['depth']}depth_{params_dict['use_mi']}"
            + f"mi_{params_dict['mi_pool']}midepth"
        )
        return param_str

    def ind_csv_fname(self, params_dict, dir):
        param_str = self.param_str_top_level(params_dict)
        ind_csv_filename = f"{param_str}_results.csv"
        return os.path.join(dir,ind_csv_filename)

    def synth_param_iter(self, all_params, *params):
        index_to_dict = {}
        params_to_iterate = []
        curr_index = 0
        if "low" in params:
            params_to_iterate.append(all_params["low"])
            index_to_dict["low"] = curr_index
            curr_index += 1
        if "high" in params:
            params_to_iterate.append(all_params["high"])
            index_to_dict["high"] = curr_index
            curr_index += 1
        if "bins" in params:
            params_to_iterate.append(all_params["bins"])
            index_to_dict["bins"] = curr_index
            curr_index += 1
        if "depth" in params:
            params_to_iterate.append(all_params["depth"])
            index_to_dict["depth"] = curr_index
            curr_index += 1
        if "pool" in params:
            params_to_iterate.append(all_params["pool"])
            index_to_dict["pool"] = curr_index
            curr_index += 1
        if "mi" in params:
            params_to_iterate.append(all_params["mi"])
            index_to_dict["mi"] = curr_index
            curr_index += 1
        if "examples" in params:
            params_to_iterate.append(all_params["examples"])
            index_to_dict["examples"] = curr_index
            curr_index += 1
        for param_tuple in itertools.product(*params_to_iterate):
            param_dict = {}
            if "low" in params:
                param_dict["low"] = param_tuple[index_to_dict["low"]]
            if "high" in params:
                param_dict["high"] = param_tuple[index_to_dict["high"]]
            if "bins" in params:
                param_dict["use_bins"] = param_tuple[index_to_dict["bins"]]
            if "depth" in params:
                param_dict["depth"] = param_tuple[index_to_dict["depth"]]
            if "mi" in params:
                param_dict["use_mi"] = param_tuple[index_to_dict["mi"]]
            if "pool" in params:
                param_dict["mi_pool"] = param_tuple[index_to_dict["pool"]]
            if "examples" in params:
                param_dict["num_examples"] = param_tuple[index_to_dict["examples"]]
            yield param_dict

    def synth_num_pred_iter(self, params):
        pred_size_combinations = list(
            itertools.product(params["num_rounds"], params["preds_per_round"])
        )
        pred_size_combinations.sort(key=lambda x: x[0] * x[1])
        return pred_size_combinations

    def compute_pareto(self):
        all_examples = pd.read_csv(self.examples_csv_fname)
        all_example_fnames = list(all_examples[all_examples["val"] == True]["fname"])
        full_results = []
        if self.debug_value:
            print(self.pareto_params)
        # Iterate through parameter combinations
        for params_dict in self.synth_param_iter(self.pareto_params,"low","high","bins","mi","pool","depth","examples"):
            ind_results = []
            if self.debug_value:
                print(params_dict)
            for run_no in range(self.pp_params["num_samples"]):
                if self.debug_value:
                    print("Run: ", run_no)
                user_examples = set(
                    random.choices(all_example_fnames, k=params_dict["num_examples"])
                )
                params_dict["examples"] = user_examples
                params_dict["run_no"] = run_no
                for num_rounds, predicates_per_round in self.synth_num_pred_iter(
                    self.pareto_params
                ):
                    if self.debug_value:
                        print("Predicates: ", num_rounds * predicates_per_round)
                    params_dict["num_rounds"] = num_rounds
                    params_dict["preds_per_round"] = predicates_per_round
                    results, prog = self.run_once(params_dict)
                    precision, recall, f1 = compute_metrics_synth(
                        results, self.pp_params["pred_thresh"]
                    )
                    f1_above_thresh = f1 > self.pp_params["f1_thresh"]
                    full_results.append(
                        [
                            params_dict["low"],
                            params_dict["high"],
                            params_dict["num_rounds"],
                            params_dict["preds_per_round"],
                            params_dict["use_bins"],
                            params_dict["depth"],
                            params_dict["use_mi"],
                            params_dict["mi_pool"],
                            params_dict["num_examples"],
                            precision,
                            recall,
                            prog,
                            run_no,
                            f1,
                            f1_above_thresh,
                            user_examples,
                        ]
                    )
                    for result in results:
                        fname = result[0]
                        expected_val = result[1]
                        preds_correct = result[2]
                        preds_total = result[3]
                        ind_results.append(
                            [
                                params_dict["low"],
                                params_dict["high"],
                                params_dict["num_rounds"],
                                params_dict["preds_per_round"],
                                params_dict["use_bins"],
                                params_dict["depth"],
                                params_dict["use_mi"],
                                params_dict["mi_pool"],
                                params_dict["num_examples"],
                                fname,
                                expected_val,
                                preds_correct,
                                preds_total,
                                run_no,
                                prog,
                            ]
                        )

                    if f1_above_thresh:
                        break
            df = pd.DataFrame(columns=self.ind_csv_header, data=ind_results)
            df.to_csv(os.path.join(self.out_dir,self.csv_dir,f"pareto_{self.param_str_top_level(params_dict)}_results.csv"))
        df = pd.DataFrame(columns=self.full_csv_header, data=full_results)
        df.to_csv(os.path.join(self.out_dir,self.csv_dir,f"pareto_{self.full_out_csv_filename}"))
        print(f"Results written to {os.path.join(self.out_dir,self.csv_dir,f'pareto_{self.full_out_csv_filename}')}")

    def compute_ablation(self):
        all_examples = pd.read_csv(self.examples_csv_fname)
        all_example_fnames = list(all_examples[all_examples["val"] == True]["fname"])
        # Iterate through parameter combinations
        for params_dict in self.synth_param_iter(self.ablation_params,"low","high","mi","bins","depth","pool"):
            ind_results = []
            if self.debug_value:
                print(params_dict)
            for run_no in range(self.pp_params["num_samples"]):
                if self.debug_value:
                    print("Run: ", run_no)
                for num_rounds, predicates_per_round in self.synth_num_pred_iter(
                    self.ablation_params
                ):
                    if self.debug_value:
                        print("Predicates: ", num_rounds * predicates_per_round)
                    params_dict["num_rounds"] = num_rounds
                    params_dict["preds_per_round"] = predicates_per_round
                    for num_examples in self.ablation_params["examples"]:
                        user_examples = set(
                            random.choices(all_example_fnames, k=num_examples)
                        )
                        params_dict["examples"] = user_examples
                        params_dict["num_examples"] = num_examples
                        params_dict["run_no"] = run_no
                        results, prog = self.run_once(params_dict)
                        for result in results:
                            fname = result[0]
                            expected_val = result[1]
                            preds_correct = result[2]
                            preds_total = result[3]
                            ind_results.append(
                                [
                                    params_dict["low"],
                                    params_dict["high"],
                                    params_dict["num_rounds"],
                                    params_dict["preds_per_round"],
                                    params_dict["use_bins"],
                                    params_dict["depth"],
                                    params_dict["use_mi"],
                                    params_dict["mi_pool"],
                                    num_examples,
                                    fname,
                                    expected_val,
                                    preds_correct,
                                    preds_total,
                                    run_no,
                                    prog,
                                ]
                            )
            df = pd.DataFrame(columns=self.ind_csv_header, data=ind_results)
            param_str = self.param_str_noex(params_dict)
            ind_csv_filename = f"{param_str}_ablation_results.csv"
            df.to_csv(os.path.join(self.out_dir,self.csv_dir,ind_csv_filename))

    def gen_synth_progs(self, depth=1):
        docs = {}
        with open(self.features_fname, "r") as f:
            json_dict = json.loads(f.read())
            for doc in json_dict:
                docs[doc] = self.lang.Document(doc, json_dict[doc])
        all_features = set()
        for fname in docs:
            for feature in docs[fname].features:
                all_features.add(feature)
        all_preds = list(self.lang.predGen(docs, all_features, depth))
        synth_progs = []
        for prog_length in self.perf_params["prog_size"]:
            synth_progs_row = []
            for _ in range(self.pp_params["num_samples"]):
                synth_progs_row.append(random.sample(all_preds, k=prog_length))
            synth_progs.append(synth_progs_row)
        return synth_progs

    def compute_performance(self):
        all_examples = pd.read_csv(self.examples_csv_fname)
        all_example_fnames = list(all_examples[all_examples["val"] == True]["fname"])
        synth_progs = self.gen_synth_progs()
        full_results = []
        # Iterate through parameter combinations
        for _ in range(2):
            full_results = []
            # with open("freq_heuristic.csv", "w") as f:
            #     f.write("run_no,prog_len,time,used_bins\n")
            for params_dict in self.synth_param_iter(self.perf_params,"low","high","bins","mi","pool","depth","examples"):
                params_dict["use_not"] = False
                user_examples = set(
                    random.choices(all_example_fnames, k=params_dict["num_examples"])
                )
                params_dict["examples"] = user_examples
                if self.debug_value:
                    print(params_dict)
                for i, prog_length in enumerate(self.perf_params["prog_size"]):
                    if self.debug_value:
                        print("Program length: ", prog_length)
                    for num_rounds, predicates_per_round in self.synth_num_pred_iter(
                        self.perf_params
                    ):
                        if self.debug_value:
                            print("Predicates: ", num_rounds * predicates_per_round)
                        params_dict["num_rounds"] = num_rounds
                        params_dict["preds_per_round"] = predicates_per_round
                        for run_no in range(self.pp_params["num_samples"]):
                            if self.debug_value:
                                print("Run: ", run_no)
                            synth_prog = synth_progs[i][run_no]
                            with open("prog_exp.txt", "w") as f:
                                f.write("\n".join([str(pred) for pred in synth_prog]))

                            params_dict["run_no"] = run_no
                            start = time.perf_counter_ns()

                            prog, final_round = self.run_once(
                                params_dict, prog_fname="prog_exp.txt"
                            )
                            success = prog_length == len(prog)
                            end = time.perf_counter_ns()
                            delta = (end - start) / 1000000000
                            if self.debug_value:
                                print("Total time: ", delta)
                            full_results.append(
                                [
                                    params_dict["low"],
                                    params_dict["high"],
                                    params_dict["num_rounds"],
                                    params_dict["preds_per_round"],
                                    params_dict["use_bins"],
                                    params_dict["depth"],
                                    params_dict["use_mi"],
                                    params_dict["mi_pool"],
                                    params_dict["num_examples"],
                                    run_no,
                                    delta,
                                    prog_length,
                                    final_round,
                                ]
                            )
                df = pd.DataFrame(columns=self.perf_csv_header, data=full_results)
                ind_csv_filename = "perf_results.csv"
                df.to_csv(os.path.join(self.out_dir,self.csv_dir,ind_csv_filename))

    def compute_heuristics(self):
        all_examples = pd.read_csv(self.examples_csv_fname)
        all_example_fnames = list(all_examples[all_examples["val"] == True]["fname"])
        # Iterate through parameter combinations
        for params_dict in self.synth_param_iter(self.heuristic_params,"low","high","bins","mi","pool","depth"):
            ind_results = []
            if self.debug_value:
                print(params_dict)
            for heuristic in self.heuristic_params["heuristics"]:
                if self.debug_value:
                    print("heuristic: ", heuristic)
                params_dict["heuristic"] = heuristic
                for run_no in range(self.pp_params["num_samples"]):
                    if self.debug_value:
                        print("Run: ", run_no)
                    for num_rounds, predicates_per_round in self.synth_num_pred_iter(
                        self.heuristic_params
                    ):
                        if self.debug_value:
                            print("Predicates: ", num_rounds * predicates_per_round)
                        params_dict["num_rounds"] = num_rounds
                        params_dict["preds_per_round"] = predicates_per_round
                        if heuristic == "freq":
                            for num_examples in self.heuristic_params["examples"]:
                                user_examples = set(
                                    random.choices(all_example_fnames, k=num_examples)
                                )
                                params_dict["examples"] = user_examples
                                params_dict["num_examples"] = num_examples
                                params_dict["run_no"] = run_no
                                results, prog = self.run_once(params_dict)
                                for result in results:
                                    fname = result[0]
                                    expected_val = result[1]
                                    preds_correct = result[2]
                                    preds_total = result[3]
                                    ind_results.append(
                                        [
                                            params_dict["low"],
                                            params_dict["high"],
                                            params_dict["num_rounds"],
                                            params_dict["preds_per_round"],
                                            params_dict["use_bins"],
                                            params_dict["depth"],
                                            params_dict["use_mi"],
                                            params_dict["mi_pool"],
                                            num_examples,
                                            fname,
                                            expected_val,
                                            preds_correct,
                                            preds_total,
                                            run_no,
                                            prog,
                                            heuristic,
                                        ]
                                    )
                        else:
                            user_examples = set(random.choices(all_example_fnames, k=1))
                            params_dict["examples"] = user_examples
                            params_dict["num_examples"] = 1
                            params_dict["run_no"] = run_no
                            results, prog = self.run_once(params_dict)
                            for result in results:
                                fname = result[0]
                                expected_val = result[1]
                                preds_correct = result[2]
                                preds_total = result[3]
                                ind_results.append(
                                    [
                                        params_dict["low"],
                                        params_dict["high"],
                                        params_dict["num_rounds"],
                                        params_dict["preds_per_round"],
                                        params_dict["use_bins"],
                                        params_dict["depth"],
                                        params_dict["use_mi"],
                                        params_dict["mi_pool"],
                                        1,
                                        fname,
                                        expected_val,
                                        preds_correct,
                                        preds_total,
                                        run_no,
                                        prog,
                                        heuristic,
                                    ]
                                )
                df = pd.DataFrame(
                    columns=self.ind_csv_header + ["heuristic"], data=ind_results
                )
                param_str = self.param_str_noex(params_dict)
            ind_csv_filename = f"{param_str}_heuristic_results.csv"
            df.to_csv(os.path.join(self.out_dir,self.csv_dir,ind_csv_filename))

    def parse_prog(self, fname):
        with open(fname, "r") as f:
            text = f.read()
        prog = text.split("\n")
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

    def parse_cmd(self, cmd):
        if not cmd or cmd[0] == "#":
            return
        terms = cmd.split(" ")
        head = terms[0]
        print(cmd)
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
                print("Invalid command: ", head)

    # syntax: set ctx1 param1 expr1 ... ctxN paramN exprN
    # warning: will evaluate arbitrary code. Don't use this for secure stuff
    def parse_set(self, terms):
        if len(terms) % 3 != 0:
            raise REPLParseError(
                "Invalid SET expression. Please use the form set ctx1 param1 expr1 ... ctxN paramN exprN"
            )
        for i in range(0, len(terms) - 1, 2):
            ctx = terms[i]
            param = terms[i + 1]
            expr = terms[i + 2]
            match ctx:
                case "pp":
                    params = self.pp_params
                case "synth":
                    params = self.synth_params
                case "baseline":
                    params = self.baseline_params
                case "pareto":
                    params = self.pareto_params
                case "ablation":
                    params = self.ablation_params
                case "perf":
                    params = self.perf_params
                case "heuristic":
                    params = self.heuristic_params
                case "task":
                    params = self.task_params
                case _:
                    raise REPLParseError("Invalid context: ", ctx)
            if param not in params:
                raise REPLParseError("Invalid parameter: ", param)
            params[param] = eval(expr)
            if ctx == "task" and param == "task":
                self.construct_lang()
            self.make_dirs()

    def parse_compute(self, terms):
        match terms[0]:
            case "pareto":
                self.compute_pareto()
            case "baseline":
                self.compute_baseline_synth_scores()
            case "ablation":
                self.compute_ablation()
            case "perf":
                self.compute_performance()
            case "heuristic":
                self.compute_heuristics()
            case _:
                raise REPLParseError(f"Invalid compute argument: {terms[0]}")

    def parse_plot(self, terms):
        match terms[0]:
            case "hist":
                self.plot_hists()
            case "pareto":
                self.plot_pareto()
            case "baseline":
                self.plot_baseline()
            case "ablation":
                self.plot_ablation()
            case "perf":
                self.plot_performance()
            case "heuristic":
                self.plot_heuristics()
            case _:
                raise REPLParseError(f"Invalid plot argument: {terms[0]}")

    def parse_select(self, terms):
        def parse_params(terms, params):
            if len(terms) % 2 == 1:
                raise REPLParseError("Invalid select statement")
            for i in range(0, len(terms), 2):
                params[terms[i]] = eval(terms[i + 1])

        match terms[0]:
            case "baseline":
                params = copy.copy(self.baseline_params)
                parse_params(terms[1:], params)
                self.extract_baseline_examples(params)
            case "synth":
                params = copy.copy(self.synth_params)
                parse_params(terms[1:], params)
                self.extract_synth_diff(params)
            case "pareto":
                params = copy.copy(self.pareto_params)
                parse_params(terms[1:], params)
                self.extract_pareto_examples(params)
            case "ablation":
                params = copy.copy(self.ablation_params)

    def extract_baseline_examples(self, params):
        ground_truth_labels = pd.read_csv(self.examples_csv_fname)
        predicted_labels = pd.read_csv(self.baseline_labels_fname)
        baseline_k = predicted_labels[
            predicted_labels["k"] == self.baseline_params["baseline_k"]
        ]
        for params_dict in self.synth_param_iter(params,"low","high","bins","mi","pool","depth"):
            for num_rounds, preds_per_round in self.synth_num_pred_iter(params):
                for k in params["k_vals"]:
                    params_dict["num_examples"] = k
                    csv_fname = self.ind_csv_fname(params_dict)
                    synth_vals = pd.read_csv(csv_fname)
                    synth_k = synth_vals[
                        (synth_vals["num_feature_selection_rounds"] == num_rounds)
                        & (synth_vals["predicates_per_round"] == preds_per_round)
                    ]
                    synth_k["pred"] = synth_k["predicates_correct"] >= 1
                    frac_df = synth_k.groupby("filename")["pred"].mean().reset_index()
                    frac_df.rename(
                        columns={"pred": "frac", "filename": "fname"}, inplace=True
                    )

                    merged_synth_baseline = pd.merge(
                        frac_df,
                        baseline_k,
                        on="fname",
                        suffixes=("_synth", "_baseline"),
                    )
                    merged_synth_baseline_ground = pd.merge(
                        merged_synth_baseline,
                        ground_truth_labels,
                        on="fname",
                        suffixes=("", "_ground"),
                    )
                    diff = merged_synth_baseline[
                        (
                            merged_synth_baseline_ground["frac"]
                            < self.baseline_params["baseline_diff_thresh"]
                        )
                        & (merged_synth_baseline_ground["val"])
                        & (merged_synth_baseline_ground["val_ground"])
                    ]
                    diff_fnames = diff["fname"].tolist()
                    print(diff)

    def plot_pareto(self):
        full_results = pd.read_csv(os.path.join(self.out_dir,self.csv_dir,f"pareto_{self.full_out_csv_filename}"))
        full_results["num_preds"] = (
            full_results["num_feature_selection_rounds"]
            * full_results["predicates_per_round"]
        )
        for param_dict in self.synth_param_iter(self.pareto_params,"low","high","bins","mi","pool","depth"):
            param_results = full_results[
                (full_results["low_threshold"] == param_dict["low"])
                & (full_results["high_threshold"] == param_dict["high"])
                & (full_results["depth"] == param_dict["depth"])
                & (full_results["use_bins"] == param_dict["use_bins"])
                & (full_results["use_mutual_information"] == param_dict["use_mi"])
                & (full_results["mutual_info_pool_size"] == param_dict["mi_pool"])
            ]
            param_dict["f1_thresh"] = self.pp_params["f1_thresh"]
            param_str = self.param_str_noex(param_dict)
            param_str = f"{os.path.join(self.out_dir,self.img_dir,self.pareto_dir,param_str)}_{self.pp_params['f1_thresh']}_{self.pp_params['num_samples']}"
            create_pareto_plot(
                param_results,
                param_dict,
                param_str,
                self.pp_params["pareto_scatterplot"],
            )

    def plot_baseline(self):
        all_synth_scores = self.extract_baseline_exp_examples()
        baseline_scores = self.compute_baseline_scores()
        for params_dict in self.synth_param_iter(self.baseline_params,"low","high","bins","mi","pool","depth"):
            synth_scores = []
            param_str = self.param_str_noex(params_dict)
            synth_scores_params = all_synth_scores[param_str]
            for num_rounds, preds_per_round in self.synth_num_pred_iter(
                self.baseline_params
            ):
                params_dict["num_rounds"] = num_rounds
                params_dict["preds_per_round"] = preds_per_round
                synth_scores_preds = np.array(
                    synth_scores_params[(num_rounds, preds_per_round)]
                )
                synth_scores_means = np.mean(synth_scores_preds, axis=0)
                synth_scores_stds = np.std(synth_scores_preds, axis=0)
                for j in range(synth_scores_preds.shape[0]):
                    for i, num_examples in enumerate(self.baseline_params["examples"]):
                        #params_dict["num_examples"] = num_examples
                        synth_scores.append(
                            [
                                j,
                                num_examples,
                                f"Synthesis with {num_rounds * preds_per_round} decisions",
                                synth_scores_preds[j,i]
                            ]
                        )
            baseline_fname = (
                f"{os.path.join(self.out_dir,self.img_dir,self.baseline_dir,param_str)}_baseline.pdf"
            )
            synth_scores = pd.DataFrame(
                data=synth_scores,
                columns=["run_no","num_examples", "num_preds", "f1"],
            )
        make_baseline_plots(baseline_scores, synth_scores, params_dict, baseline_fname)

    def plot_ablation(self):
        all_synth_scores = self.extract_ablation_results()
        synth_scores = []
        for params_dict in self.synth_param_iter(self.ablation_params,"low","high","bins","mi","pool","depth"):
            param_str = self.param_str_noex(params_dict)
            if self.debug_value:
                print(param_str)
            synth_scores_params = all_synth_scores[param_str]
            for num_rounds, preds_per_round in self.synth_num_pred_iter(
                self.ablation_params
            ):
                params_dict["num_rounds"] = num_rounds
                params_dict["preds_per_round"] = preds_per_round
                synth_scores_preds = np.array(
                    synth_scores_params[(num_rounds, preds_per_round)]
                )
                for j in range(synth_scores_preds.shape[0]):
                    for i, num_examples in enumerate(self.ablation_params["examples"]):
                        params_dict["num_examples"] = num_examples
                        bins_str = "Iterative" if params_dict["use_bins"] else "Direct"
                        mi_str = (
                            "Mutual Info" if params_dict["use_mi"] else "No Mutual Info"
                        )
                        num_preds = num_rounds * preds_per_round
                        synth_scores.append(
                            [
                                num_examples,
                                num_preds,
                                synth_scores_preds[j,i],
                                f"{bins_str}, {mi_str}",
                            ]
                        )
        ablation_fname = os.path.join(self.out_dir,self.img_dir,self.ablation_dir,"ablation.pdf")
        synth_scores = pd.DataFrame(
            data=synth_scores,
            columns=["num_examples", "num_preds", "f1", "label"],
        )
        make_ablation_plots(synth_scores, ablation_fname)

    def plot_heuristics(self):
        all_synth_scores = self.extract_heuristic_results()
        synth_scores = []
        for params_dict in self.synth_param_iter(self.heuristic_params,"low","high","bins","mi","pool","depth"):
            param_str = self.param_str_noex(params_dict)
            if self.debug_value:
                print(param_str)
            synth_scores_params = all_synth_scores[param_str]
            for i, num_examples in enumerate(self.heuristic_params["examples"]):
                params_dict["num_examples"] = num_examples
                synth_scores_preds = np.array(synth_scores_params[num_examples])
                synth_scores_means = np.mean(synth_scores_preds, axis=0)
                synth_scores_stds = np.std(synth_scores_preds, axis=0)
                for j in range(synth_scores_preds.shape[0]):
                    for i, (num_rounds, preds_per_round) in enumerate(
                        self.synth_num_pred_iter(self.heuristic_params)
                    ):
                        num_preds = num_rounds * preds_per_round
                        synth_scores.append(
                            [
                                num_examples,
                                num_preds,
                                synth_scores_preds[j,i],
                                f"Predicate selection algorithm, {num_examples} examples",
                            ]
                        )
            rand_synth_scores_preds = np.array(synth_scores_params["rand"])
            for j in range(rand_synth_scores_preds.shape[0]):
                for i, (num_rounds, preds_per_round) in enumerate(
                    self.synth_num_pred_iter(self.heuristic_params)
                ):
                    num_preds = num_rounds * preds_per_round
                    synth_scores.append(
                        [
                            1,
                            num_preds,
                            rand_synth_scores_preds[j,i],
                            "Random",
                        ]
                    )
        fname = os.path.join(self.out_dir,self.img_dir,self.heuristic_dir,"heuristic_ablation.pdf")
        synth_scores = pd.DataFrame(
            data=synth_scores,
            columns=["num_examples", "num_preds", "f1", "label"],
        )
        make_heuristic_plots(synth_scores, fname)

    def plot_performance(self):
        ind_csv_filename = "perf_results.csv"
        out_fname = os.path.join(self.out_dir,self.img_dir,self.perf_dir,"performance.pdf")
        results = pd.read_csv(os.path.join(self.out_dir,self.csv_dir,ind_csv_filename))
        make_performance_plot(results, out_fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Synthesis CLI", description="CLI for running synthesis"
    )
    parser.add_argument("-f", "--fname", required=False)
    args = parser.parse_args()
    app = App(debug_value=False)
    if not args.fname:
        app.run_repl()
    else:
        app.parse_prog(args.fname)
