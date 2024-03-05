import json
import pandas as pd
import numpy as np
import yaml
import random
from lang import ImgLang, MusicLang
from music_lang import parse_music_pred, music_freq_heuristic
from utils import user_input_yn


class RunConfig:
    def __init__(self, low_threshold=0.2, high_threshold=0.8, features_fname="features.json", full_csv=None,
                 num_feature_selection_rounds=5, features_combine_count=5,depth=0,mutual_info_pool_size=2,output_threshold=0.5,
                 predicates_per_round=5, use_mutual_information=False,manual=False,use_bins = False, eval = False, debug=False,
                 prog_fname="program.txt",num_examples=10,examples=None,lang=None,task="image"):
        # Validate thresholds
        assert 0 <= low_threshold <= 1, "Low threshold must be between 0 and 1."
        assert 0 <= high_threshold <= 1, "High threshold must be between 0 and 1."
        assert low_threshold < high_threshold, "Low threshold must be less than high threshold."

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.features_fname = features_fname
        self.prog_fname = prog_fname 
        self.num_feature_selection_rounds = num_feature_selection_rounds
        self.features_combine_count = features_combine_count
        self.predicates_per_round = predicates_per_round
        self.use_mutual_information = use_mutual_information
        self.manual = manual
        self.depth = depth
        self.use_bins = use_bins
        self.eval = eval
        self.full_csv = full_csv
        self.output_threshold = output_threshold
        self.mutual_info_pool_size = mutual_info_pool_size
        self.debug = debug
        self.num_examples=num_examples
        self.examples=examples
        self.lang = lang or ImgLang() if task == "image" else MusicLang()

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    
class Runner:
    def __init__(self,config):
        self.lang = config.lang
        self.config = config
        self.docs = {}
        with open(config.features_fname,'r') as f:
            json_dict = json.loads(f.read())
            for doc in json_dict:
                self.docs[doc] = self.lang.Document(doc,json_dict[doc])
        self.all_features = set()
        for fname in self.docs:
            for feature in self.docs[fname].features:
                self.all_features.add(feature)

        if not config.examples:
            all_examples = pd.read_csv(config.full_csv)
            all_example_fnames = list(all_examples[all_examples['val']==True]['fname'])
            self.user_examples = set(random.choices(all_example_fnames,k=config.num_examples))
        else:
            self.user_examples = set(config.examples)
        self.prog = self.lang.Program(config.low_threshold,config.high_threshold)

        
    def run(self):
        if self.config.manual:
            self.runManual()
        else:
            self.runAutomatic()
        return self.prog
    
    def eval(self):
        ground_truth = pd.read_csv(self.config.full_csv)
        output = []
        for fname in self.docs:
            if fname in ground_truth['fname'].values:
                expected_val = ground_truth[ground_truth['fname'] == fname]['val'].array[0]
                preds_correct,total_preds = self.prog.eval(self.docs[fname],False)
                output.append([fname,expected_val,preds_correct,total_preds])
        return output
    
    def runAutomatic(self):
        with open(self.config.prog_fname,"r") as f:
            text = f.read()
            pred_strs = text.split("\n")
            ref_preds = [self.lang.parse_pred(s) for s in pred_strs]
        if self.config.debug:
            print(ref_preds)
        rejected_preds = []
        for _ in range(self.config.num_feature_selection_rounds):
            new_preds = self.lang.heuristic(self.prog,self.docs,self.all_features,self.user_examples,rejected_preds,self.config)
            for pred in new_preds:
                if pred in ref_preds:
                    self.prog.add_pred(pred)
                else:
                    rejected_preds.append(pred)

    def runManual(self):
        rejected_preds = []
        prog_complete = user_input_yn(f"Current program is: \n{self.prog}\nAre you satisfied with this?",default = False)
        while not prog_complete:
            new_preds = self.lang.heuristic(self.prog,self.docs,self.all_features,self.user_examples,rejected_preds,self.config)
            print("New predicates generated: ")
            for pred in new_preds:
                self.lang.displayPred(pred)
            add_preds = user_input_yn(f"Would you like to add all of these to your program?")
            if add_preds:
                for pred in new_preds:
                    self.prog.add_pred(pred)
            else:
                add_any = user_input_yn(f"Would you like to add any of these to your program?")
                if add_any:
                    for pred in new_preds:
                        add_pred = user_input_yn(f"Would you like to add this predicate to your program: {pred}")
                        if add_pred:
                            prog.add_pred(pred)
                        else:
                            rejected_preds.append(pred)
                else:
                    for pred in new_preds:
                        rejected_preds.append(pred)
            prog_complete = user_input_yn(f"Current program is {self.prog}. Are you satisfied with this?",default = False)
            
yaml_filename = "config.yml"

if __name__ == "__main__":
    config = RunConfig.from_yaml(yaml_filename)
    runner = Runner(config)
    prog = runner.run()
    print("Final program: ",prog)
    if config.eval:
        runner.eval()