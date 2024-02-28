import json
import pandas as pd
import numpy as np
import yaml
from queue import PriorityQueue
import random
from pyparsing import Word, printables, QuotedString, Forward, Group, Suppress, ZeroOrMore, Literal, nums
from lang import *

class Document:
    def __init__(self,fname,features):
        self.features = set(features)
        self.fname = fname

def user_input_yn(prompt,default=True):
    prompt += f"\nEnter y/n. (Default: {'y' if default else 'n' }): "
    response = input(prompt)
    while response not in ['y','n','']:
        response = input("Please enter y or n and try again: ")
    if response == 'y':
        return True
    elif response == 'n':
        return False
    else:
        return default
    
def predGen(features,depth):
    def ZeroDepth():
        for feature in features:
            yield Exists(feature)
    if depth == 0:
        for pred in ZeroDepth():
            yield pred
    else:
        for i, pred1 in enumerate(ZeroDepth()):
            yield pred1
            for j, pred2 in enumerate(ZeroDepth()):
                if i < j:
                    yield AndPred(pred1,pred2)
                    #yield OrPred(pred1, pred2)
    
def mutual_info(pred1,pred2,docs):
    A_count = 0
    B_count = 0
    AB_count = 0
    if pred2 == None:
        for fname in docs:
            doc = docs[fname]
            if pred1.eval(doc):
                A_count += 1
        return A_count / len(docs)
    else:
        for fname in docs:
            doc = docs[fname]
            in_A = pred1.eval(doc)
            in_B = pred2.eval(doc)
            if in_A and in_B:
                AB_count += 1
            if in_A:
                A_count += 1
            if in_B:
                B_count += 1
        total = len(docs)
        p_AB = AB_count / total
        p_A = A_count / total
        p_B = B_count / total
        if (p_AB) == 0:
            return 0
        mi = p_AB * np.log(p_AB / (p_A * p_B))
        return mi

def filter_topk(k,topk,docs):
    combined_pred = None
    top_preds = []
    all_preds = set()
    while not topk.empty():
        item_i = topk.get()
        if item_i[3] < 0:
            all_preds.add(Not(item_i[2]))
        else:
            all_preds.add(item_i[2])
    for _ in range(k):
        mis = {}
        for pred in all_preds:
            mi = mutual_info(pred,combined_pred,docs)
            mis[pred] = mi
        if not combined_pred:
            max_pred = max(mis,key=lambda pred:mis[pred])
        else:
            max_pred = min(mis,key=lambda pred:mis[pred])
        all_preds.remove(max_pred)
        top_preds.append(max_pred)
        if combined_pred:
            combined_pred = OrPred(max_pred,combined_pred)
        else:
            combined_pred = max_pred
    return top_preds

def freq_heuristic(program, docs, features_set,examples, rejected_preds, config):
    
    top_docs = set()
    bottom_docs = set()
    for fname in docs:
        doc = docs[fname]
        doc_score = program.eval(doc)
        if doc_score == 2:
            top_docs.add(fname)
        elif doc_score == 0:
            bottom_docs.add(fname)
    if config.use_mutual_information and len(top_docs) > 0:
        size = config.predicates_per_round * config.mutual_info_pool_size
    else:
        size = config.predicates_per_round
    topk = PriorityQueue(size)
    i = 0
    if config.debug:
        print("Examples:",examples)
    for pred in predGen(features_set,config.depth):
        i += 1
        if i % 100 == 0 and config.debug:
            #print(i)
            pass
        if pred in program or pred in rejected_preds or Not(pred) in program or Not(pred) in rejected_preds:
            continue
        
        if len(program) == 0 or not config.use_bins or (len(top_docs) == 0 or len(bottom_docs) == 0):
            user_true = 0
            nonuser_true = 0
            user_false = 0
            nonuser_false = 0
            for fname in docs:
                doc = docs[fname]
                if pred.eval(doc):
                    if fname in examples:
                        user_true += 1
                    else:
                        nonuser_true += 1
                else:
                    if fname in examples:
                        user_false += 1
                    else:
                        nonuser_false += 1
            if user_true + user_false == 0:
                user_approved = 0
            else:
                user_approved = user_true / (user_true + user_false)
            full_docs = (user_true + nonuser_true) / (user_true + user_false + nonuser_true + nonuser_false)
            score = user_approved - full_docs
        else:
            top_true = 0
            top_false = 0
            mid_true = 0
            mid_false = 0
            bottom_true = 0
            bottom_false = 0
            for fname in docs:
                doc = docs[fname]
                if pred.eval(doc):
                    if fname in top_docs:
                        top_true += 1
                    elif fname in bottom_docs:
                        bottom_true += 1
                    else:
                        mid_true += 1
                else:
                    if fname in top_docs:
                        top_false += 1
                    elif fname in bottom_docs:
                        bottom_false += 1
                    else:
                        mid_false += 1
            top_score = top_true / (top_true + top_false)
            bottom_score = bottom_true / (bottom_true + bottom_false)
            score = top_score - bottom_score
        item_i = (abs(score),i,pred,score)
        if not topk.full():
            topk.put(item_i)
        else:
            item_k = topk.get()
            to_push = max(item_i,item_k,key=lambda item:item[0])
            topk.put(to_push)
    if config.use_mutual_information and len(top_docs) > 0:
        return filter_topk(config.predicates_per_round,topk,docs)
    else:
        preds = []
        while not topk.empty():
            item_i = topk.get()
            if config.debug:
                print(item_i)
            if item_i[3] < 0:
                preds.append(Not(item_i[2]))
            else:
                preds.append(item_i[2])
        return preds

class RunConfig:
    def __init__(self, low_threshold=0.2, high_threshold=0.8, features_fname="features.json", full_csv=None,
                 num_feature_selection_rounds=5, features_combine_count=5,depth=0,mutual_info_pool_size=2,output_threshold=0.5,
                 predicates_per_round=5, use_mutual_information=False,manual=False,use_bins = False, eval = False, debug=False,
                 prog_fname="program.txt",num_examples=10,examples=None):
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

    @classmethod
    def from_yaml(cls, filename):
        with open(filename, 'r') as file:
            config_data = yaml.safe_load(file)
        return cls(**config_data)

    
class Runner:
    def __init__(self,config,heuristic=freq_heuristic):
        self.heuristic = heuristic
        self.config = config
        self.docs = {}
        with open(config.features_fname,'r') as f:
            json_dict = json.loads(f.read())
            for doc in json_dict:
                self.docs[doc] = Document(doc,json_dict[doc])
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
        self.prog = Program(config.low_threshold,config.high_threshold)

        
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
            ref_preds = [parse_pred(s) for s in pred_strs]
        if self.config.debug:
            print(ref_preds)
        rejected_preds = []
        for _ in range(self.config.num_feature_selection_rounds):
            new_preds = self.heuristic(self.prog,self.docs,self.all_features,self.user_examples,rejected_preds,self.config)
            for pred in new_preds:
                if pred in ref_preds:
                    self.prog.add_pred(pred)
                else:
                    rejected_preds.append(pred)

    def runManual(self):
        rejected_preds = []
        prog_complete = user_input_yn(f"Current program is: \n{self.prog}\nAre you satisfied with this?",default = False)
        while not prog_complete:
            new_preds = self.heuristic(self.prog,self.docs,self.all_features,self.user_examples,rejected_preds,self.config)
            new_preds_str = '\n'.join([str(pred) for pred in new_preds])
            add_preds = user_input_yn(f"New predicates generated: \n{new_preds_str}\n. Would you like to add all of these to your program?")
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