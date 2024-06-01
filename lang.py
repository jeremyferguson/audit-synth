from pyparsing import *
from queue import PriorityQueue
import numpy as np
from util_scripts.write_midis import create_midi
import os
import re


import random
import time

if os.environ["RUN_MUSIC_TASK"] == 1:
    import pygame
    pygame.init()
    pygame.mixer.init(frequency=44100, size=-16, channels=1)


class Lang:
    def __init__(self):
        self.preds = {}

    class Program:
        def __init__(self, low_thresh, high_thresh, preds=None):
            if not preds:
                self.preds = []
            else:
                self.preds = preds
            self.low_thresh = low_thresh
            self.high_thresh = high_thresh

        def add_pred(self, pred):
            self.preds.append(pred)

        def eval(self, doc, use_bins=True):
            score = 0
            if self.preds:
                for pred in self.preds:
                    if pred.eval(doc):
                        score += 1
                fraction = score / len(self.preds)
                if use_bins:
                    if fraction >= self.high_thresh:
                        return 2
                    elif fraction >= self.low_thresh:
                        return 1
                    else:
                        return 0
                else:
                    return score, len(self.preds)
            else:
                return 0, 0

        def __repr__(self):
            if not self.preds:
                return "Empty"
            return "\n".join([str(pred) for pred in self.preds])

        def __len__(self):
            return len(self.preds)

        def __contains__(self, pred):
            return pred in self.preds

    def mutual_info(self, pred1, pred2, docs):
        A_count = 0
        B_count = 0
        AB_count = 0
        if pred2 is None:
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
            mi = p_AB * np.log2(p_AB / (p_A * p_B))
            return mi

    def filter_topk(self, k, topk, docs,use_not=True):
        combined_pred = None
        top_preds = []
        all_preds = set()
        i = 0
        while not topk.empty():
            item_i = topk.get()
            i += 1
            if item_i[3] < 0 and use_not:
                all_preds.add(self.Not(item_i[2]))
            else:
                all_preds.add(item_i[2])
        for _ in range(k):
            mis = {}
            for pred in all_preds:
                mi = self.mutual_info(pred, combined_pred, docs)
                mis[pred] = mi
            if not mis:
                break
            # max_pred = max(mis,key=lambda pred:mis[pred])
            if not combined_pred:
                max_pred = max(mis, key=lambda pred: mis[pred])
            else:
                max_pred = min(mis, key=lambda pred: mis[pred])
            all_preds.remove(max_pred)
            top_preds.append(max_pred)
            if combined_pred:
                combined_pred = self.OrPred(max_pred, combined_pred)
            else:
                combined_pred = max_pred
        return top_preds

    def freq_heuristic(
        self, program, docs, features_set, examples, rejected_preds, config
    ):
        top_docs = set()
        bottom_docs = set()
        for fname in docs:
            doc = docs[fname]
            doc_score = program.eval(doc)
            if doc_score == 2:
                top_docs.add(fname)
            elif doc_score == 0:
                bottom_docs.add(fname)
        if config.debug:
            print(len(bottom_docs), len(top_docs))
        if config.use_mutual_information and len(top_docs) > 0:
            size = config.predicates_per_round * config.mutual_info_pool_size
        else:
            size = config.predicates_per_round
        topk = PriorityQueue(size)
        i = 0
        if config.debug:
            print("Examples:", examples)
        for pred in self.predGen(docs, features_set, config.depth):
            i += 1
            if i % 100 == 0 and config.debug:
                # print(i)
                pass
            if (
                pred in program
                or pred in rejected_preds
                or self.Not(pred) in program
                or self.Not(pred) in rejected_preds
            ):
                continue

            if (
                len(program) == 0
                or not config.use_bins
                or (len(top_docs) == 0 or len(bottom_docs) == 0)
            ):
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
                full_docs = (user_true + nonuser_true) / (
                    user_true + user_false + nonuser_true + nonuser_false
                )
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
            item_i = (abs(score), i, pred, score)
            if not topk.full():
                topk.put(item_i)
            else:
                item_k = topk.get()
                to_push = max(item_i, item_k, key=lambda item: item[0])
                topk.put(to_push)

        if config.use_mutual_information and len(top_docs) > 0:
            return self.filter_topk(config.predicates_per_round, topk, docs, config.use_not)
        else:
            preds = []
            while not topk.empty():
                item_i = topk.get()
                if config.debug:
                    print(item_i)
                if item_i[3] < 0 and config.use_not:
                    preds.append(self.Not(item_i[2]))
                else:
                    preds.append(item_i[2])
            return preds

    def rand_heuristic(
        self, program, docs, features_set, examples, rejected_preds, config
    ):
        return random.sample(
            self.predGen(docs, features_set, config.depth),
            config.predicates_per_round
        )


class ImgLang(Lang):
    class Document:
        def __init__(self, fname, features):
            self.features = set(features)
            self.fname = fname

    class Pred:
        def __repr__(self):
            pass

        def eval(self, doc):
            pass

    class Exists(Pred):
        def __init__(self, feature: str):
            self.feature = feature

        def eval(self, doc):
            return self.feature in doc.features

        def __repr__(self):
            return f'Exists("{self.feature}")'

        def __eq__(self, other):
            return isinstance(other, ImgLang.Exists) and\
                self.feature == other.feature

        def __hash__(self):
            return hash(("Exists", self.feature))

    class Not(Pred):
        def __init__(self, pred):
            self.pred = pred

        def eval(self, doc):
            return not self.pred.eval(doc)

        def __repr__(self):
            return f"Not({self.pred})"

        def __eq__(self, other):
            return isinstance(other, ImgLang.Not) and self.pred == other.pred

        def __hash__(self):
            return hash(("Not", self.pred))

    class AndPred(Pred):
        def __init__(self, pred1, pred2):
            self.pred1 = pred1
            self.pred2 = pred2

        def eval(self, doc):
            return self.pred1.eval(doc) and self.pred2.eval(doc)

        def __repr__(self):
            return f"And({self.pred1},{self.pred2})"

        def __eq__(self, other):
            return isinstance(other, ImgLang.AndPred) and (
                (self.pred1 == other.pred1 and self.pred2 == other.pred2)
                or (self.pred2 == other.pred1 and self.pred1 == other.pred2)
            )

        def __hash__(self):
            return hash(("And", self.pred1, self.pred2))

    class OrPred(Pred):
        def __init__(self, pred1, pred2):
            self.pred1 = pred1
            self.pred2 = pred2

        def eval(self, doc):
            return self.pred1.eval(doc) or self.pred2.eval(doc)

        def __repr__(self):
            return f"Or({self.pred1},{self.pred2})"

        def __eq__(self, other):
            return isinstance(other, ImgLang.OrPred) and (
                (self.pred1 == other.pred1 and self.pred2 == other.pred2)
                or (self.pred2 == other.pred1 and self.pred1 == other.pred2)
            )

        def __hash__(self):
            return hash(("Or", self.pred1, self.pred2))

    class MultiPred(Pred):
        def __init__(self, preds, thresh):
            self.preds = preds
            self.thresh = thresh

        def eval(self, doc):
            score = 0
            for pred in self.preds:
                if pred.eval(doc):
                    score += 1
            return score >= score.thresh

        def __repr__(self):
            return f"Multi([{','.join(self.preds)}],{self.thresh})"

    def parse_pred(self, input_str):
        def parse_exists(tokens):
            return ImgLang.Exists(tokens[1])

        def parse_not(tokens):
            return ImgLang.Not(tokens[1])

        def parse_and(tokens):
            return ImgLang.AndPred(tokens[1], tokens[2])

        def parse_or(tokens):
            return ImgLang.OrPred(tokens[1], tokens[2])

        def parse_multi(tokens):
            return ImgLang.MultiPred(tokens[0][0], int(tokens[0][1]))

        identifier = QuotedString('"', multiline=True, unquoteResults=True) | Word(
            printables
        )
        exists_parser = Literal("Exists") + Suppress("(") + identifier + Suppress(")")
        not_parser = Forward()
        and_parser = Forward()
        or_parser = Forward()
        multi_parser = Forward()

        exists_parser.setParseAction(parse_exists)
        not_parser.setParseAction(parse_not)
        and_parser.setParseAction(parse_and)
        or_parser.setParseAction(parse_or)
        multi_parser.setParseAction(parse_multi)

        pred_parser = (
            exists_parser | not_parser | and_parser | or_parser
        )  # | multi_parser
        not_parser <<= Literal("Not") + Suppress("(") + pred_parser + Suppress(")")
        and_parser <<= (
            Literal("And")
            + Suppress("(")
            + pred_parser
            + Suppress(",")
            + pred_parser
            + Suppress(")")
        )
        or_parser <<= (
            Literal("Or")
            + Suppress("(")
            + pred_parser
            + Suppress(",")
            + pred_parser
            + Suppress(")")
        )
        # multi_parser <<= Literal("Multi(") + Group(ZeroOrMore(pred_parser + Suppress(","))) + Literal(",") + Word(nums)

        return pred_parser.parseString(input_str, parseAll=True)[0]

    def predGen(self, docs, features, depth):
        def genPredsLevel(depth):
            if depth in self.preds:
                return self.preds[depth]
            elif depth == 0:
                levelPreds = [ImgLang.Exists(feature) for feature in features]
                self.preds[0] = levelPreds
                return levelPreds
            elif depth == 1:
                levelPreds = set()
                for fname in docs:
                    doc = docs[fname]
                    for i, f1 in enumerate(list(doc.features)):
                        for j, f2 in enumerate(list(doc.features)):
                            if i < j:
                                pred = ImgLang.AndPred(
                                    ImgLang.Exists(f1), ImgLang.Exists(f2)
                                )
                                invPred = ImgLang.Not(ImgLang.AndPred(
                                    ImgLang.Exists(f2), ImgLang.Exists(f1)
                                ))
                                if pred not in levelPreds and\
                                        invPred not in levelPreds:
                                    levelPreds.add(pred)
                levelPreds = list(levelPreds)
                self.preds[1] = levelPreds
                return self.preds[1]

        return genPredsLevel(depth)

    def displayPred(self, pred):
        print(pred)


class Chord:
    root_to_num = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    def __init__(self, s):
        self.root = s[: s.find("_")]
        self.quality = s[s.find("_") + 1:]

    def __sub__(self, other):
        if not isinstance(other, Chord):
            raise TypeError(f"Cannot subtract Chord and {type(other)}")
        return (Chord.root_to_num[self.root] - Chord.root_to_num[other.root])\
            % 12


class MusicLang(Lang):
    class Document:
        def __init__(self, fname, features):
            self.features = [Chord(feature) for feature in features]
            self.fname = fname
            self.key = self.features[0].root

    class Pred:
        def __repr__(self):
            pass

        def eval(self, doc):
            pass

    class BasePred:

        def __init__(self,seq_len):
            self.seq_len = seq_len

        def evalAt(self, doc, i):
            if i < 0 or i >= len(doc.features) - self.seq_len:
                return False
            return self.matches(doc,i)
    
        def matches(self,doc,i):
            pass

    class Not(Pred):
        def __init__(self, pred):
            self.pred = pred
            self.preds = self.pred.preds

        def eval(self, doc):
            return not self.pred.eval(doc)

        def __repr__(self):
            return f"Not({self.pred})"

        def __eq__(self, other):
            return isinstance(other, MusicLang.Not) and self.pred == other.pred

        def __hash__(self):
            return hash(("Not", self.pred))

        def chords(self):
            return self.pred.chords()

        def __len__(self):
            return len(self.pred)

    class OrPred(Pred):
        def __init__(self, pred1, pred2):
            self.pred1 = pred1
            self.pred2 = pred2

        def eval(self, doc):
            return self.pred1.eval(doc) or self.pred2.eval(doc)

        def __repr__(self):
            return f"Or({self.pred1},{self.pred2})"

        def __eq__(self, other):
            return isinstance(other, MusicLang.OrPred) and (
                (self.pred1 == other.pred1 and self.pred2 == other.pred2)
                or (self.pred2 == other.pred1 and self.pred1 == other.pred2)
            )

        def __hash__(self):
            return hash(("Or", self.pred1, self.pred2))

    class Stepwise(BasePred):
        def __repr__(self):
            return f"Stepwise({self.difference})"

        def __init__(self, difference):
            super().__init__(2)
            self.difference = difference
            match self.difference:
                case 1:
                    self.chords = ["C#_M", "C_M"]
                case 2:
                    self.chords = ["D_M", "C_M"]
                case 11:
                    self.chords = ["C_M", "C#_M"]
                case 10:
                    self.chords = ["C_M", "D_M"]

        def matches(self, doc, i):
            return doc.features[i + 1] - doc.features[i] == self.difference
        
        def __hash__(self):
            return hash(f"Stepwise({self.difference})")
        
        def __eq__(self,other):
            return isinstance(other,MusicLang.Stepwise) and other.difference == self.difference
        

    class BasicAuthenticCadence(BasePred):
        def __repr__(self):
            return "Authentic Cadence"

        def __init__(self):
            super().__init__(3)

        chords = ["F_M", "G_M", "C_M"]

        def matches(self, doc, i):
            diff1 = doc.features[i + 1] - doc.features[i]
            diff2 = doc.features[i + 2] - doc.features[i + 1]
            return (diff1 == 2 or diff1 == 5) and doc.features[i + 1].quality == "M" and diff2 == 7 and doc.features[i + 2].quality == "M"

        def __hash__(self):
            return hash("AuthenticCadence")
        
        def __eq__(self,other):
            return isinstance(other,MusicLang.BasicAuthenticCadence)
        
    class PlagalMotion(BasePred):
        def __repr__(self):
            return "Plagal"

        chords = ["F_M", "C_M"]

        def __init__(self):
            super().__init__(2)

        def matches(self, doc,i):
            return (doc.features[i + 1] - doc.features[i]) == 5
        
        def __hash__(self):
            return hash("Plagal")
        
        def __eq__(self,other):
            return isinstance(other,MusicLang.PlagalMotion)

    class MinorPlagalMotion(BasePred):
        def __repr__(self):
            return "Minor Plagal"

        chords = ["F_m", "C_M"]
        
        def __init__(self):
            super().__init__(2)

        def matches(self, doc, i):
            return (
                (doc.features[i + 1] - doc.features[i]) == 5
                and doc.features[i].quality == "m"
                and doc.features[i + 1].quality == "M"
            )

        def __hash__(self):
            return hash("MinorPlagal")
        
        def __eq__(self,other):
            return isinstance(other,MusicLang.MinorPlagalMotion)
        
    class Galant(BasePred):
        def __repr__(self):
            return "Galant"

        chords = ["C_M", "G_M", "A_m", "E_m", "F_M", "C_M"]
        
        def __init__(self):
            super().__init__(6)

        def matches(self, doc,i):
            jumps_expected = [7, 2, 7, 1, 7]
            qualities_expected = ["M", "M", "m", "m", "M", "M"]
            jumps_actual = [
                doc.features[j] - doc.features[j - 1] for j in range(i + 1, i + 6)
            ]
            qualities_actual = [doc.features[j].quality for j in range(i, i + 6)]
            return (
                    jumps_actual == jumps_expected
                    and qualities_actual == qualities_expected
                )
        
        def __hash__(self):
            return hash("Galant")
        
        def __eq__(self,other):
            return isinstance(other,MusicLang.Galant)
        
    class SeqEventually(Pred):
        def __init__(self,preds):
            self.preds = preds

        def __len__(self):
            return len(self.pred)
        def __repr__(self):
            return f"SeqEventually({self.preds})"
        
        def eval(self,doc):
            current_pred_index = 0
            for i in range(len(doc.features)):
                if self.preds[current_pred_index].evalAt(doc,i):
                    current_pred_index += 1
                if current_pred_index >= len(self.preds):
                    return True
            return False
        
        def __hash__(self):
            hashes = []
            for pred in self.preds:
                hashes.append(hash(pred))
            return hash(f"Eventually{''.join([str(h) for h in hashes])}")
        
        def __eq__(self,other):
            return isinstance(other,MusicLang.SeqEventually) and other.preds == self.preds
        
        def __len__(self):
            return len(self.preds)

    class SeqUnordered(Pred):
        def __init__(self,preds):
            self.preds = preds

        def __repr__(self):
            return f"SeqUnordered({self.preds})"

        def eval(self,doc):
            preds_to_find = self.preds[:]
            for i in range(len(doc.features)):
                for pred in preds_to_find:
                    if pred.evalAt(doc,i):
                        preds_to_find.remove(pred)
                        continue
                if not preds_to_find:
                    return True
            return False
        
        def __hash__(self):
            hashes = []
            for pred in self.preds:
                hashes.append(hash(pred))
            hashes.sort()
            h = hash(f"Unordered{''.join([str(h) for h in hashes])}")
            return h
        
        def __eq__(self,other):
            if not isinstance(other,MusicLang.SeqUnordered):
                return True
            from collections import Counter
            self_occurrences = Counter(self.preds)
            other_occurrences = Counter(other.preds)
            return self_occurrences == other_occurrences
        
        def __len__(self):
            return len(self.preds)

    class SeqNearby(Pred):
        def __init__(self,preds):
            self.preds = preds

        def __repr__(self):
            return f"SeqNearby({self.preds})"

        def eval(self,doc):
            for i in range(len(doc.features)-20):
                current_pred_index=0
                for j in range(i,i+20):
                    if self.preds[current_pred_index].evalAt(doc,j):
                        current_pred_index+=1
                    if current_pred_index >= len(self.preds):
                        return True
            return False
    
        def __hash__(self):
            hashes = []
            for pred in self.preds:
                hashes.append(hash(pred))
            return hash(f"Nearby{''.join([str(h) for h in hashes])}")

        def __eq__(self,other):
            return isinstance(other,MusicLang.SeqNearby) and self.preds == other.preds
        
        def __len__(self):
            return len(self.preds)
        
    def playPred(self, pred):
        #print(pred)
        cwd = os.getcwd()
        create_midi(pred.chords, "tmp", cwd)
        midi_path = os.path.join(cwd, "tmp.mid")
        pygame.mixer.music.load(midi_path, namehint="midi")
        pygame.mixer.music.play()
        time.sleep(len(pred.chords)*0.5 + 0.5)
    
    def displayPred(self,pred):
        print(pred)
        for basePred in pred.preds:
            self.playPred(basePred)
            time.sleep(0.5)

    def parse_base_pred(self, input_str):
        match input_str:
            case "PlagalMotion":
                return MusicLang.PlagalMotion()
            case "BasicCadence":
                return MusicLang.BasicAuthenticCadence()
            case "Galant":
                return MusicLang.Galant()
            case "MinorPlagalMotion":
                return MusicLang.MinorPlagalMotion()
            case _:
                if input_str[:8] == "Stepwise":
                    return MusicLang.Stepwise(int(input_str[9:-1]))
                else:
                    raise ValueError(f"Invalid input string: {input_str}")

    def parse_pred(self,input_str):
        unordered_pat = r"SeqUnordered\((\S+)\)"
        eventually_pat = r"SeqEventually\((\S+)\)"
        nearby_pat = r"SeqNearby\((\S+)\)"
        not_pat = r"Not\((\S+)\)"
        match = re.match(unordered_pat,input_str)
        if match:
            return MusicLang.SeqUnordered([self.parse_base_pred(pred) for pred in match.group(1).split(",")])
        match = re.match(eventually_pat,input_str)
        if match:
            return MusicLang.SeqEventually([self.parse_base_pred(pred) for pred in match.group(1).split(",")])
        match = re.match(nearby_pat,input_str)
        if match:
            return MusicLang.SeqNearby([self.parse_base_pred(pred) for pred in match.group(1).split(",")])
        match = re.match(not_pat,input_str)
        if match:
            return MusicLang.Not(self.parse_pred(match.group(1)))

    def predGen(self, doc, features, depth, prog_len=None):
        def genPredsLevel(depth):
            if depth in self.preds:
                return self.preds[depth]
            elif depth == 0:
                levelPreds = [MusicLang.SeqUnordered([pred]) for pred in
                [
                    MusicLang.Galant(),
                    MusicLang.MinorPlagalMotion(),
                    MusicLang.PlagalMotion(),
                    MusicLang.BasicAuthenticCadence(),
                ] + [MusicLang.Stepwise(i) for i in [1, 2, 10, 11]]]
                self.preds[0] = levelPreds
                return levelPreds
            else:
                basePreds = genPredsLevel(0)
                n_1_preds = genPredsLevel(depth-1)
                levelPreds = set()
                for pred1 in n_1_preds:
                    for pred2 in basePreds:
                        new_preds = pred1.preds[:] + pred2.preds[:]
                        levelPreds.add(MusicLang.SeqEventually(new_preds))
                        levelPreds.add(MusicLang.SeqNearby(new_preds))
                        levelPreds.add(MusicLang.SeqUnordered(new_preds))
                levelPreds = list(levelPreds)
                self.preds[depth] = levelPreds
                return self.preds[depth]
        preds = []
        for d in range(depth+1):
            preds = preds + genPredsLevel(d)
        return preds
        
