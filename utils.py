import numpy as np

def compute_max_f1_scores(results,thresholds):
    f1_scores = map(lambda thresh: compute_metrics_synth(results,thresh)[2],thresholds)
    return max(f1_scores)

def compute_metrics_synth(results,pred_threshold):
    expected_vals = [row[1] for row in results]
    correct_fractions = [0 if row[3] == 0 else row[2]/row[3] for row in results]
    return compute_metrics(zip(expected_vals,correct_fractions),lambda frac: frac > pred_threshold)

def compute_metrics_baseline(results):
    return compute_metrics(results,lambda pred: pred)

def compute_metrics(results,eval):
    true_pos = len(filter(lambda expected, pred: expected and eval(pred), results))
    false_pos = len(filter(lambda expected, pred: not expected and eval(pred), results))
    false_neg = len(filter(lambda expected, pred: expected and not eval(pred), results))
    if true_pos + false_pos == 0:
        precision = 0
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos + false_neg == 0:
        recall = 0
    else:
        recall = true_pos / (true_pos + false_neg)
    if precision + recall < 1e-5:
        f1 = 0
    else:
        f1 = (2*precision*recall) / (precision + recall)
    return precision, recall, f1