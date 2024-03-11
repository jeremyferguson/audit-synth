import numpy as np

def compute_max_f1_scores(results,thresholds) -> float:
    #print(list(map(lambda thresh: compute_metrics_synth(results,thresh),thresholds)))
    f1_scores = list(map(lambda thresh: compute_metrics_synth(results,thresh)[2],thresholds))
    return max(f1_scores)

def compute_metrics_synth(results,pred_threshold) -> tuple[int,int,float]:
    expected_vals = [row[1] for row in results]
    correct_fractions = [0 if row[3] == 0 else float(row[2])/float(row[3]) for row in results]
    #print(correct_fractions)
    return compute_metrics(list(zip(expected_vals,correct_fractions)),lambda frac: frac > pred_threshold)

def compute_metrics_baseline(results):
    return compute_metrics(results,lambda pred: pred)

def compute_metrics(results,eval) -> tuple[int,int,float]:
    true_pos = len(list(filter(lambda t: t[0] and eval(t[1]), results)))
    false_pos = len(list(filter(lambda t: not t[0] and eval(t[1]), results)))
    false_neg = len(list(filter(lambda t: t[0] and not eval(t[1]), results)))
    if true_pos + false_pos == 0:
        precision = 0.0
    else:
        precision = true_pos / (true_pos + false_pos)
    if true_pos + false_neg == 0:
        recall = 0.0
    else:
        recall = true_pos / (true_pos + false_neg)
    if precision + recall < 1e-5:
        f1 = 0.0
    else:
        f1 = (2.0*precision*recall) / (precision + recall)
    return precision, recall, f1

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