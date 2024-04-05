import pandas as pd
import matplotlib
matplotlib.use('Agg')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = ['serif']
print(plt.rcParams['font.serif'])
plt.rcParams['font.serif'] = 'Times New Roman'
title_size = 50
ax_size = 46
tick_size = 42
fig_size = (14, 12)
def create_hist(df, params_dict, param_str,style):
    plt.figure(figsize=(10, 6))
    predicates_correct = list(df['predicates_correct'])
    bins=range(min(predicates_correct), max(predicates_correct) + 2)
    if style == "stacked":
        sns.histplot(data=df, x='predicates_correct', hue='expected_value',
                bins=bins,
                multiple="stack", palette="viridis")
    elif style == 'facet':
        g = sns.FacetGrid(df, row='expected_value', height=6, aspect=2)
        g.map(plt.hist, 'predicates_correct', bins=bins, color="skyblue", alpha=0.7)

        # Customize plot
        g.set_axis_labels("Predicates Correct", "Frequency")
        g.set_titles(row_template="Ground Truth: {row_name}")
    else:
        raise ValueError(f"Invalid histogram style: {style}")
    bins_str = "bins" if params_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {params_dict['mi_pool']*params_dict['preds_per_round']} preds" if params_dict['use_mi'] else "no mutual info"
    plt.title(f"Histogram for Parameter Combination: Thresholds {params_dict['low']} - {params_dict['high']}, " + \
              f"{params_dict['num_rounds']} x {params_dict['preds_per_round']} preds,{bins_str}, depth {params_dict['depth']}, {mi_str}")
    plt.xlabel("Predicates Correct")
    plt.ylabel("Frequency")
    plt.legend(title="Expected Value", labels=["False", "True"])
    plt.savefig(param_str)
    plt.close()

def create_pareto_plot(results,param_dict,param_str,scatterplot=True):
    columns = results.groupby('num_examples')
    means = []
    stds = []
    timeouts = []
    xs = []
    all_vals = []
    timeout_nos = []
    complete_nos = []
    all_xs = []
    for num_examples, column in columns:
        all_xs.append(num_examples)
        runs = column.groupby('run_no')
        min_values_run_no = []
        timeout = 0
        complete = 0
        for _, run in runs:
            run_filtered = run[run['f1_above_thresh']]
            if run_filtered.empty:
                timeouts.append((num_examples,run['num_preds'].max()))
                timeout += 1
            else:
                complete += 1
                min_values_run_no.append(run_filtered['num_preds'].min())
                all_vals.append((num_examples,run_filtered['num_preds'].min()))
        timeout_nos.append(timeout)
        complete_nos.append(complete)
        if min_values_run_no:
            means.append(np.mean(min_values_run_no))
            stds.append(np.std(min_values_run_no))
            xs.append(num_examples)
    percentage_complete = [complete / (complete+timeout) for complete,timeout in zip(complete_nos,timeout_nos)]
    
    print(xs)
    print(percentage_complete)
    print(means)
    all_vals = np.array(all_vals)
    timeouts_df = pd.DataFrame(columns=['col','y'],data=timeouts)
    plt.figure(figsize=(10, 6)) 
    plt.xticks(range(0,22,2))
    plt.yticks(range(0,80,20))
    plt.xlim(min(all_xs)-1,max(all_xs)+1)
    #sns.lineplot(x=xs, y=means,color='b',label = "Successful Runs")
    if scatterplot and len(all_vals) > 0:
        sns.plot(x=all_vals[:,:1],y=all_vals[:,1:])
    plt.errorbar(x=xs, y=means, yerr=stds, fmt='b', capsize=5,label = "Successful Runs")
    #plot = sns.stripplot(data=timeouts_df,x='col',y='y',color='r',jitter=1.0,marker='^',label="Timeouts")
    plt.ylim(0,80)
    plt.legend()
    plt.xlabel('Number of Examples')
    plt.ylabel(f'Programmer Decisions')
    bins_str = "bins" if param_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {param_dict['mi_pool']}k preds" if param_dict['use_mi'] else "no mutual info"
    plt.title(f"Programmer Decisions vs Number of Examples ")
    plt.grid(True)
    plt.savefig(f"{param_str}_pareto.png")
    plt.clf()
    sns.lineplot(x=all_xs,y=percentage_complete,label = "Complete Runs",marker='o')
    plt.ylim(-0.05,1.05)
    plt.yticks([0.0,0.25,0.5,0.75,1.0])
    plt.xticks(range(0,22,2))
    plt.title(f"Synthesis Completion Percentage vs Number of Examples")
    plt.ylabel('Fraction of Complete Runs')
    plt.xlabel('Number of Examples')
    plt.savefig(f"{param_str}_bar.png")
    plt.close()

def create_highk_plot(results,param_dict,param_str,scatterplot=True):
    columns = results.groupby('num_examples')
    means = []
    stds = []
    timeouts = []
    xs = []
    all_vals = []
    timeout_nos = []
    complete_nos = []
    all_xs = []
    for num_examples, column in columns:
        all_xs.append(num_examples)
        runs = column.groupby('run_no')
        min_values_run_no = []
        timeout = 0
        complete = 0
        for _, run in runs:
            run_filtered = run[run['f1_above_thresh']]
            if run_filtered.empty:
                timeouts.append((num_examples,run['num_preds'].max()))
                timeout += 1
            else:
                complete += 1
                min_values_run_no.append(run_filtered['num_preds'].min())
                all_vals.append((num_examples,run_filtered['num_preds'].min()))
        timeout_nos.append(timeout)
        complete_nos.append(complete)
        if min_values_run_no:
            means.append(np.mean(min_values_run_no))
            stds.append(np.std(min_values_run_no))
            xs.append(num_examples)
    percentage_complete = [complete / (complete+timeout) for complete,timeout in zip(complete_nos,timeout_nos)]
    print(xs)
    print(percentage_complete)
    print(means)
    lwidth = 4.0
    all_vals = np.array(all_vals)
    timeouts_df = pd.DataFrame(columns=['col','y'],data=timeouts)
    plt.figure(figsize=(14, 12)) 
    plt.xticks(range(0,330,50),font={'size':tick_size})
    plt.yticks(range(0,35,5),font={'size':tick_size})
    plt.xlim(0,310)
    #sns.lineplot(x=xs, y=means,color='b',label = "Successful Runs")
    if scatterplot and len(all_vals) > 0:
        sns.plot(x=all_vals[:,:1],y=all_vals[:,1:],linewidth=lwidth)
    plt.errorbar(x=xs, y=means, yerr=stds, fmt='b', capsize=10,linewidth=lwidth)
    #plot = sns.stripplot(data=timeouts_df,x='col',y='y',color='r',jitter=1.0,marker='^',label="Timeouts")
    plt.ylim(0,35)
    #plt.legend()
    plt.xlabel('Number of Examples',fontdict={'size':ax_size})
    plt.ylabel(f'Programmer Decisions',fontdict={'size':ax_size})
    bins_str = "bins" if param_dict['use_bins'] else "no bins"
    mi_str = f"mutual info with {param_dict['mi_pool']}k preds" if param_dict['use_mi'] else "no mutual info"
    plt.title(f"Programmer Decisions vs Number of Examples ",fontdict={'size':title_size})
    plt.grid(True)
    plt.savefig(f"{param_str}_pareto.png")
    plt.clf()
    plt.figure(figsize=(15, 12)) 
    sns.lineplot(x=all_xs,y=percentage_complete,marker='o',linewidth=lwidth)
    plt.ylim(-0.05,1.05)
    plt.xlim(0,310)
    plt.yticks([0.0,0.25,0.5,0.75,1.0],font={'size':tick_size})
    plt.xticks(range(0,330,50),font={'size':tick_size})
    plt.title(f"Fraction of Runs that Reach 0.8 F1 Score",fontdict={'size':title_size})
    plt.ylabel('Fraction',fontdict={'size':ax_size})
    plt.xlabel('Number of Examples',fontdict={'size':ax_size})
    #plt.legend(loc='lower right')
    plt.savefig(f"{param_str}_bar.png")
    plt.close()

def make_baseline_plots(baseline_scores,synth_scores:pd.DataFrame,params_dict,fname):
    grouped = synth_scores.groupby('num_preds')
    plt.figure(figsize=(15,12)) 
    lwidth = 3.0
    labels = []
    colors = ['blue','blueviolet','limegreen','turquoise']
    max_baseline_score = max(baseline_scores['f1'])
    baseline_scores['f1'] = max_baseline_score
    merged_df = pd.merge(synth_scores, baseline_scores, on='num_examples', how='left')
    merged_df['f1'] = merged_df['f1'].fillna(max_baseline_score)
    for i, (num_preds, score_group) in enumerate(grouped):
        sns.lineplot(data=score_group,x="num_examples",y="f1_mean",color=colors[i%4],label=f"Synthesis with {num_preds} User Decisions",linewidth=lwidth)
        plt.errorbar(data=score_group,x="num_examples",y="f1_mean",yerr="f1_std",fmt=colors[i%4],label='_',linewidth=lwidth)
    sns.lineplot(data=merged_df,x='num_examples',y='f1', label='Baseline',color='r',linewidth=lwidth)
    labels.append('Baseline')
    plt.ylim(0,1)
    plt.xlim(-1,31)
    plt.yticks(np.linspace(0.0,1.0,5),font={'size':tick_size})
    plt.xticks(range(0,32,5),font={'size':tick_size})
    plt.title(f"Synthesis Performance vs Baseline",fontdict={'size':title_size})
    plt.legend(loc = 'lower right',fontsize=34)
    plt.ylabel('Max F1 Score',fontdict={'size':ax_size})
    plt.xlabel('Number of Examples',fontdict={'size':ax_size})
    plt.savefig(fname)
    plt.close()

def make_ablation_plots(synth_scores:pd.DataFrame,fname):
    runs = synth_scores.groupby('label')
    plt.figure(figsize=(15,12)) 
    labels = []
    colors = ['blue','blueviolet','turquoise','limegreen']
    lwidth = 3.0
    for i, (label, run) in enumerate(runs):
        sns.lineplot(data=run,x="num_examples",y="f1_mean",color=colors[i%4],label=label,linewidth=lwidth)
        plt.errorbar(data=run,x="num_examples",y="f1_mean",yerr="f1_std",fmt=colors[i%4],label='_',linewidth=lwidth)
    plt.ylim(0,1)
    plt.xlim(-1,21)
    plt.yticks(np.linspace(0.0,1.0,5),font={'size':tick_size})
    plt.xticks(range(0,22,2),font={'size':tick_size})
    plt.title(f"Synthesis Performance for Ablated Parameters",fontdict={'size':title_size},y=1.04)
    plt.legend(loc = 'lower right',fontsize=tick_size)
    plt.ylabel('Max F1 Score',fontdict={'size':ax_size})
    plt.xlabel('Number of Examples',fontdict={'size':ax_size})
    #plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def make_performance_plot(results:pd.DataFrame,fname,xs):
    plt.figure(figsize=fig_size)
    lwidth = 3.0
    means = [results[results['prog_length'] == x].iloc[0]['time_mean'] for x in xs]
    stds = [results[results['prog_length'] == x].iloc[0]['time_std'] for x in xs]
    plt.xlim(min(xs)-1,max(xs)+1)
    sns.lineplot(x=xs,y=means,linewidth=lwidth,color='b')
    plt.errorbar(x=xs,y=means,yerr=stds,linewidth=lwidth,capsize=10,color='b')
    plt.title(f"Execution Time vs Program Size",fontdict={'size':title_size})
    #plt.legend(loc = 'lower right')
    plt.ylabel('Execution Time (seconds)',fontdict={'size':ax_size})
    plt.xlabel('Program Size (# of predicates)',fontdict={'size':ax_size})
    plt.xticks(range(0,25,5),font={'size':tick_size})
    plt.yticks(np.linspace(1.5,3.0,4),font={'size':tick_size})
    plt.savefig(fname)
    plt.close()
    
def make_heuristic_plots(synth_scores:pd.DataFrame,fname):
    plt.figure(figsize=(22,12))
    runs = synth_scores.groupby('label')
    labels = []
    colors = ['blue','blueviolet','red','limegreen']
    for i, (label, run) in enumerate(runs):
        sns.lineplot(data=run,x="num_preds",y="f1_mean",color=colors[i%4],label=label)
        plt.errorbar(data=run,x="num_preds",y="f1_mean",yerr="f1_std",fmt=colors[i%4],label='_')
    plt.ylim(0,1)
    plt.xlim(-1,101)
    plt.yticks(np.linspace(0.0,1.0,5),font={'size':tick_size})
    plt.xticks(range(0,110,10),font={'size':tick_size})
    plt.title(f"Synthesized Program Performance, by Predicate Selection Approach",fontdict={'size':title_size},y=1.04)
    plt.legend(loc =(0,0.15),fontsize=34)
    plt.ylabel('Max F1 Score',fontdict={'size':ax_size})
    plt.xlabel('Number of User Decisions',fontdict={'size':ax_size})
    plt.savefig(fname)
    plt.close()





