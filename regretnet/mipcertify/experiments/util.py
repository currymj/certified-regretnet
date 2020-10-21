import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt



def compare_output(results_dict):
    has_keys = [r for r in results_dict if "gurobi_better_allocs" in r]
    alloc_diff = [np.sum(np.abs(np.array(r["gurobi_better_allocs"]) - r["better_allocs"].flatten().detach().cpu().numpy())) for r in has_keys]
    util_diff = [r["better_util_gurobi"] - r["better_util"] for r in has_keys]
    return alloc_diff, util_diff


#%%

def load_exp_data(filename):
    p = Path(filename)
    all_pickle_files = [list(p2.glob('*.pickle'))[0] for p2 in p.iterdir()]
    exp_data_dict = {}
    for pf in all_pickle_files:
        key = pf.stem
        with open(pf, 'rb') as f:
            exp_data_dict[key] = pickle.load(f)
    return exp_data_dict
#%%


# next: compute means by dict key

def get_value_by_key(exp_results, key):
    return [r[key] for r in exp_results]

def mean_std_exps(all_exp_results, key):
    mean_dict = {k: np.mean(get_value_by_key(e, key)) for k, e in all_exp_results.items()}
    std_dict = {k: np.std(get_value_by_key(e, key)) for k, e in all_exp_results.items()}
    return mean_dict, std_dict

#%%
def plot_misreports(experiment_dicts, player_ind, regret_thresh=5e-3):
    regret_results = [d for d in experiment_dicts if ("regret" in d) and (d["regret"] >= regret_thresh)]
    all_regrets = [d["regret"] for d in regret_results]
    regret_percents = [ float(p) / len(all_regrets) for p in np.argsort(all_regrets)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlabel("Item 1 Valuation")
    ax.set_xlim(0.0,1.0)
    ax.set_ylabel("Item 2 Valuation")
    ax.set_ylim(0.0,1.0)
    ax.set_aspect(0.5)
    for i, d in enumerate(regret_results):
        truthful = d["truthful_input"][player_ind,:].numpy()
        misreport = d["better_bid"][player_ind,:].numpy()
        difference = misreport - truthful
        ax.plot([truthful[0]], [truthful[1]], marker='o', markersize=5.0*regret_percents[i], color='dodgerblue')
        ax.plot([misreport[0]], [misreport[1]], marker='o', markersize=5.0*regret_percents[i], color='red')
        ax.arrow(truthful[0], truthful[1], difference[0], difference[1], head_width=0.01, alpha=0.3, length_includes_head=True)
        # ax.annotate("test", xytext=(truthful[0], truthful[1]), xy=(difference[0], difference[1]), arrowprops=dict(arrowstyle='->'))
    fig.show()
    return fig


#%%

def make_boxplot(experiment_dicts, dict_key, ordered_keys, ordered_labels, title):
    fig, ax = plt.subplots()
    ax.set_title(title)
    all_results = []
    for key in ordered_keys:
        results = [r[dict_key] for r in experiment_dicts[key]]
        all_results.append(results)
    ax.boxplot(all_results)
    plt.xticks(range(1, len(ordered_keys)+1), ordered_labels)
    fig.show()
    return fig

