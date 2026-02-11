import sys
from evaluation import evaluate_rag
import pandas as pd
import numpy as np
from utils import bootstrap_ci


def translate_actions(action):
    if not pd.isna(action):
        action = eval(action)
    else:
        action = {}

    if "reranker" in action:
        action = "Reranker activation"
    elif "prompt_edit" in action:
        if action["prompt_edit"] == "paraphrase_qa":
            action = "Paraphrase query"
        elif action["prompt_edit"] == "simplify_qa":
            action = "Simplify query"
    elif "retriever" in action:
        action = f"{action['retriever']}, top-{action['topk']}, reindex={action['reindex']}"
    else:
        action = "Default Action"
            
    return action

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["fever", "fever_ts", "hotpotqa", "hotpotqa_ts", "fever_selfrag", \
                            "hotpotqa_selfrag", "fever_nogate", "fever_nocost", "hotpotqa_nogate", \
                            "hotpotqa_nocost", "fever_paraph", "fever_top20", "fever_bestarm", \
                            "hotpotqa_paraph", "hotpotqa_top20", "hotpotqa_bestarm", "fever_posthoc", "hotpotqa_posthoc", \
                            "fever_tb", "fever_lb", "hotpotqa_tb", "hotpotqa_lb"]
    
    if dataset_name == "fever":
        filepath = "files_fever_v"
        postfix = ""
    elif dataset_name == "fever_ts":
        dataset_name = "fever"
        postfix = ""
        filepath = "files_fever_ts_v"
    elif dataset_name == "hotpotqa":
        filepath = "files_hotpotqa_v"
        postfix = ""
    elif dataset_name == "hotpotqa_ts":
        filepath = "files_hotpotqa_ts_v"
        postfix = ""
    elif dataset_name == "fever_selfrag":
        dataset_name = "fever"
        postfix = ""
        filepath = "files_fever_selfrag_v"
    elif dataset_name == "hotpotqa_selfrag":
        filepath = "files_hotpotqa_selfrag_v"
        postfix = ""
    elif dataset_name == "fever_nogate":
        filepath = "files_fever_v"
        postfix = "_nogate"
    elif dataset_name == "fever_nocost":
        filepath = "files_fever_v"
        postfix = "_nocost"
    elif dataset_name == "hotpotqa_nogate":
        filepath = "files_hotpotqa_v"
        postfix = "_nogate"
    elif dataset_name == "hotpotqa_nocost":
        filepath = "files_hotpotqa_v"
        postfix = "_nocost"
    elif dataset_name == "fever_paraph":
        filepath = "files_fever_v"
        postfix = "_paraph"
    elif dataset_name == "fever_top20":
        filepath = "files_fever_v"
        postfix = "_top20"
    elif dataset_name == "fever_bestarm":
        filepath = "files_fever_v"
        postfix = "_bestarm"
    elif dataset_name == "hotpotqa_paraph":
        filepath = "files_hotpotqa_v"
        postfix = "_paraph"
    elif dataset_name == "hotpotqa_top20":
        filepath = "files_hotpotqa_v"
        postfix = "_top20"
    elif dataset_name == "hotpotqa_bestarm":
        filepath = "files_hotpotqa_v"
        postfix = "_bestarm"
    elif dataset_name == "fever_posthoc":
        filepath = "files_fever_v"
        postfix = "_posthoc"
    elif dataset_name == "hotpotqa_posthoc":
        filepath = "files_hotpotqa_v"
        postfix = "_posthoc"
    elif dataset_name == "fever_tb":
        filepath = "files_fever_v"
        postfix = "_tb"
    elif dataset_name == "fever_lb":
        filepath = "files_fever_v"
        postfix = "_lb"
    elif dataset_name == "hotpotqa_tb":
        filepath = "files_hotpotqa_v"
        postfix = "_tb"
    elif dataset_name == "hotpotqa_lb":
        filepath = "files_hotpotqa_v"
        postfix = "_lb"

    pre_df = pd.read_csv(f"{filepath}/failure_statistics.csv", sep="\t")
    post_df = pd.read_csv(f"{filepath}/bandit_eval_dataset{postfix}.csv", sep="\t")

    failure_action_set = []
    delta_latency_per_patch = {}
    delta_vram_per_patch = {}
    for (_, pre_row), (_, post_row) in zip(pre_df.iterrows(), post_df.iterrows()):
        if pre_row["failure_label"] != "NO_FAILURE":
            failure_action_set.append((pre_row["failure_label"], translate_actions(post_row["params"])))

        delta_latency = delta_latency_per_patch.get(post_row["params"], [])
        delta_latency.append(post_row["latency"] - pre_row["latency"])
        delta_latency_per_patch[post_row["params"]] = delta_latency

        delta_vram = delta_vram_per_patch.get(post_row["params"], [])
        delta_vram.append(post_row["vram_usage"] - pre_row["vram_usage"])
        delta_vram_per_patch[post_row["params"]] = delta_vram

    action_frequency_by_failure_type = np.unique(failure_action_set, return_counts=True, axis=0)
    print("Action frequency by failure type:", action_frequency_by_failure_type)

    delta_latency_per_patch = {k:np.mean(v) for k, v in delta_latency_per_patch.items()}
    print("Delta latency per patch:", delta_latency_per_patch)

    delta_vram_per_patch = {k:np.mean(v) for k, v in delta_vram_per_patch.items()}
    print("Delta VRAM per patch:", delta_vram_per_patch)

    print("Postpatch-latency (per failure label):", post_df[["failure_label", "latency"]].groupby("failure_label").apply(bootstrap_ci, include_groups=False))
    print("Postpatch-latency (overall):", bootstrap_ci(post_df["latency"]))

    print("Postpatch-VRAM (per failure label):", post_df[["failure_label", "vram_usage"]].groupby("failure_label").apply(bootstrap_ci, include_groups=False))
    print("Postpatch-VRAM (overall):", bootstrap_ci(post_df["vram_usage"]))

    post_df['is_consistent'] = post_df['kg_consistency'] == 'CONSISTENT'
    print("Postpatch-KG Consistency (per failure label):", post_df[["failure_label", "is_consistent"]].groupby("failure_label").apply(bootstrap_ci, include_groups=False))
    print("Postpatch-KG Consistency (overall):", bootstrap_ci(post_df['kg_consistency'] == 'CONSISTENT'))

    for label in post_df["failure_label"].unique():
        print("evaluaing", label)
        df_label = post_df[post_df["failure_label"] == label]
        evaluate_rag(df_label, dataset_name)
        print("evaluaing", label, "done", df_label.shape)