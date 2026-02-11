import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import random
import sys
from knowledge_graph import KnowledgeGraph
from utils import load_fever, setup_settings
from rag_engine import RAGEngine
import tqdm
from patcher import BanditPatcher 


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["fever", "fever_ts", "fever_nogate", "fever_nocost", "fever_tb", "fever_lb"]
    if dataset_name == "fever":
        method = "linucb"
        filepath = "files_fever_t"
        dataset, knowledgebase = load_fever("files_fever_v", split='train')
        with_gating = True
        with_cost = True
        latency_budget = 3
        vram_budget = 6
        postfix = ""
    elif dataset_name == "fever_ts":
        dataset_name = "fever"
        method = "thompsonsampling"
        filepath = "files_fever_ts_t"
        dataset, knowledgebase = load_fever("files_fever_ts_v", split='train')
        with_gating = True
        with_cost = True
        latency_budget = 3
        vram_budget = 6
        postfix = ""
    elif dataset_name == "fever_nogate":
        dataset_name = "fever"
        method = "linucb"
        filepath = "files_fever_t"
        dataset, knowledgebase = load_fever("files_fever_v", split='train')
        with_gating = False
        with_cost = True
        latency_budget = 3
        vram_budget = 6
        postfix = "_nogate"
    elif dataset_name == "fever_nocost":
        dataset_name = "fever"
        method = "linucb"
        filepath = "files_fever_t"
        dataset, knowledgebase = load_fever("files_fever_v", split='train')
        with_gating = True
        with_cost = False
        latency_budget = 3
        vram_budget = 6
        postfix = "_nocost"
    elif dataset_name == "fever_tb":
        dataset_name = "fever"
        method = "linucb"
        filepath = "files_fever_t"
        dataset, knowledgebase = load_fever("files_fever_v", split='train')
        with_gating = True
        with_cost = True
        latency_budget = 0.7*3
        vram_budget = 0.7*6
        postfix = "_tb"
    elif dataset_name == "fever_lb":
        dataset_name = "fever"
        method = "linucb"
        filepath = "files_fever_t"
        dataset, knowledgebase = load_fever("files_fever_v", split='train')
        with_gating = True
        with_cost = True
        latency_budget = 1.5*3
        vram_budget = 1.5*6
        postfix = "_lb"
        
    print(method, filepath, "None", with_gating, with_cost, latency_budget, vram_budget, postfix)
    setup = setup_settings(dataset_name)

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    rag = RAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=latency_budget, vram_budget=vram_budget, method=method, alpha=2., with_gating=with_gating, with_cost=with_cost)

    EPOCHS = 2

    for epoch in range(EPOCHS):
        print("Epoch:", epoch+1)
        random.shuffle(dataset)
        for row in tqdm.tqdm(dataset):
            gt_context, question, gt_answer = tuple(row)

            params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
            response_obj = rag.query(question, params=params, consistency_check=True, entailment_check=True)

            failure_label, new_label = patcher.get_failure_label(response_obj)
            response_obj["label"] = new_label
            reward = patcher.calculate_reward(failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["response"], response_obj["latency"], response_obj["vram_usage"])

            if failure_label != "NO_FAILURE":
                print(gt_answer, response_obj)
                print(failure_label)

                context = patcher.get_context(question, failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["query"], response_obj["entailment_check"]["response"], response_obj["latency"])
                patchset = "retriever" if response_obj["label"] == "UNVERIFIED" else "all"
                print("patchset:", patchset)
                action = patcher.predict(context, patchset=patchset)
                action_idx, params_updates = action
                print(params_updates)

                params.update(params_updates)
                response_obj_patched = rag.query(question, params=params, consistency_check=True, entailment_check=True)

                failure_label_patched, new_label_patched = patcher.get_failure_label(response_obj_patched)
                response_obj_patched["label"] = new_label_patched
                print(gt_answer, response_obj_patched)

                reward = patcher.calculate_reward(failure_label_patched, response_obj_patched["consistency_check"], response_obj_patched["entailment_check"]["response"], response_obj_patched["latency"], response_obj_patched["vram_usage"])
                patcher.update_bandit(context, action_idx, reward["total_reward"])

                print(failure_label_patched)
                print(reward)

            patcher.save_bandit(postfix)

            if "reindex" in params_updates.keys() and params_updates["reindex"]:
                knowledgebase.reset()
                rag.build_nodes()