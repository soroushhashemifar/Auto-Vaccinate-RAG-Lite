import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys 
from knowledge_graph import KnowledgeGraph
from utils import load_hotpotqa, setup_settings
from rag_engine import RAGEngine
import tqdm
from datasets import Dataset
from patcher import BanditPatcher
from qa_metrics.em import em_match


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["hotpotqa", "hotpotqa_ts", "hotpotqa_nogate", "hotpotqa_nocost", "hotpotqa_tb", "hotpotqa_lb"]
    if dataset_name == "hotpotqa":
        method = "linucb"
        filepath = "files_hotpotqa_v"
        patcher_filepath = "files_hotpotqa_t"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        with_gating = True
        with_cost = True
        latency_budget = 3
        vram_budget = 6
        postfix = ""
    elif dataset_name == "hotpotqa_ts":
        dataset_name = "hotpotqa"
        method = "thompsonsampling"
        filepath = "files_hotpotqa_ts_v"
        patcher_filepath = "files_hotpotqa_ts_t"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        with_gating = True
        with_cost = True
        latency_budget = 3
        vram_budget = 6
        postfix = ""
    elif dataset_name == "hotpotqa_nogate":
        dataset_name = "hotpotqa"
        method = "linucb"
        filepath = "files_hotpotqa_v"
        patcher_filepath = "files_hotpotqa_t"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        with_gating = False
        with_cost = True
        latency_budget = 3
        vram_budget = 6
        postfix = "_nogate"
    elif dataset_name == "hotpotqa_nocost":
        dataset_name = "hotpotqa"
        method = "linucb"
        filepath = "files_hotpotqa_v"
        patcher_filepath = "files_hotpotqa_t"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        with_gating = True
        with_cost = False
        latency_budget = 3
        vram_budget = 6
        postfix = "_nocost"
    elif dataset_name == "hotpotqa_tb":
        dataset_name = "hotpotqa"
        method = "linucb"
        filepath = "files_hotpotqa_v"
        patcher_filepath = "files_hotpotqa_t"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        with_gating = True
        with_cost = True
        latency_budget = 0.7*3
        vram_budget = 0.7*6
        postfix = "_tb"
    elif dataset_name == "hotpotqa_lb":
        dataset_name = "hotpotqa"
        method = "linucb"
        filepath = "files_hotpotqa_v"
        patcher_filepath = "files_hotpotqa_t"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        with_gating = True
        with_cost = True
        latency_budget = 1.5*3
        vram_budget = 1.5*6
        postfix = "_lb"

    print(method, filepath, patcher_filepath, with_gating, with_cost, latency_budget, vram_budget, postfix)
    setup = setup_settings(dataset_name)

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    rag = RAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=latency_budget, vram_budget=vram_budget, method=method, alpha=2., with_gating=with_gating, with_cost=with_cost)
    patcher.load_bandit(patcher_filepath, postfix)

    data = {
        "index": [],
        "question": [],
        "response": [],
        "contexts": [],
        "kg_consistency": [],
        "query_entailment": [],
        "response_entailment": [],
        "latency": [],
        "vram_usage": [],
        "gt_response": [],
        "gt_contexts": [],
        "action": [],
        "params": [],
        "failure_label": [],
        "EM": [],
        "bandit_reward": [],
    }
    query_idx = -1
    for row in tqdm.tqdm(dataset):
        query_idx += 1
        gt_context, question, gt_answer = tuple(row)

        action_idx = -1
        params_updates = params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
        response_obj = rag.query_shortanswer(question, params=params, consistency_check=True, entailment_check=True)

        failure_label, _ = patcher.get_failure_label(response_obj)
        reward = patcher.calculate_reward(failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["response"], response_obj["latency"], response_obj["vram_usage"])

        if failure_label != "NO_FAILURE":
            print(gt_answer, response_obj)
            print(failure_label)

            context = patcher.get_context(question, failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["query"], response_obj["entailment_check"]["response"], response_obj["latency"])
            action = patcher.predict(context)
            action_idx, params_updates = action
            print(params_updates)

            params.update(params_updates)
            response_obj = rag.query_shortanswer(question, params=params, consistency_check=True, entailment_check=True)

            failure_label, _ = patcher.get_failure_label(response_obj)
            print(gt_answer, response_obj)
            
            reward = patcher.calculate_reward(failure_label, response_obj["consistency_check"], response_obj["entailment_check"]["response"], response_obj["latency"], response_obj["vram_usage"])

            print(failure_label)
            print(reward)
        
        data["index"].append(query_idx)
        retrieved_context = [item["text"] for item in response_obj["retrieved_context"]]
        data["question"].append(question)
        data["response"].append(response_obj["response"])
        data["contexts"].append(retrieved_context)
        data["kg_consistency"].append(response_obj["consistency_check"])
        data["query_entailment"].append(response_obj["entailment_check"]["query"])
        data["response_entailment"].append(response_obj["entailment_check"]["response"])
        data["latency"].append(response_obj["latency"])
        data["vram_usage"].append(response_obj["vram_usage"])
        data["gt_response"].append(gt_answer[0])
        data["gt_contexts"].append(gt_context)
        data["action"].append(action_idx)
        data["params"].append(str(params_updates))
        data["failure_label"].append(failure_label)
        data["EM"].append(em_match(list(map(lambda item: item.lower(), gt_answer)), response_obj["response"].lower()))
        data["bandit_reward"].append(str(reward))

        if "reindex" in params_updates.keys() and params_updates["reindex"]:
            knowledgebase.reset()
            rag.build_nodes()

    dataset = Dataset.from_dict(data)
    dataset = dataset.to_pandas()

    for c in ["kg_consistency", "query_entailment", "response_entailment", "failure_label"]:
        print(f"######## {c} (Correct) #######")
        print(dataset[dataset["EM"] == True][c].value_counts())
        print(f"######## {c} (Incorrect) #######")
        print(dataset[dataset["EM"] == False][c].value_counts())
    
    dataset.to_csv(f"{filepath}/bandit_eval_dataset{postfix}.csv", sep="\t", index=False)