import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import random
import sys 
from knowledge_graph import KnowledgeGraph
from utils import load_hotpotqa, setup_settings
from rag_engine import RAGEngine
import tqdm
from datasets import Dataset
from patcher import BanditPatcher
from qa_metrics.em import em_match
import copy


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["hotpotqa"]
    if dataset_name == "hotpotqa":
        method = "linucb"
        filepath = "files_hotpotqa_v"
        patcher_filepath = "files_hotpotqa_t"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        latency_budget = 3
        vram_budget = 6
        postfix = "_posthoc"
        
    print(method, filepath, patcher_filepath, True, True, latency_budget, vram_budget, postfix)
    setup = setup_settings(dataset_name)

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    rag = RAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=latency_budget, vram_budget=vram_budget, method=method, alpha=2.)

    random.seed(42)
    dataset = random.choices(dataset, k=200)

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

            action_idx = None
            params_updates = None
            response_obj = None
            failure_label = None
            reward = {"total_reward": -1000000.}
            possible_actions = patcher.possible_actions
            for action in possible_actions:
                action_idx_, params_updates_ = action
                print(params_updates_)

                params_ = copy.deepcopy(params)
                params_.update(params_updates_)
                response_obj_ = rag.query_shortanswer(question, params=params_, consistency_check=True, entailment_check=True)

                failure_label_, _ = patcher.get_failure_label(response_obj_)
                print(gt_answer, response_obj_)
                
                reward_ = patcher.calculate_reward(failure_label_, response_obj_["consistency_check"], response_obj_["entailment_check"]["response"], response_obj_["latency"], response_obj_["vram_usage"])

                print(failure_label_)
                print(reward_)

                if reward_["total_reward"] > reward["total_reward"]:
                    action_idx = copy.deepcopy(action_idx_)
                    params_updates = copy.deepcopy(params_updates_)
                    response_obj = copy.deepcopy(response_obj_)
                    failure_label = copy.deepcopy(failure_label_)
                    reward = copy.deepcopy(reward_)
        
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