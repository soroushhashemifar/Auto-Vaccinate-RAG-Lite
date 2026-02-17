import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys 
import tqdm
from datasets import Dataset
from qa_metrics.em import em_match

from src.knowledge_graph import KnowledgeGraph
from src.utils import load_hotpotqa, setup_settings
from src.rag_engine import RAGEngine
from src.patcher import BanditPatcher


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["hotpotqa", "hotpotqa_ts"]
    if dataset_name == "hotpotqa":
        method = "linucb"
        filepath = "files_hotpotqa_v"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        latency_budget = 3
        vram_budget = 6
    elif dataset_name == "hotpotqa_ts":
        dataset_name = "hotpotqa"
        method = "thompsonsampling"
        filepath = "files_hotpotqa_ts_v"
        dataset, knowledgebase = load_hotpotqa(split='validation')
        latency_budget = 3
        vram_budget = 6
        
    print(method, filepath, "None", True, True, latency_budget, vram_budget, "None")
    setup = setup_settings(dataset_name)

    kgraph = KnowledgeGraph(filepath, **setup)
    kgraph.build(knowledgebase)
    kgraph.plot()
    rag = RAGEngine(filepath, knowledgebase, knowledge_graph=kgraph, **setup)

    patcher = BanditPatcher(filepath, latency_budget=latency_budget, vram_budget=vram_budget, method=method, alpha=2.)

    data = {
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
    }
    for row in tqdm.tqdm(dataset):
        gt_context, question, gt_answer = tuple(row)

        action_idx = -1
        params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
        response_obj = rag.query_shortanswer(question, params=params, consistency_check=True, entailment_check=True)

        failure_label, _ = patcher.get_failure_label(response_obj)
        
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
        data["params"].append(str(params))
        data["failure_label"].append(failure_label)
        data["EM"].append(em_match(list(map(lambda item: item.lower(), gt_answer)), response_obj["response"].lower()))

    dataset = Dataset.from_dict(data)
    dataset = dataset.to_pandas()

    for c in ["kg_consistency", "query_entailment", "response_entailment", "failure_label"]:
        print(f"######## {c} (Correct) #######")
        print(dataset[dataset["EM"] == True][c].value_counts())
        print(f"######## {c} (Incorrect) #######")
        print(dataset[dataset["EM"] == False][c].value_counts())
    
    dataset.to_csv(f"{filepath}/failure_statistics.csv", sep="\t", index=False)