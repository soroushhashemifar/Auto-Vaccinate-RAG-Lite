import os
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

import sys
from knowledge_graph import KnowledgeGraph
from utils import context_precision, context_recall, setup_settings, load_fever
from rag_engine import RAGEngine
import tqdm
from datasets import Dataset
from patcher import BanditPatcher 
from qa_metrics.em import em_match


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    assert dataset_name in ["fever", "fever_ts"]
    if dataset_name == "fever":
        method = "linucb"
        filepath = "files_fever_v"
        dataset, knowledgebase = load_fever(filepath, split='validation')
        latency_budget = 3
        vram_budget = 6
    elif dataset_name == "fever_ts":
        dataset_name = "fever"
        method = "thompsonsampling"
        filepath = "files_fever_ts_v"
        dataset, knowledgebase = load_fever(filepath, split='validation')
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
        "response_label": [],
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
        "context_precision@5": [],
        "context_recall": [],
    }
    for row in tqdm.tqdm(dataset):
        gt_context, question, gt_answer = tuple(row)

        params = {'retriever': 'dense', 'topk': 5, 'reranker': False, 'prompt_edit': "simple_qa", 'reindex': False}
        response_obj = rag.query(question, params=params, consistency_check=True, entailment_check=True)

        failure_label, new_label = patcher.get_failure_label(response_obj)
        response_obj["label"] = new_label

        retrieved_context = [item["text"] for item in response_obj["retrieved_context"]]
        data["question"].append(question)
        data["response"].append(response_obj["response"])
        data["response_label"].append(response_obj["label"])
        data["contexts"].append(retrieved_context)
        data["kg_consistency"].append(response_obj["consistency_check"])
        data["query_entailment"].append(response_obj["entailment_check"]["query"])
        data["response_entailment"].append(response_obj["entailment_check"]["response"])
        data["latency"].append(response_obj["latency"])
        data["vram_usage"].append(response_obj["vram_usage"])
        data["gt_response"].append(gt_answer)
        data["gt_contexts"].append(gt_context)
        data["action"].append(-1)
        data["params"].append(str(params))
        data["failure_label"].append(failure_label)
        data["EM"].append(em_match([gt_answer], response_obj["label"]))
        data["context_precision@5"].append(context_precision(gt_context, retrieved_context, K=5))
        data["context_recall"].append(context_recall(gt_context, retrieved_context))

    dataset = Dataset.from_dict(data)
    dataset = dataset.to_pandas()

    for c in ["kg_consistency", "query_entailment", "response_entailment", "failure_label"]:
        print(f"######## {c} (Correct) #######")
        print(dataset[dataset["EM"] == True][c].value_counts())
        print(f"######## {c} (Incorrect) #######")
        print(dataset[dataset["EM"] == False][c].value_counts())
    
    print(f"######## {c} (Correct) #######")
    print("context_precision@5", dataset[dataset["EM"] == True]["context_precision@5"].dropna().mean())
    print("context_recall", dataset[dataset["EM"] == True]["context_recall"].dropna().mean())
    print(f"######## {c} (Incorrect) #######")
    print("context_precision@5", dataset[dataset["EM"] == False]["context_precision@5"].dropna().mean())
    print("context_recall", dataset[dataset["EM"] == False]["context_recall"].dropna().mean())

    print(f"######## Label (Correct) #######")
    print(dataset[dataset["EM"] == True]["gt_response"].value_counts())
    print(dataset[dataset["EM"] == True]["response_label"].value_counts())
    print(f"######## Label (Incorrect) #######")
    print(dataset[dataset["EM"] == False]["gt_response"].value_counts())
    print(dataset[dataset["EM"] == False]["response_label"].value_counts())

    dataset.to_csv(f"{filepath}/failure_statistics.csv", sep="\t", index=False)