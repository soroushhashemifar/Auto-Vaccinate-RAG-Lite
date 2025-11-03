import os 
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

from knowledge_graph import WikiMoviesKnowledgeGraph
from utils import setup_settings
from rag_engine import RAGEngine
from utils import load_fever
import tqdm
from datasets import Dataset
from patcher import BanditPatcher, BanditPatcherGR 


if __name__ == "__main__":
    setup = setup_settings()

    kgraph = WikiMoviesKnowledgeGraph(**setup)
    kgraph.build("./movieqa", "out/wikipages_knowledge_base.pkl", -1)
    rag = RAGEngine("out/wikipages_knowledge_base.pkl", knowledge_graph=kgraph, **setup)

    fever_dataset = load_fever("./shared_task_dev.jsonl")
    # subset = random.choices(fever_dataset, k=50)
    subset = fever_dataset[200:250]

    # patcher = BanditPatcher(latency_budget=6, vram_budget=None, method="linucb", alpha=0.1)
    patcher = BanditPatcherGR(latency_budget=5, vram_budget=None, method="linucb", alpha=(1.2, 1.2))
    patcher.load_bandit()

    data = {
        "index": [],
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "rag_question": [],
        "rag_label": [],
        "kg_consistency": [],
        "claim_entailment": [],
        "response_entailment": [],
        "latency": [],
        "vram_usage": [],
        "action": [],
        "params": [],
        "failure_label": [],
        "bandit_reward": [],
    }
    claim_idx = 0
    for claim, label in tqdm.tqdm(subset):        
        params = {'retriever': 'dense', 'topk': 10, 'reranker': False, 'prompt_edit': False, 'reindex': False}
        pred = rag.query(claim, params=params, consistency_check=True, entailment_check=True)

        failure_label = patcher.get_failure_label(pred)
        if failure_label == "LABEL_RESPONSE_MISMATCH_FAILURE":
            if pred["prediction"] == "SUPPORTS":
                pred["prediction"] = "REFUTES"
            elif pred["prediction"] == "REFUTES":
                pred["prediction"] = "SUPPORTS"

            failure_label = patcher.get_failure_label(pred)

        reward = patcher.calculate_reward(failure_label, pred["raw_response"], pred["prediction"], label, pred["consistency_check"], pred["entailment_check"]["response"], pred["latency"], pred["vram_usage"])

        data["index"].append(claim_idx)
        data["question"].append(claim)
        data["ground_truth"].append(label)
        data["answer"].append(pred["response"])
        data["contexts"].append([item["text"] for item in pred["retrieved_context"]])
        data["rag_question"].append(pred["question"])
        data["rag_label"].append(pred["prediction"])
        data["kg_consistency"].append(pred["consistency_check"])
        data["claim_entailment"].append(pred["entailment_check"]["claim"])
        data["response_entailment"].append(pred["entailment_check"]["response"])
        data["latency"].append(pred["latency"])
        data["vram_usage"].append(pred["vram_usage"])
        data["action"].append(-1)
        data["params"].append(str(params))
        data["failure_label"].append(failure_label)
        data["bandit_reward"].append(str(reward))

        if failure_label != "NO_FAILURE":
            context = patcher.get_context(claim, len(pred["retrieved_context"]), failure_label, pred["consistency_check"], pred["entailment_check"]["claim"], pred["entailment_check"]["response"], pred["raw_response"])
            action_idx, params_updates = patcher.predict(context, failure_label=failure_label)
            params.update(params_updates)
            
            pred = rag.query(claim, params=params, consistency_check=True, entailment_check=True)

            failure_label = patcher.get_failure_label(pred)
            if failure_label == "LABEL_RESPONSE_MISMATCH_FAILURE":
                if pred["prediction"] == "SUPPORTS":
                    pred["prediction"] = "REFUTES"
                elif pred["prediction"] == "REFUTES":
                    pred["prediction"] = "SUPPORTS"

                failure_label = patcher.get_failure_label(pred)

            reward = patcher.calculate_reward(failure_label, pred["raw_response"], pred["prediction"], label, pred["consistency_check"], pred["entailment_check"]["response"], pred["latency"], pred["vram_usage"])

            data["index"].append(claim_idx)
            data["question"].append(claim)
            data["ground_truth"].append(label)
            data["answer"].append(pred["response"])
            data["contexts"].append([item["text"] for item in pred["retrieved_context"]])
            data["rag_question"].append(pred["question"])
            data["rag_label"].append(pred["prediction"])
            data["kg_consistency"].append(pred["consistency_check"])
            data["claim_entailment"].append(pred["entailment_check"]["claim"])
            data["response_entailment"].append(pred["entailment_check"]["response"])
            data["latency"].append(pred["latency"])
            data["vram_usage"].append(pred["vram_usage"])
            data["action"].append(action_idx)
            data["params"].append(str(params_updates))
            data["failure_label"].append(failure_label)
            data["bandit_reward"].append(str(reward))

            print(params_updates)
            print(pred["prediction"], "|", pred["response"])
            print(failure_label, reward)

        claim_idx += 1

    dataset = Dataset.from_dict(data)
    dataset.to_pandas().to_csv("out/bandit_eval_dataset.csv", sep="\t", index=False)