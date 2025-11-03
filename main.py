import os 
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

from knowledge_graph import WikiMoviesKnowledgeGraph
from utils import setup_settings
from rag_engine import RAGEngine
from utils import load_fever
import tqdm
import random
from datasets import Dataset


if __name__ == "__main__":
    setup = setup_settings()

    kgraph = WikiMoviesKnowledgeGraph(**setup)
    # kgraph.manual_check_triplets("./movieqa", "out/wikipages_knowledge_base.pkl", 100)
    kgraph.build("./movieqa", "out/wikipages_knowledge_base.pkl", -1)
    kgraph.plot()
    # kgraph = None

    rag = RAGEngine("out/wikipages_knowledge_base.pkl", knowledge_graph=kgraph, **setup)

    fever_dataset = load_fever("./shared_task_dev.jsonl")
    subset = random.choices(fever_dataset, k=50)

    data = {
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
    }
    for claim, label in tqdm.tqdm(subset):
        data["question"].append(claim)
        data["ground_truth"].append(label)

        action_idx = 0
        params = {'retriever': 'bm25', 'topk': 5, 'reranker': False, 'prompt_edit': False, 'reindex': False}
        pred = rag.query(claim, params=params, consistency_check=True, entailment_check=True)

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
        data["params"].append(params)

    dataset = Dataset.from_dict(data)
    df = dataset.to_pandas()
    df.to_csv("out/eval_dataset.csv", sep="\t", index=False)

    print(df["ground_truth"].value_counts())
    print(df[["rag_label"]].value_counts())
    print(df[["kg_consistency"]].value_counts())
    print(df[["claim_entailment"]].value_counts())
    print(df[["latency"]].mean())
    print(df[["vram_usage"]].mean())