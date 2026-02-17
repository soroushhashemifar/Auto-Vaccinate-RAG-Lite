from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import torch
from llama_index.llms.openai import OpenAI
from datasets import load_dataset
import json
import random
random.seed(42)
import pickle
import tqdm
import numpy as np
import yaml
from scipy.stats import bootstrap

from src.create_knowledge_base import WikipagesKnowledgeBase


def setup_settings(dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

    prompts_filepath = "src/prompts_shortanswer.yaml" if dataset in ["hotpotqa"] else "src/prompts.yaml"
    with open(prompts_filepath, 'r') as f:
        prompts = yaml.load(f, Loader=yaml.FullLoader)

    if dataset in ["fever"]:
        kg_thresholds = {"subject_score": 0.9, "relation_score": 0.9, "object_score": 0.1}
    elif dataset in ["hotpotqa"]:
        kg_thresholds = {"subject_score": 0.9, "relation_score": 0.9, "object_score": 0.5}
    
    llm_model = OpenAI(model="gpt-4o-mini")
    
    Settings.llm = llm_model
    Settings.embed_model = embedding_model

    similarity_top_k = 3
    reranker_top_n = 3
    similarity_cutoff = 0.4
    rag_storage_dir = "storage_rag"
    kg_storage_dir = "storage_kg"
    dense_retriever_storage = "dense_storage"
    bm25_retriever_storage = "bm25_storage"

    return {
        "device": device, 
        "prompts": prompts,
        "similarity_top_k": similarity_top_k, 
        "reranker_top_n": reranker_top_n,
        "similarity_cutoff": similarity_cutoff,
        "rag_storage_dir": rag_storage_dir,
        "kg_storage_dir": kg_storage_dir,
        "kg_thresholds": kg_thresholds,
        "dense_retriever_storage": dense_retriever_storage,
        "bm25_retriever_storage": bm25_retriever_storage}

def load_fever(filepath, split='train'):
    with open(os.path.join(filepath, "knowledge_base.pkl"), 'rb') as f:
        content = pickle.load(f)
        meta_data = content["meta_data"]
        evidences = content["sentences"]
        evidence_dict = {j["doc_id"]:i for i, j in zip(evidences, meta_data)}

    contexts = list(set([" ".join(evidence).replace("\t", " ").strip() for evidence in evidences if len(evidence) > 0]))

    wkb = WikipagesKnowledgeBase()
    secondary_contexts = []
    for f in tqdm.tqdm(wkb.iter_files("./wiki-pages")):
        documents = wkb.get_contents(f)
        secondary_contexts.extend(list(map(lambda item: wkb.preprocess(item[1]), documents)))
        if len(secondary_contexts) > 100000:
            secondary_contexts = secondary_contexts[:100000]
            break

    print("Num unique contexts:", len(contexts), len(secondary_contexts))

    kb = KnowledgeBaseSimulator(contexts, secondary_contexts, random_seed=42)

    with open("./shared_task_dev.jsonl", 'r') as json_file:
        if split == "train":
            json_list = list(json_file)[:1000]
        else:
            json_list = list(json_file)[1000:2000]

        fever_dataset = []
        for json_str in tqdm.tqdm(json_list):
            result = json.loads(json_str)
            evidence_sentences = []
            for evidence in result['evidence']: 
                if evidence[0][2] is not None: 
                    evidence_sentences.append(evidence_dict[evidence[0][2]][1:][evidence[0][3]].replace("\t", " "))
            
            evidence_sentences = list(set(evidence_sentences))
            fever_dataset.append((evidence_sentences, result['claim'], result['label'].replace(' ', '')))

    print("fever_dataset size:", len(fever_dataset))

    return fever_dataset, kb

def load_hotpotqa(split='train'):
    dataset = load_dataset('hotpotqa/hotpot_qa', 'fullwiki', split=f"{split}[:500]")
    contexts = list(set(["".join(hop) for context in dataset['context'] for hop in context['sentences']]))
    
    dataset_ = load_dataset('hotpotqa/hotpot_qa', 'fullwiki', split=f"{split}[:100000]")
    secondary_contexts = list(set(["".join(hop) for context in dataset_['context'] for hop in context['sentences']]))
    
    print("Num unique contexts:", len(contexts), len(secondary_contexts))

    kb = KnowledgeBaseSimulator(contexts, secondary_contexts, random_seed=42)

    dataset = [([], q_item, [a_item]) for q_item, a_item in zip(dataset['question'], dataset['answer'])]

    return dataset, kb

def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)

        return instances[cls]

    return _singleton

def precision_at_k(gt_contexts, contexts, k):    
    states = [any([gt_context in ctx for gt_context in gt_contexts]) for ctx in contexts[:k]]
    prec = sum([sum(states) / len(states)]) 
    return prec

def context_precision(gt_contexts, contexts, K=5):
    if len(gt_contexts) == 0 or len(contexts) == 0:
        return None
    
    prec = sum([precision_at_k(gt_contexts, contexts, k) for k in range(1, K+1)]) 
    prec = prec / K
    return prec

def context_recall(gt_contexts, contexts):
    if len(gt_contexts) == 0 or len(contexts) == 0:
        return None
    
    recall = [any([gt_context in ctx for ctx in contexts]) for gt_context in gt_contexts]
    recall = sum(recall) / len(recall)
    return recall

def bootstrap_ci(df):
    data = df.to_numpy().astype(float)
    if data.shape[0] == 1:
        mean = data[0]
        ci = [np.nan, np.nan]
    else:
        result = bootstrap((data,), np.mean, confidence_level=0.95, n_resamples=10)
        ci = result.confidence_interval
        mean = result.bootstrap_distribution.mean()

    return np.round(mean, 8).item(), (np.round(ci[0], 8).item(), np.round(ci[1], 8).item())


class KnowledgeBaseSimulator:

    def __init__(self, primary_kb, secondary_kb, random_seed=42):
        self.primary_kb = primary_kb.copy()
        self.secondary_kb = secondary_kb.copy()
        self.random_seed = random_seed

        self.current_kb = self.primary_kb.copy()
        self.set_random_seed()
        self._index = 0

    def set_random_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def evolve(self):
        if random.random() > 0.5:
            print("Current knowledge base size:", len(self.current_kb))
            return
        
        num_passages_to_add = 100
        num_to_add = min(num_passages_to_add, len(self.secondary_kb))
        passages_to_add = random.sample(self.secondary_kb, num_to_add)
        self.current_kb.extend(passages_to_add)
        print("Current knowledge base size:", len(self.current_kb))

    def reset(self):
        self.set_random_seed()
        self.current_kb = self.primary_kb.copy()
        self._index = 0

    def get_current_kb(self):
        return self.current_kb.copy()

    def __len__(self):
        return len(self.current_kb)

    def __getitem__(self, index):
        return self.current_kb.copy()[index]

    def __iter__(self):
        return self

    def __next__(self):
        if self.has_next():
            return self.next()
        else:
            raise StopIteration

    def has_next(self):
        flag = self._index < len(self.current_kb)

        if not flag:
            self._index = 0

        return flag

    def next(self):
        if not self.has_next():
            raise IndexError("No more elements in the list.")
        element = self.current_kb.copy()[self._index]
        self._index += 1
        return element