import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
import tqdm
import json


def setup_settings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

    with open(os.path.join("prompts", "fact_verif.txt"), 'r') as file:
        rag_fact_verif_prompt = file.read()

    with open(os.path.join("prompts", "prompt_edit_WRONG_PREDICATE.txt"), 'r') as file:
        prompt_WP = file.read()

    with open(os.path.join("prompts", "prompt_edit_WRONG_RESPONSE.txt"), 'r') as file:
        prompt_WR = file.read()

    with open(os.path.join("prompts", "kg_consist.txt"), 'r') as file:
        knowledge_graph_completion_prompt = file.read()
    
    llm_model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_model = HuggingFaceLLM(
        model_name=llm_model_name,
        tokenizer_name=llm_model_name,
        context_window=8192,
        is_chat_model=True,
        generate_kwargs={"do_sample": False},
        # model_kwargs={"load_in_4bit": True},
        model_kwargs={"dtype": torch.bfloat16},
        max_new_tokens=256,
    )
    
    Settings.llm = llm_model
    Settings.embed_model = embedding_model

    similarity_top_k = 3
    reranker_top_n = 3
    similarity_cutoff = 0.4
    rag_storage_dir = "./storage_rag"
    kg_storage_dir = "./storage_kg"
    dense_retriever_storage = "./dense_storage"
    bm25_retriever_storage = "./bm25_storage"

    return {
        "device": device, 
        "rag_fact_verif_prompt": rag_fact_verif_prompt, 
        "prompt_WP": prompt_WP,
        "prompt_WR": prompt_WR,
        "similarity_top_k": similarity_top_k, 
        "reranker_top_n": reranker_top_n,
        "similarity_cutoff": similarity_cutoff,
        "rag_storage_dir": rag_storage_dir,
        "kg_storage_dir": kg_storage_dir,
        "KG_completion_prompt": knowledge_graph_completion_prompt,
        "dense_retriever_storage": dense_retriever_storage,
        "bm25_retriever_storage": bm25_retriever_storage}

def load_fever(fever_json_path):
    with open(fever_json_path, 'r') as json_file:
        json_list = list(json_file)

        fever_dataset = []
        for json_str in tqdm.tqdm(json_list):
            result = json.loads(json_str)
            fever_dataset.append((result['claim'], result['label'].replace(' ', '')))

    print("fever_dataset size:", len(fever_dataset))

    return fever_dataset

def singleton(cls, *args, **kwargs):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)

        return instances[cls]

    return _singleton