from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
import tqdm
import json


def setup_settings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

    system_prompt = """
    You are a fact verification assistant.
    Given the evidence, assign one of these three labels to user claim:
    - SUPPORTS: if the evidence contains a clear, direct fact that proves the claim is correct.
    - REFUTES: if the evidence contains a clear, direct fact that proves the claim is false.
    - NOTENOUGHINFO: evidence is empty or none, or it does not clearly prove or disprove the claim.
    
    Return ONLY a valid JSON object in the exact format below—no extra text, comments, or explanations:
    
    {
        "prediction": "SUPPORTS" | "REFUTES" | "NOTENOUGHINFO",
        "statement": "<one sentence inferred from the evidence that justify your prediction>"
    }

    Rules:
    - "prediction" must be exactly one of the three uppercase labels.
    - "statement" must be a direct quote or very close paraphrase from the evidence, not your own reasoning.
    - Do not add any text outside the JSON braces.
    - JSON must have proper delimiters.
    """

    text_qa_template = """
    Evidence:
    ---------------------
    {context_str}
    ---------------------

    Claim: 
    {query_str}
    """
    
    # llm_model_name = "HuggingFaceTB/SmolLM3-3B"
    llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # llm_model_name = "tiiuae/Falcon3-1B-Instruct"
    llm_model = HuggingFaceLLM(
        model_name=llm_model_name,
        tokenizer_name=llm_model_name,
        # max_new_tokens=8,
        context_window=8192,
        system_prompt=system_prompt,
        is_chat_model=True,
    )
    
    Settings.llm = llm_model
    Settings.embed_model = embedding_model

    similarity_top_k = 3
    similarity_cutoff = 0.4

    return {
        "device": device, 
        "text_qa_template": text_qa_template, 
        "similarity_top_k": similarity_top_k, 
        "similarity_cutoff": similarity_cutoff}

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