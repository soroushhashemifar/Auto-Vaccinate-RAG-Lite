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
    Given the evidence, assign one of these three labels to user claim:
    - SUPPORTS: claim can be exactly inferred from the evidence.
    - REFUTES: claim strongly disagrees with evidence or the claim is wrong according to the evidence.
    - NOTENOUGHINFO: evidence is empty or none, or it does not clearly prove or disprove the claim.
    
    Then, return your answer in JSON format:
    ```json
    {
        "prediction": "SUPPORTS"|"REFUTES"|"NOTENOUGHINFO" given the evidence,
        "statement": part of the evidence supporting your predicted label without further explanation,
    }```
    """

    fact_prompt = """
    Evidence information is below.
    ---------------------
    {context_str}
    ---------------------
    claim: {query_str}
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

    similarity_top_k = 5
    similarity_cutoff = 0.4

    return {
        "device": device, 
        "fact_prompt": fact_prompt, 
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