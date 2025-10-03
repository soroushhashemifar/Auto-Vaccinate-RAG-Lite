from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import Settings
import torch
import tqdm
import json


def setup_settings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", device=device)

    rag_fact_verif_prompt = """
    You are a fact verification assistant.

    Convert the following claim to one neutral question. Do not miss out anything 
    important from the claim. Question the claim, not the fact.

    Claim: {query_str}

    Then, find the answer to the question strictly based on this context:
    ---------------------
    {context_str}
    ---------------------

    Follow these strict rules:
    1. You must output in valid JSON format.
    2. Do not include extra text, explanations, or formatting.
    3. The field "prediction" MUST be exactly one of:
    - "SUPPORTS" : context indicates that the claim is correct.
    - "REFUTES" : context directly refusing the claim.
    - "NOTENOUGHINFO" : if you cannot find any conclusive factual evidence either supporting or refuting the claim.
    4. The field "evidence" MUST be a single sentence that explains WHY the prediction was chosen, explicitly referencing the context shortly.
    5. If the preiction is "NOTENOUGHINFO", leave the "evidence" as blank string "".
    6. The extracted information should be a conclusive answer, either affirmative or negative, and concise, without any irrelevant words. 
    9. You are strictly forbidden from generating any text of your own.
    10. ONLY answer the question if there are relevant references in the context.

    Output format:
    ```json
    {
    "question": "your generated question"
    "prediction": "SUPPORTS" | "REFUTES" | "NOTENOUGHINFO",
    "evidence": "evidence to support your answer shortly"
    }```
    """

    knowledge_graph_completion_prompt = """
    You are a knowledge graph query engine. 

    The context text is:
    ------------
    {context_str}
    ------------

    The user query is:
    {query_str}

    Strict rules:
    - Subject or object might appear in the context as a single clause or in a delimiter-separated clause, or a list.
    - IF either the subject or the object do not appear anywhere in the context, return only MISSING.
    - IF the subject and the object have the specified relationship in the context, return only CONSISTENT.
    - IF any other condition applies, other than the two above, return only CONFLICT.
    - Use ONLY explicit text in context and do not use prior knowledge or external facts.
    - Do NOT output any other text, punctuation, explanation.
    """
    
    # llm_model_name = "HuggingFaceTB/SmolLM3-3B"
    # llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    # llm_model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct-16k"
    llm_model_name = "Qwen/Qwen2.5-3B-Instruct"
    llm_model = HuggingFaceLLM(
        model_name=llm_model_name,
        tokenizer_name=llm_model_name,
        # max_new_tokens=50,
        context_window=16000,
        # system_prompt=system_prompt,
        is_chat_model=True,
    )
    
    Settings.llm = llm_model
    Settings.embed_model = embedding_model

    similarity_top_k = 3
    similarity_cutoff = 0.4
    rag_storage_dir = "./storage_rag"
    kg_storage_dir = "./storage_kg"

    return {
        "device": device, 
        "text_qa_template": rag_fact_verif_prompt, 
        "similarity_top_k": similarity_top_k, 
        "similarity_cutoff": similarity_cutoff,
        "rag_storage_dir": rag_storage_dir,
        "kg_storage_dir": kg_storage_dir,
        "KG_completion_prompt": knowledge_graph_completion_prompt}

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