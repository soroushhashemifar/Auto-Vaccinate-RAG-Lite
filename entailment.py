from transformers import pipeline
from utils import singleton


@singleton
class EntailmentChecker:

    def __init__(self, **kwargs):
        self.model = pipeline("text-classification", model="tasksource/ModernBERT-base-nli", device=kwargs["device"])

    def check(self, claim, rag_evidence, retrieved_context):
        text_input = " ".join([item['text'] for item in retrieved_context])

        claim_pred = self.model([{"text": text_input, "text_pair": claim}], top_k=None)[0]
        claim_results = [(p['label'].upper(), p['score']) for p in claim_pred]
        claim_entailment_label = max(claim_results, key=lambda item: item[1])[0]

        rag_pred = self.model([{"text": text_input, "text_pair": rag_evidence}], top_k=None)[0]
        rag_results = [(p['label'].upper(), p['score']) for p in rag_pred]
        rag_entailment_label = max(rag_results, key=lambda item: item[1])[0]

        return {"claim": claim_entailment_label, "evidence": rag_entailment_label}