from sentence_transformers import CrossEncoder
import torch
from utils import singleton


@singleton
class EntailmentChecker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/nli-roberta-base")
        self.entailment_labels = ['CONTRADICTION', 'ENTAILMENT', 'NEUTRAL']

    def check(self, claim, rag_evidence, retrieved_context):
        claim_entailment_checks = []
        evidence_entailment_checks = []
        for chunk in retrieved_context:
            context = chunk['text']
            logits = self.model.predict([(context, claim), (context, rag_evidence)])
            label_scores = torch.softmax(torch.from_numpy(logits), -1).tolist()

            claim_entailment_checks.append(label_scores[0])
            evidence_entailment_checks.append(label_scores[1])

        claim_entailment_scores = torch.tensor(claim_entailment_checks).mean(0)
        evidence_entailment_scores = torch.tensor(evidence_entailment_checks).mean(0)

        claim_entailment_label = self.entailment_labels[claim_entailment_scores.argmax()]
        evidence_entailment_label = self.entailment_labels[evidence_entailment_scores.argmax()]

        return {"claim": claim_entailment_label, "evidence": evidence_entailment_label}