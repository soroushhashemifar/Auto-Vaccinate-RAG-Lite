from sentence_transformers import CrossEncoder
import torch


class EntailmentChecker:

    def __init__(self):
        self.model = CrossEncoder("dleemiller/ModernCE-base-nli")
        self.entailment_labels = ['CONTRADICTION', 'ENTAILMENT', 'NEUTRAL']

    def check(self, claim, rag_evidence, retrieved_context):
        claim_consistency_checks = []
        evidence_consistency_checks = []
        for chunk in retrieved_context:
            context = chunk['text']
            scores = self.model.predict([(context, claim), (context, rag_evidence)])
            labels = [self.entailment_labels[score_max] for score_max in scores.argmax(axis=1)]
            label_scores = torch.softmax(torch.from_numpy(scores), -1).max(-1).values.tolist()

            claim_consistency_checks.append((chunk, labels[0], label_scores[0]))
            evidence_consistency_checks.append((chunk, labels[1], label_scores[1]))

        return {"claim": claim_consistency_checks, "evidence": evidence_consistency_checks}