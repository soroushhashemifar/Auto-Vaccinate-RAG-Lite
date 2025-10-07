
def get_failure_label(rag_response):
    rag_prediction = rag_response["prediction"]
    kg_result = rag_response.get("consistency_check", "CONSISTENT")
    nli_evidence_result = rag_response.get("entailment_check", {"evidence": "ENTAILMENT"})["evidence"]
    nli_claim_result = rag_response.get("entailment_check", {"claim": "ENTAILMENT"})["claim"]

    failure_label = "RAG_SUCCESS"
    if nli_claim_result == "NEUTRAL":
        failure_label = "RETRIEVAL_MISSING_EVIDENCE"
    elif rag_prediction != "NOTENOUGHINFO" and nli_evidence_result == "NEUTRAL":
        failure_label = "RAG_HALLUCINATION"
    
    elif rag_prediction == "NOTENOUGHINFO" and kg_result == "MISSING" and nli_claim_result != "NEUTRAL":
        failure_label = "RAG_WRONG_NEI"
    elif rag_prediction != "NOTENOUGHINFO" and kg_result == "MISSING" and nli_claim_result == "NEUTRAL":
        failure_label = "RAG_HALLUCINATION"
    elif (rag_prediction == "NOTENOUGHINFO" and (kg_result != "MISSING" or nli_claim_result != "NEUTRAL")) or kg_result != "CONSISTENT" or nli_evidence_result == "CONTRADICTION":
        failure_label = "RAG_HALLUCINATION"
    elif nli_claim_result == "NEUTRAL":
        failure_label = "RETRIEVAL_MISSING"
    
    return failure_label