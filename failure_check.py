
def get_failure_label(rag_response):
    # rag_evidence = rag_response["evidence"]
    rag_prediction = rag_response["prediction"]
    kg_result = rag_response.get("consistency_check", "CONSISTENT")
    # nli_evidence_result = rag_response.get("entailment_check", {}).get("evidence", "ENTAILMENT")
    nli_claim_result = rag_response.get("entailment_check", {}).get("claim", "ENTAILMENT")

    if kg_result == "MISSING":
        failure_label = "MISSING_ENTITY"
    elif kg_result == "CONFLICT":
        failure_label = "MISSING_PREDICATE"
    elif rag_prediction == "SUPPORTS" and nli_claim_result == "ENTAILMENT":
        failure_label = "SUCCESSFUL"
    elif rag_prediction == "REFUTES" and nli_claim_result == "CONTRADICTION":
        failure_label = "SUCCESSFUL"
    elif rag_prediction == "NOTENOUGHINFO" and nli_claim_result == "NEUTRAL":
        failure_label = "SUCCESSFUL"
    else:
        if nli_claim_result == "ENTAILMENT":
            failure_label = "WRONG_LABEL_SUPPORTS"
        elif nli_claim_result == "CONTRADICTION":
            failure_label = "WRONG_LABEL_REFUTES"
        elif nli_claim_result == "NEUTRAL":
            failure_label = "WRONG_LABEL_NOTENOUGHINFO"
    
    return failure_label


if __name__ == "__main__":
    for evidence in ["", "test"]:
        for rag_prediction in ["SUPPORTS", "REFUTES", "NOTENOUGHINFO"]:
            for kg_result in ["MISSING", "CONSISTENT", "CONFLICT"]:
                for nli_claim_result in ["NEUTRAL", "CONTRADICTION", "ENTAILMENT"]:
                    print(evidence, rag_prediction, kg_result, nli_claim_result, get_failure_label({
                        "evidence": evidence,
                        "prediction": rag_prediction,
                        "consistency_check": kg_result,
                        "entailment_check": {"claim": nli_claim_result}
                    }))