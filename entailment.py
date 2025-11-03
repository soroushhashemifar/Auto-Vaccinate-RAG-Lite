from transformers import pipeline
from utils import singleton


@singleton
class EntailmentChecker:

    def __init__(self, **kwargs):
        self.model = pipeline("text-classification", model="tasksource/ModernBERT-base-nli", device=kwargs["device"])

    def check(self, claim, rag_response, retrieved_context):
        rag_entailment_label = "NEUTRAL"
        for item in retrieved_context:
            text_input = item['text']

            claim_pred = self.model([{"text": text_input, "text_pair": claim}], top_k=None)[0]
            claim_results = [(p['label'].upper(), p['score']) for p in claim_pred]
            claim_entailment_label = max(claim_results, key=lambda item: item[1])[0]

            if claim_entailment_label != "NEUTRAL":
                if len(rag_response) != 0:
                    rag_pred = self.model([{"text": text_input, "text_pair": rag_response}], top_k=None)[0]
                    rag_results = [(p['label'].upper(), p['score']) for p in rag_pred]
                    rag_entailment_label = max(rag_results, key=lambda item: item[1])[0]
                
                break

        return {"claim": claim_entailment_label, "response": rag_entailment_label}
    

if __name__ == "__main__":
    ec = EntailmentChecker(device="cpu")

    claim = "Anne Rice was born in New Jersey."
    rag_response = "The context states that Anne Rice was born in New Orleans, not in New Jersey."
    retrieved_context = [
        {"text": "Anne Rice ( born Howard Allen Frances O'Brien ; October 4 , 1941 ) is an American author of gothic fiction , Christian literature , and erotica . She is perhaps best known for her popular and influential series of novels , The Vampire Chronicles , revolving around the central character of Lestat . Books from The Vampire Chronicles were the subject of two film adaptations , Interview with the Vampire in 1994 , and Queen of the Damned in 2002 .   Born in New Orleans , Rice spent much of her early life there before moving to Texas , and later to San Francisco ."},
        {"text": "metropolitan areas of Fairfax County , Virginia , Montgomery County , Maryland , and Prince George 's County , Maryland ; Queens , New York ; Long Island , New York ; Newark , New Jersey , Plainfield , New Jersey ; Jersey City , New Jersey ; Elizabeth , New Jersey ; the Boston , Massachusetts area ; Charlotte , North Carolina ; and Houston , Texas . There is also a presence of MS-13 in Toronto , Ontario , Canada .   Members of MS are characterised by tattoos covering the body and also often the face , and by the use of their own sign language ."},
        {"text": "The New Jersey Turnpike ( NJTP ) , colloquially known to New Jerseyans as `` the Turnpike '' , is a toll road in New Jersey , maintained by the New Jersey Turnpike Authority . According to the International Bridge , Tunnel and Turnpike Association , the Turnpike is the nation 's sixth-busiest toll road and is one of the most heavily traveled highways in the United States ."},
        {"text": 'In addition to her vampire novels , Rice has authored books such as The Feast of All Saints ( adapted for television in 2001 ) and Servant of the Bones , which formed the basis of a 2011 comic book miniseries . Several books from The Vampire Chronicles have been adapted as comics by various publishers . Rice has also authored erotic fiction under the pen names Anne Rampling and A. N. Roquelaure , including Exit to Eden , which was later adapted into a 1994 film .'},
        {"text": 'Camden is a city in Camden County , New Jersey . Camden is located directly across the Delaware River from Philadelphia , Pennsylvania . At the 2010 United States Census , the city had a population of 77,344 . [ 10 ] [ 12 ] [ 13 ] Camden is the 12th most populous municipality in New Jersey . The city was incorporated on February 13 , 1828 . On March 13 , 1844 , Camden became a county seat in New Jersey .'}
        ]

    print(ec.check(claim, rag_response, retrieved_context))