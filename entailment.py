from transformers import AutoModelForSequenceClassification, AutoTokenizer
from utils import singleton


@singleton
class EntailmentChecker:

    def __init__(self, **kwargs):
        self.device = kwargs["device"]
        self.nli_model = AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli').to(self.device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained('MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli')

    def check(self, query, response, retrieved_context):
        if len(retrieved_context) == 0:
            return {"query": "NEUTRAL", "response": "NEUTRAL"}
        
        query_entailment_label = [self.nli_model(self.tokenizer.encode(ctx['text'], query, return_tensors='pt').to(self.device)).logits.argmax(1).item() for ctx in retrieved_context]
        if all(list(map(lambda item: item == 1, query_entailment_label))):
            query_entailment_label = "NEUTRAL"
        elif 0 in query_entailment_label:
            query_entailment_label = "ENTAILMENT"
        else:
            query_entailment_label = "CONTRADICTION"

        rag_entailment_label = [self.nli_model(self.tokenizer.encode(ctx['text'], response, return_tensors='pt').to(self.device)).logits.argmax(1).item() for ctx in retrieved_context]
        if all(list(map(lambda item: item == 1, rag_entailment_label))):
            rag_entailment_label = "NEUTRAL"
        elif 0 in rag_entailment_label:
            rag_entailment_label = "ENTAILMENT"
        else:
            rag_entailment_label = "CONTRADICTION"

        return {"query": query_entailment_label, "response": rag_entailment_label}


if __name__ == "__main__":
    ec = EntailmentChecker(device="cuda")

    query = "What color jersey has Denver 0-4?"
    response = "Denver is 0-4 in Super Bowls when wearing orange jerseys."
    retrieved_context = [{'text': 'As the designated home team in the annual rotation between AFC and NFC teams, the Broncos elected to wear their road white jerseys with matching white pants. Elway stated, "We\'ve had Super Bowl success in our white uniforms." The Broncos last wore matching white jerseys and pants in the Super Bowl in Super Bowl XXXIII, Elway\'s last game as Denver QB, when they defeated the Atlanta Falcons 34–19. In their only other Super Bowl win in Super Bowl XXXII, Denver wore blue jerseys, which was their primary color at the time.', 'node_id': '32b08193-0e01-4fec-ad3e-1c927846120a', 'score': 7.811087131500244}, {'text': 'The flagship stations of each station in the markets of each team will carry their local play-by-play calls. In Denver, KOA (850 AM) and KRFX (103.5 FM) will carry the game, with Dave Logan on play-by-play and Ed McCaffrey on color commentary. In North Carolina, WBT (1110 AM) will carry the game, with Mick Mixon on play-by-play and Eugene Robinson and Jim Szoke on color commentary.', 'node_id': 'f3f97bcc-d4b5-45e3-a435-413e2e0cdbec', 'score': 4.739192008972168}, {'text': "They also lost Super Bowl XXI when they wore white jerseys, but they are 0-4 in Super Bowls when wearing orange jerseys, losing in Super Bowl XII, XXII, XXIV, and XLVIII. The only other AFC champion team to have worn white as the designated home team in the Super Bowl was the Pittsburgh Steelers; they defeated the Seattle Seahawks 21–10 in Super Bowl XL 10 seasons prior. The Broncos' decision to wear white meant the Panthers would wear their standard home uniform: black jerseys with silver pants.", 'node_id': '3c79e5bb-599a-4377-ae91-9fa9eea9f8e3', 'score': 3.8320493698120117}, {'text': 'Currently, New Jersey, Rhode Island and Delaware are the only U.S. states where ABC does not have a locally licensed affiliate (New Jersey is served by New York City O&O WABC-TV and Philadelphia O&O WPVI-TV; Rhode Island is served by New Bedford, Massachusetts-licensed WLNE; and Delaware is served by WPVI and Salisbury, Maryland affiliate WMDT).', 'node_id': '8d5158e7-db3a-4bc7-9e75-a8b0f9e3d093', 'score': 3.7186264991760254}, {'text': 'Plastid differentiation is not permanent, in fact many interconversions are possible. Chloroplasts may be converted to chromoplasts, which are pigment-filled plastids responsible for the bright colors seen in flowers and ripe fruit. Starch storing amyloplasts can also be converted to chromoplasts, and it is possible for proplastids to develop straight into chromoplasts. Chromoplasts and amyloplasts can also become chloroplasts, like what happens when a carrot or a potato is illuminated.', 'node_id': '9e83aec0-2a6f-48b9-85b1-4aac5f1a5724', 'score': 3.351297378540039}, {'text': "Detailed statistical investigation has not suggested the function of ctenophores' bioluminescence nor produced any correlation between its exact color and any aspect of the animals' environments, such as depth or whether they live in coastal or mid-ocean waters.", 'node_id': '817eb5e1-0bca-4ab8-817e-878cdd41f3ce', 'score': 3.3178701400756836}, {'text': "The 1960s would be marked by the rise of family-oriented series in an attempt by ABC to counterprogram its established competitors, but the decade was also marked by the network's gradual transition to color. On September 30, 1960, ABC premiered The Flintstones, another example of counterprogramming; although the animated series from William Hanna and Joseph Barbera was filmed in color from the beginning, it was initially broadcast in black-and-white, as ABC had not made the necessary technical upgrades to broadcast its programming in color at the time.", 'node_id': '636bc072-7cf5-4e7b-8635-e0b6b7c2a1d5', 'score': 2.9968080520629883}, {'text': "The Broncos took an early lead in Super Bowl 50 and never trailed. Newton was limited by Denver's defense, which sacked him seven times and forced him into three turnovers, including a fumble which they recovered for a touchdown. Denver linebacker Von Miller was named Super Bowl MVP, recording five solo tackles, 2½ sacks, and two forced fumbles.", 'node_id': '186fe47b-c1f0-4958-b032-56aff1bd43b4', 'score': 2.948000431060791}, {'text': 'Another major division within Islamism is between what Graham E. Fuller has described as the fundamentalist "guardians of the tradition" (Salafis, such as those in the Wahhabi movement) and the "vanguard of change and Islamic reform" centered around the Muslim Brotherhood.', 'node_id': 'fe54841f-12da-4bcd-840a-0415ea515b2d', 'score': 2.8593292236328125}, {'text': "In the early 1970s, ABC completed its transition to color; the decade as a whole would mark a turning point for ABC, as it began to pass CBS and NBC in the ratings to become the first place network. It also began to use behavioral and demographic data to better determine what types of sponsors to sell advertising slots to and provide programming that would appeal towards certain audiences. ABC's gains in audience share were greatly helped by the fact that several smaller markets had grown large enough to allow full-time affiliations from all three networks.", 'node_id': '4385c3fa-b31e-4423-afb9-c4d165dd57f9', 'score': 2.8047642707824707}]

    print(ec.check(query, response, retrieved_context))