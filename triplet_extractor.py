from transformers import pipeline
from utils import singleton


@singleton
class TripletExtractor:

    """
        reference: https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledge_graph2/
    """

    def __init__(self, **kwargs):
        self.triplet_extractor = pipeline(
            "text2text-generation",
            model="Babelscape/rebel-large",
            tokenizer="Babelscape/rebel-large",
            device=kwargs["device"],
        )

    def extract_triplets(self, input_text):
        text = self.triplet_extractor.tokenizer.batch_decode(
            [
                self.triplet_extractor(
                    input_text, return_tensors=True, return_text=False
                )[0]["generated_token_ids"]
            ]
        )[0]

        triplets = []
        relation, subject, relation, object_ = "", "", "", ""
        text = text.strip()
        current = "x"
        for token in (
            text.replace("<s>", "")
            .replace("<pad>", "")
            .replace("</s>", "")
            .split()
        ):
            if token == "<triplet>":
                current = "t"
                if relation != "":
                    triplets.append(
                        (subject.strip(), relation.strip(), object_.strip())
                    )
                    relation = ""
                subject = ""
            elif token == "<subj>":
                current = "s"
                if relation != "":
                    triplets.append(
                        (subject.strip(), relation.strip(), object_.strip())
                    )
                object_ = ""
            elif token == "<obj>":
                current = "o"
                relation = ""
            else:
                if current == "t":
                    subject += " " + token
                elif current == "s":
                    object_ += " " + token
                elif current == "o":
                    relation += " " + token

        if subject != "" and relation != "" and object_ != "":
            triplets.append((subject.strip(), relation.strip(), object_.strip()))

        return triplets