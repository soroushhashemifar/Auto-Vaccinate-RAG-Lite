import tqdm
import json
import re
import pickle
import os
from drqa.retriever import utils


class WikipagesKnowledgeBase:

    def create_inverse_evidence_map(self, fever_json_path):
        with open(fever_json_path, 'r') as json_file:
            json_list = list(json_file)

            inverse_evidence_map = {}
            for json_str in tqdm.tqdm(json_list):
                result = json.loads(json_str)

                evidences = result['evidence']
                transformed_evidences = []
                for evidence in evidences:
                    if evidence[0][2] is not None:
                        claim_id_list = inverse_evidence_map.get(evidence[0][2], [])
                        claim_id_list.append(result['id'])
                        inverse_evidence_map[evidence[0][2]] = claim_id_list

        print("inverse_evidence_map size:", len(inverse_evidence_map))

        return inverse_evidence_map

    def iter_files(self, path):
        """
            Walk through all files located under a root path.
        """
        if os.path.isfile(path):
            yield path
        elif os.path.isdir(path):
            for dirpath, _, filenames in os.walk(path):
                for f in filenames:
                    yield os.path.join(dirpath, f)
        else:
            raise RuntimeError('Path %s is invalid' % path)

    def get_contents(self, filename):
        """
            Parse the contents of a file. Each line is a JSON encoded document.
        """
        documents = []
        with open(filename) as f:
            for line in f:
                # Parse document
                doc = json.loads(line)
                if not doc:
                    continue

                # Add the document
                documents.append((utils.normalize(doc['id']), doc['text']))

        return documents

    def convert_brc(self, string):
        """
            source: https://github.com/easonnie/combine-FEVER-NSMN/blob/master/src/utils/fever_db.py
        """

        string = re.sub('-LRB-', '(', string)
        string = re.sub('-RRB-', ')', string)
        string = re.sub('-LSB-', '[', string)
        string = re.sub('-RSB-', ']', string)
        string = re.sub('-LCB-', '{', string)
        string = re.sub('-RCB-', '}', string)
        string = re.sub('-COLON-', ':', string)
        return string

    def preprocess(self, text):
        text = text.replace("\t", " ").replace("\n", " ")
        # text = text.split('.')[0]
        text = self.convert_brc(text)
        text = text.strip()

        return text

    def create_knowledge_base(self, wikipages_dir_path, inverse_evidence_map, target_read_slices):
        filtered_evidences = []

        num_read_slices = 0
        for f in tqdm.tqdm(self.iter_files(wikipages_dir_path)):
            documents = self.get_contents(f)
            for evidence_id in inverse_evidence_map.keys():
                found_evidences = list(filter(lambda item: item[0] == evidence_id, documents))
                if len(found_evidences) > 0:
                    filtered_evidences.extend(found_evidences)

            num_read_slices += 1

            if target_read_slices > -1 and num_read_slices == target_read_slices:
                break

        print("filtered_evidences size:", len(filtered_evidences))

        knowledge_base = []
        meta_data = []
        for evidence in filtered_evidences:
            # sentences = re.split(r'\n\d+\t', "\n" + evidence[1])
            # sentences = [preprocess(sentence) for sentence in sentences]
            # sentences = list(filter(lambda sentence: len(sentence) > 0, sentences))

            text = self.preprocess(evidence[1])

            knowledge_base.append(text)
            meta_data.append({"doc_id": evidence[0]})
            # for idx, sentence in enumerate(sentences):
            #     meta_data.append({"doc_id": evidence[0], "sentence_id": idx})

        print("knowledge_base size:", len(knowledge_base))

        return knowledge_base, meta_data

    def build(self, fever_json_path, wikipages_dir_path, target_read_slices):
        inverse_evidence_map = self.create_inverse_evidence_map(fever_json_path)
        knowledge_base, meta_data = self.create_knowledge_base(wikipages_dir_path, inverse_evidence_map, target_read_slices)

        print(knowledge_base[0])
        print(meta_data[0])

        with open('./wikipages_knowledge_base.pkl', 'wb') as f:
            pickle.dump({"knowledge_base": knowledge_base, "meta_data": meta_data}, f)


if __name__ == "__main__":
    WikipagesKnowledgeBase().build("./shared_task_dev.jsonl", "./wiki-pages", 5)