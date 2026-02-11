import json
from llama_index.core import KnowledgeGraphIndex, load_index_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
from llama_index.core import Document
import os
import tqdm
from pyvis.network import Network
from triplet_extractor import TripletExtractor
from llama_index.core import PromptTemplate
from sentence_transformers import SentenceTransformer
import numpy as np


class KnowledgeGraph:

    """
        reference: https://github.com/run-llama/llama_index/issues/13129
        reference: https://www.datacamp.com/tutorial/knowledge-graph-rag
        reference: https://github.com/run-llama/llama_index/issues/13129
        reference: https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledge_graph2/
    """

    def __init__(self, output_path, **kwargs):
        self.output_path = output_path
        self.kwargs = kwargs
        self.triplet_extractor = TripletExtractor(**kwargs)
        self.embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def create_KB_documents(self, knowledgebase):
        documents = []
        for text in knowledgebase:
            document = Document(text=text, extra_info={})
            documents.append(document)

        return documents
    
    def manual_check_triplets(self, knowledgebase):
        documents = self.create_KB_documents(knowledgebase)
        print("kb_documents size:", len(documents))
        print(documents[0])

        with open(f"{self.output_path}/KB_triplets.txt", "w+") as file:
            for document in tqdm.tqdm(documents):
                text = document.text
                triplets = self.triplet_extractor.extract_triplets(text)
                for triplet in triplets:
                    triplet = f"SUBJECT: {triplet[0]} || PREDICATE: {triplet[1]} || OBJECT: {triplet[2]}"
                    file.write(text + " || " + triplet + " \n")

    def index_nodes(self, triplet_documents):
        graph_store = SimpleGraphStore()
        storage_context = StorageContext.from_defaults(graph_store=graph_store)

        self.index = KnowledgeGraphIndex.from_documents(
            triplet_documents,
            max_triplets_per_chunk=10,
            include_embeddings=True,
            kg_triplet_extract_fn=self.triplet_extractor.extract_triplets,
            storage_context=storage_context,
            show_progress=True,
        )
        self.index.storage_context.persist(persist_dir=os.path.join(self.output_path, self.kwargs["kg_storage_dir"]))

    def build(self, knowledgebase):
        if os.path.exists(os.path.join(self.output_path, self.kwargs["kg_storage_dir"], 'index_store.json')):
            print("[MSG] Loading knowledge graph...")

            graph_store = SimpleGraphStore.from_persist_dir(os.path.join(self.output_path, self.kwargs["kg_storage_dir"]))
            storage_context = StorageContext.from_defaults(graph_store=graph_store, persist_dir=os.path.join(self.output_path, self.kwargs["kg_storage_dir"]))
            self.index = load_index_from_storage(storage_context)
        else:
            print("[MSG] Building knowledge graph...")

            documents = self.create_KB_documents(knowledgebase)
            print("kb_documents size:", len(documents))
            print(documents[0])

            self.index_nodes(documents)

        # prompt = self.kwargs["prompts"]["kg_consistency"]
        # self.query_engine = self.index.as_query_engine(
        #     text_qa_template=PromptTemplate(prompt),
        # )
        self.retriever_engine = self.index.as_retriever(include_text=False, similarity_top_k=self.kwargs["similarity_top_k"])
        print("[MSG] Knowledge graph is ready to go.")

    def encode_triplet_elements(self, triplet):
        embs = self.embedding_model.encode(triplet, convert_to_numpy=True, normalize_embeddings=True)
        s_emb, r_emb, o_emb = embs[0], embs[1], embs[2]

        return s_emb, r_emb, o_emb

    def check_triplet(self, q_s, q_r, q_o, db_s, db_r, db_o):
        # cosine computed as dot because embeddings normalized

        s_score = float(np.dot(q_s, db_s))
        r_score = float(np.dot(q_r, db_r))
        o_score = float(np.dot(q_o, db_o))

        if s_score < self.kwargs["kg_thresholds"]["subject_score"]: 
            return "MISSING"
        elif r_score < self.kwargs["kg_thresholds"]["relation_score"]:
            return "MISSING"
        elif o_score < self.kwargs["kg_thresholds"]["object_score"]:
            return "CONFLICT"
        
        return "CONSISTENT"

    def consistency_check(self, input_text):
        consistency_checks = []
        if len(input_text) > 0:
            # print("input_text:", input_text)
            triplets = self.triplet_extractor.extract_triplets(input_text)
            for sub, rel, obj in triplets:
                # print(f"{sub}, {rel}, {obj}")
                response_obj = self.retriever_engine.retrieve(f"""[subject:{sub}] - [predicate:{rel}] - [object:{obj}]""")
                ret_triplet_list = [eval(triplet) for resp in response_obj for triplet in resp.metadata["kg_rel_texts"]]
                # print("ret_triplet_list:", ret_triplet_list)
                ret_triplet_embed_list = map(self.encode_triplet_elements, ret_triplet_list)
                q_s, q_r, q_o = self.encode_triplet_elements((sub, rel, obj))
                status = [self.check_triplet(q_s, q_r, q_o, ds, dr, do) for ds, dr, do in ret_triplet_embed_list]
                # print("status:", status)
                if "CONSISTENT" in status:
                    prediction = "CONSISTENT"
                elif "CONFLICT" in status:
                    prediction = "CONFLICT"
                elif "MISSING" in status:
                    prediction = "MISSING"
                
                consistency_checks.append(prediction)

        if "CONFLICT" in consistency_checks:
            return "CONFLICT"
        elif "MISSING" in consistency_checks:
            return "MISSING"
        elif len(input_text) == 0:
            return "EMPTYINPUT"
        elif len(consistency_checks) == 0:
            return "NOTRIPLETS"
        else:
            return "CONSISTENT"

    def plot(self):
        g = self.index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.show(f'{self.output_path}/knowledge_gragh_plot.html')