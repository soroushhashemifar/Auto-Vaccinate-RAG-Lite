from llama_index.core import KnowledgeGraphIndex, load_index_from_storage
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
import re
from llama_index.core import Document
import os
import tqdm
from pyvis.network import Network
from triplet_extractor import TripletExtractor
from llama_index.core import PromptTemplate
import pickle


class WikiMoviesKnowledgeGraph:

    """
        reference: https://github.com/run-llama/llama_index/issues/13129
        reference: https://www.datacamp.com/tutorial/knowledge-graph-rag
        reference: https://github.com/run-llama/llama_index/issues/13129
        reference: https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledge_graph2/
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.triplet_extractor = TripletExtractor(**kwargs)

    def load_wikimovies_triplets(self, wikimovies_dir_path, cutoff):
        with open(os.path.join(wikimovies_dir_path, "knowledge_source/full/full_kb.txt")) as f:
            documents = []
            for line in f:
                parts = re.findall(r'(\d+)\s([\w\s\d]+)\s(\w+\_\w+)\s([\w\s\d,;:\.\!\?\-\_]+)', line.strip())
                if len(parts) > 0:
                    subj, rel, obj = parts[0][1:]
                    document = Document(text=f"{subj} {rel.replace('_', ' ')} {obj}", metadata={"subject": subj, "relationship": rel, "object": obj})
                    documents.append(document)

                    if cutoff > -1 and len(documents) == cutoff:
                        break

        return documents

    def create_wikimovies_triplets(self, wikimovies_dir_path, cutoff):
        with open(os.path.join(wikimovies_dir_path, "questions/full/full_qa_dev.txt"), "r") as file:
            content = file.readlines()

        documents = []
        for line in tqdm.tqdm(content):
            parts = re.findall(r'\d+\s([\w\s\d]+\?)[\s\t]+([\w\s\d,;:\.\!\?\-\_]+)', line.strip())
            if len(parts) > 0:
                text = " ".join(parts[0])
                document = Document(text=text, metadata={"question": parts[0][0], "answer": parts[0][1]})
                documents.append(document)

                if cutoff > -1 and len(documents) == cutoff:
                    break

        return documents

    def load_KB_documents(self, knowledge_base_pkl_path):
        with open(knowledge_base_pkl_path, 'rb') as f:
            content = pickle.load(f)
            self.knowledge_base = content["knowledge_base"]
            self.meta_data = content["meta_data"]

        documents = []
        for doc, mdata in zip(self.knowledge_base, self.meta_data):
            for sentence in doc.split(" . "):
                document = Document(text=sentence, extra_info=mdata)
                documents.append(document)

        return documents
    
    def manual_check_triplets(self, wikimovies_dir_path, knowledge_base_pkl_path, cutoff=-1):
        triplet_documents = self.create_wikimovies_triplets(wikimovies_dir_path, cutoff)
        print("triplet_documents size:", len(triplet_documents))
        print(triplet_documents[0])

        kb_documents = self.load_KB_documents(knowledge_base_pkl_path)
        triplet_documents.extend(kb_documents)
        print("kb_documents size:", len(kb_documents))
        print(kb_documents[0])

        with open("./KB_triplets.txt", "w+") as file:
            for document in tqdm.tqdm(triplet_documents):
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
            max_triplets_per_chunk=5,
            include_embeddings=True,
            kg_triplet_extract_fn=self.triplet_extractor.extract_triplets, #self.extract_triplets,
            storage_context=storage_context,
            show_progress=False,
        )
        self.index.storage_context.persist(persist_dir=self.kwargs["kg_storage_dir"])

    def build(self, wikimovies_dir_path, knowledge_base_pkl_path, cutoff=-1):
        if os.path.exists(os.path.join(self.kwargs["kg_storage_dir"], 'index_store.json')):
            print("[MSG] Loading knowledge graph...")

            graph_store = SimpleGraphStore.from_persist_dir(self.kwargs["kg_storage_dir"])
            storage_context = StorageContext.from_defaults(graph_store=graph_store, persist_dir=self.kwargs["kg_storage_dir"])
            self.index = load_index_from_storage(storage_context)
        else:
            print("[MSG] Building knowledge graph...")

            triplet_documents = self.create_wikimovies_triplets(wikimovies_dir_path, cutoff)
            print("triplet_documents size:", len(triplet_documents))
            print(triplet_documents[0])

            kb_documents = self.load_KB_documents(knowledge_base_pkl_path)
            triplet_documents.extend(kb_documents)
            print("kb_documents size:", len(kb_documents))
            print(kb_documents[0])

            self.index_nodes(triplet_documents)

        self.query_engine = self.index.as_query_engine(
            text_qa_template=PromptTemplate(self.kwargs["KG_completion_prompt"]),
        )
        self.retriever_engine = self.index.as_retriever(include_text=True, similarity_top_k=self.kwargs["similarity_top_k"])
        print("[MSG] Knowledge graph is ready to go.")

    def consistency_check(self, input_text):
        consistency_checks = []
        if len(input_text) > 0:
            triplets = self.triplet_extractor.extract_triplets(input_text)
            for sub, rel, obj in triplets:
                response_obj = self.query_engine.query(f"""[subject:{sub}] - [predicate:{rel}] - [object:{obj}]""")
                prediction = response_obj.response.strip()

                context = " ".join([node.dict()['node']['text'] for node in response_obj.source_nodes])
                if sub not in context or obj not in context:
                    prediction = "MISSING"

                if "CONFLICT" in prediction:
                    prediction = "CONFLICT"
                elif "CONSISTENT" in prediction:
                    prediction = "CONSISTENT"

                consistency_checks.append(prediction)

        if "CONFLICT" in consistency_checks:
            return "CONFLICT"
        elif "MISSING" in consistency_checks:
            return "MISSING"
        else:
            return "CONSISTENT"

    def plot(self):
        g = self.index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.show('./knowledge_gragh_plot.html')