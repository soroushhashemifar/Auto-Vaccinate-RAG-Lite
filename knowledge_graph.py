from llama_index.core import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core import StorageContext
import re
from llama_index.core import Document
import os
import tqdm
from pyvis.network import Network

from triplet_extractor import TripletExtractor


class WikiMoviesKnowledgeGraph:

    """
        source: https://github.com/run-llama/llama_index/issues/13129
        source: https://www.datacamp.com/tutorial/knowledge-graph-rag
        source: https://github.com/run-llama/llama_index/issues/13129
        source: https://developers.llamaindex.ai/python/examples/index_structs/knowledge_graph/knowledge_graph2/
    """

    def __init__(self, **kwargs):
        graph_store = SimpleGraphStore()
        self.storage_context = StorageContext.from_defaults(graph_store=graph_store)

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
            # TODO: extracts 8852 a out of 100000, WHY?
            parts = re.findall(r'\d+\s([\w\s\d]+\?)[\s\t]+([\w\s\d,;:\.\!\?\-\_]+)', line.strip())
            if len(parts) > 0:
                text = " ".join(parts[0])
                # triplets = self.triplet_extractor.extract_triplets(text)
                # for subj, rel, obj in triplets:
                #     document = Document(text=f"{subj} {rel} {obj}", metadata={"subject": subj, "relationship": rel, "object": obj})
                #     documents.append(document)

                #     if len(documents) == cutoff:
                #         break

                document = Document(text=text, metadata={"question": parts[0][0], "answer": parts[0][1]})
                documents.append(document)

                if cutoff > -1 and len(documents) == cutoff:
                    break

        return documents

    # def extract_triplets(self, input_text):
    #     subject = re.findall(r'subject\:\s(.+)\n', input_text)[0]
    #     relationship = re.findall(r'relationship\:\s(.+)\n', input_text)[0]
    #     object_ = re.findall(r'object\:\s(.+)\n', input_text)[0]

    #     return [(subject, relationship, object_)]

    def build(self, wikimovies_dir_path):
        # triplet_documents = self.load_wikimovies_triplets(wikimovies_dir_path, 10000)
        triplet_documents = self.create_wikimovies_triplets(wikimovies_dir_path, 100)
        print("triplet_documents size:", len(triplet_documents))
        print(triplet_documents[0])

        self.index = KnowledgeGraphIndex.from_documents(
            triplet_documents,
            max_triplets_per_chunk=5,
            include_embeddings=True,
            kg_triplet_extract_fn=self.triplet_extractor.extract_triplets, #self.extract_triplets,
            storage_context=self.storage_context,
            show_progress=True,
        )

        # retriever = self.index.as_retriever(
        #     include_text=True,
        #     similarity_top_k=2,
        # )

    def query(self):
        # retriever.retrieve("Fox 2000 Pictures released the film Soul Food.")
        pass

    def plot(self):
        g = self.index.get_networkx_graph()
        net = Network(notebook=True, cdn_resources="in_line", directed=True)
        net.from_nx(g)
        net.show('./knowledge_gragh_plot.html')