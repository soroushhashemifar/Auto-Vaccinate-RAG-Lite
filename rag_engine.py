from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever, RecursiveRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate
import pickle
import faiss
import json
from triplet_extractor import TripletExtractor
import os
from pathlib import Path
from json_repair import repair_json
from entailment import EntailmentChecker
from llama_index.retrievers.bm25 import BM25Retriever
import Stemmer
from llama_index.core.postprocessor import SentenceTransformerRerank


class RAGEngine:

    def __init__(self, knowledge_base_pkl_path, knowledge_graph=None, **kwargs):
        self.kwargs = kwargs
        self.knowledge_graph = knowledge_graph

        with open(knowledge_base_pkl_path, 'rb') as f:
            content = pickle.load(f)
            self.knowledge_base = content["knowledge_base"]
            self.meta_data = content["meta_data"]

        self.build_nodes()
        # self.build_dense_retriever(self.kwargs["similarity_top_k"])
        self.build_bm25_retriever(self.kwargs["similarity_top_k"])
        self.build_query_engine(reranker=False)

        self.triplet_extractor = TripletExtractor(**kwargs)
        self.entailment_checker = EntailmentChecker(**kwargs)

    def build_nodes(self):
        """
            source: https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
        """

        if os.path.exists(os.path.join(self.kwargs["rag_storage_dir"], 'nodes.pkl')):
            print("[MSG] Loading nodes from disk...")
            with open(os.path.join(self.kwargs["rag_storage_dir"], 'nodes.pkl'), 'rb') as f:
                content = pickle.load(f)
                self.all_nodes = content["all_nodes"]
                self.all_nodes_dict = content["all_nodes_dict"]

            return

        print("[MSG] Building nodes from documents...")

        documents = [Document(text=sentence, extra_info=mdata) for sentence, mdata in zip(self.knowledge_base, self.meta_data)]

        node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=16)
        base_nodes = node_parser.get_nodes_from_documents(documents)
        self.all_nodes = base_nodes

        print("nodes size:", len(self.all_nodes))
        self.all_nodes_dict = {n.node_id: n for n in self.all_nodes}

        Path(self.kwargs["rag_storage_dir"]).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.kwargs["rag_storage_dir"], 'nodes.pkl'), 'wb') as f:
            pickle.dump({"all_nodes": self.all_nodes, "all_nodes_dict": self.all_nodes_dict}, f)        

    def build_dense_retriever(self, similarity_top_k, reindex=False):
        """
            source: https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/
            source: https://github.com/run-llama/llama_index/issues/6977
            source: https://github.com/run-llama/llama_index/issues/10631
            source: https://app.readytensor.ai/publications/retrieval-augmented-generation-using-llamaindex-faiss-and-openai-gpt-4-SfLlZniaZJ9C
        """

        if os.path.exists(os.path.join(self.kwargs["dense_retriever_storage"], 'index_store.json')) and not reindex:
            print("[MSG] Loading vector index...")
            
            vector_store = FaissVectorStore.from_persist_dir(self.kwargs["dense_retriever_storage"])
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.kwargs["dense_retriever_storage"])
            index = load_index_from_storage(storage_context)
        else:
            print("[MSG] Building dense vector index...", f"k = [{similarity_top_k}]")

            faiss_index = faiss.IndexFlatL2(384)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(self.all_nodes, storage_context=storage_context, show_progress=False)
            index.storage_context.persist(persist_dir=self.kwargs["dense_retriever_storage"])

        self.retriever = VectorIndexRetriever(
            index=index, 
            similarity_top_k=similarity_top_k
        )

    def build_bm25_retriever(self, similarity_top_k, reindex=False):
        """
            reference: https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/
        """
        
        if os.path.exists(self.kwargs["bm25_retriever_storage"]) and not reindex:
            print("[MSG] Loading BM25 retriever...")
            
            self.retriever = BM25Retriever.from_persist_dir(self.kwargs["bm25_retriever_storage"])
        else:
            print("[MSG] Building BM25 retriever...", f"k = [{similarity_top_k}]")

            self.retriever = BM25Retriever.from_defaults(
                nodes=self.all_nodes,
                similarity_top_k=similarity_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )
            self.retriever.persist(self.kwargs["bm25_retriever_storage"])

    def build_query_engine(self, similarity_posprocess=False, reranker=False, prompt_edit=False):
        """
            source: https://www.llamaindex.ai/blog/evaluating-rag-with-deepeval-and-llamaindex
            source: https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
        """

        print("[MSG] Building RAG query engine...")

        node_postprocessors = []
        if similarity_posprocess:
            node_postprocessors.append(
                SimilarityPostprocessor(
                    similarity_cutoff=self.kwargs["similarity_cutoff"], 
                    filter_empty=True,
                    filter_duplicates=True,
                    filter_similar=True
                )
            )
        elif reranker:
            node_postprocessors.append(
                SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
                    top_n=self.kwargs["reranker_top_n"]
                )
            )

        self.query_engine = RetrieverQueryEngine.from_args(
            self.retriever, 
            response_mode="compact_accumulate",
            text_qa_template=PromptTemplate(
                self.kwargs["rag_fact_verif_prompt"] if not prompt_edit else self.kwargs["rag_fact_verif_edit_prompt"]
            ),
            node_postprocessors=node_postprocessors
        )

        print("[MSG] RAG query engine is ready to go.")

    def query(self, query_str, add_entity_triplets=False, consistency_check=False, entailment_check=False, failure_check=False):
        response_object = self.query_engine.query(query_str)
        response = response_object.response.strip()
        if response == "Empty Response":
            # In case no node found by the retriever:
            # https://github.com/run-llama/llama_index/blob/fe72a2f5dbefb92d8c91cb460d4299de5637aa5a/llama-index-core/llama_index/core/response_synthesizers/base.py#L284

            response = {"prediction": "NOTENOUGHINFO"}
        else:
            response = json.loads(repair_json(response))

        response["claim"] = query_str
        response["retrieved_context"] = [{"text": n.node.text, 'doc_id': n.node.metadata['doc_id'], 'node_id': n.id_, 'score': n.score} for n in response_object.source_nodes]

        if add_entity_triplets and "evidence" in response:
            response["entity_relations"] = self.triplet_extractor.extract_triplets(response["evidence"])

        if self.knowledge_graph and consistency_check and "evidence" in response:
            response["consistency_check"] = self.knowledge_graph.consistency_check(response["evidence"])

        if entailment_check and "evidence" in response:
            response["entailment_check"] = self.entailment_checker.check(query_str, response["evidence"], response["retrieved_context"])

        return response