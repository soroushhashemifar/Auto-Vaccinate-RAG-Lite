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


class RAGEngine:

    def __init__(self, knowledge_base_pkl_path, knowledge_graph=None, **kwargs):
        self.kwargs = kwargs
        self.knowledge_graph = knowledge_graph

        with open(knowledge_base_pkl_path, 'rb') as f:
            content = pickle.load(f)
            self.knowledge_base = content["knowledge_base"]
            self.meta_data = content["meta_data"]

        self.build_nodes()
        self.build_index()
        self.build_query_engine()

        self.triplet_extractor = TripletExtractor(**kwargs)
        self.entailment_checker = EntailmentChecker()

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

        # build parent chunks via NodeParser
        node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=32)
        base_nodes = node_parser.get_nodes_from_documents(documents)
        self.all_nodes = base_nodes

        # # define smaller child chunks
        # sub_chunk_sizes = [32, 64, 128]
        # sub_node_parsers = [
        #     SentenceSplitter(chunk_size=c, chunk_overlap=16) for c in sub_chunk_sizes
        # ]
        # self.all_nodes = []
        # for base_node in base_nodes:
        #     for n in sub_node_parsers:
        #         sub_nodes = n.get_nodes_from_documents([base_node])
        #         sub_inodes = [
        #             IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
        #         ]
        #         self.all_nodes.extend(sub_inodes)

        #     # also add original node to node
        #     original_node = IndexNode.from_text_node(base_node, base_node.node_id)
        #     self.all_nodes.append(original_node)

        print("nodes size:", len(self.all_nodes))
        self.all_nodes_dict = {n.node_id: n for n in self.all_nodes}

        Path(self.kwargs["rag_storage_dir"]).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.kwargs["rag_storage_dir"], 'nodes.pkl'), 'wb') as f:
            pickle.dump({"all_nodes": self.all_nodes, "all_nodes_dict": self.all_nodes_dict}, f)

    def build_index(self):
        """
            source: https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/
            source: https://github.com/run-llama/llama_index/issues/6977
            source: https://github.com/run-llama/llama_index/issues/10631
            source: https://app.readytensor.ai/publications/retrieval-augmented-generation-using-llamaindex-faiss-and-openai-gpt-4-SfLlZniaZJ9C
        """

        if os.path.exists(os.path.join(self.kwargs["rag_storage_dir"], 'index_store.json')):
            print("[MSG] Loading vector index...")
            
            vector_store = FaissVectorStore.from_persist_dir(self.kwargs["rag_storage_dir"])
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.kwargs["rag_storage_dir"])
            self.index = load_index_from_storage(storage_context)

            return

        print("[MSG] Building vector index...")

        faiss_index = faiss.IndexFlatL2(384)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex(self.all_nodes, storage_context=storage_context, show_progress=True)
        self.index.storage_context.persist(persist_dir=self.kwargs["rag_storage_dir"])

    def build_retriever(self):
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.kwargs["similarity_top_k"])

        # # build RecursiveRetriever
        # retriever = RecursiveRetriever(
        #     "vector",
        #     retriever_dict={"vector": retriever},
        #     node_dict=self.all_nodes_dict,
        #     verbose=False,
        # )

        return retriever

    def build_query_engine(self):
        """
            source: https://www.llamaindex.ai/blog/evaluating-rag-with-deepeval-and-llamaindex
            source: https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
        """

        print("[MSG] Building RAG query engine...")

        retriever = self.build_retriever()
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever, #retriever_chunk,
            response_mode="compact_accumulate",
            text_qa_template=PromptTemplate(self.kwargs["text_qa_template"]),
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=self.kwargs["similarity_cutoff"], 
                                        filter_empty=True,
                                        filter_duplicates=True,
                                        filter_similar=True,)
                ],
        )

        print("[MSG] RAG query engine is ready to go.")
    
    def failure_label(self, response):
        rag_prediction = response["prediction"]
        kg_result = response["consistency_check"]
        nli_evidence_result = response["entailment_check"]["evidence"]
        nli_claim_result = response["entailment_check"]["claim"]

        failure_label = None
        if rag_prediction == "SUPPORTS" and kg_result == "CONSISTENT" and nli_evidence_result == "ENTAILMENT" and nli_claim_result == "ENTAILMENT":
            failure_label = "RAG_SUCCESS"
        elif rag_prediction == "REFUTES" and kg_result == "CONSISTENT" and nli_evidence_result == "ENTAILMENT" and nli_claim_result == "CONTRADICTION":
            failure_label = "RAG_SUCCESS"
        elif rag_prediction == "NOTENOUGHINFO" and kg_result == "MISSING" and nli_claim_result == "NEUTRAL":
            failure_label = "RAG_SUCCESS"
        elif rag_prediction == "NOTENOUGHINFO" and kg_result == "MISSING" and nli_claim_result != "NEUTRAL":
            failure_label = "RAG_WRONG_NEI"
        elif rag_prediction != "NOTENOUGHINFO" and kg_result == "MISSING" and nli_claim_result == "NEUTRAL":
            failure_label = "RAG_HALLUCINATION"
        elif (rag_prediction == "NOTENOUGHINFO" and (kg_result != "MISSING" or nli_claim_result != "NEUTRAL")) or kg_result != "CONSISTENT" or nli_evidence_result == "CONTRADICTION":
            failure_label = "RAG_HALLUCINATION"
        elif nli_claim_result == "NEUTRAL":
            failure_label = "RETRIEVAL_MISSING"
        
        return failure_label

    def query(self, query_str, add_entity_triplets=False, consistency_check=False, entailment_check=False):
        response_object = self.query_engine.query(query_str)
        response = response_object.response.strip()
        if response == "Empty Response":
            # In case no node found by the retriever:
            # https://github.com/run-llama/llama_index/blob/fe72a2f5dbefb92d8c91cb460d4299de5637aa5a/llama-index-core/llama_index/core/response_synthesizers/base.py#L284

            response = {"prediction": "NOTENOUGHINFO"}
        else:
            response = json.loads(repair_json(response))

        response["claim"] = query_str
        response["retrieved_context"] = [{"text": n.node.text, 'doc_id': n.node.metadata['doc_id'], 'node_id': n.id_} for n in response_object.source_nodes]

        if add_entity_triplets and "evidence" in response:
            response["entity_relations"] = self.triplet_extractor.extract_triplets(response["evidence"])

        if self.knowledge_graph and consistency_check and "evidence" in response:
            response["consistency_check"] = self.knowledge_graph.consistency_check(response["evidence"])

        if entailment_check and "evidence" in response:
            response["entailment_check"] = self.entailment_checker.check(query_str, response["evidence"], response["retrieved_context"])

        return response
