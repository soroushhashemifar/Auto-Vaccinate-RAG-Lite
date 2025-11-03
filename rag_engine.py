import torch
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
# from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
import time


class RAGEngine:

    def __init__(self, knowledge_base_pkl_path, knowledge_graph=None, **kwargs):
        self.kwargs = kwargs
        self.knowledge_graph = knowledge_graph
        self.knowledge_base_pkl_path = knowledge_base_pkl_path

        self.build_nodes()

        self.query_engines = {}
        for retriever_type in ["dense", "bm25"]:
            for topk in [5, 10, 20]:
                for reranker in [True, False]:
                    for prompt_edit in [False, "WP", "WR"]:
                        if retriever_type == "dense":
                            retriever = self.build_dense_retriever(topk)
                        elif retriever_type == "bm25":
                            retriever = self.build_bm25_retriever(topk)
                        
                        self.query_engines[f"{retriever_type}_{topk}_{reranker}_{prompt_edit}"] = self.build_query_engine(retriever, reranker=reranker, prompt_edit=prompt_edit)

        self.triplet_extractor = TripletExtractor(**kwargs)
        self.entailment_checker = EntailmentChecker(**kwargs)

        # self.question_tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
        # self.question_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct").to(kwargs["device"])

    def build_nodes(self):
        """
            source: https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
        """

        # if os.path.exists(os.path.join(self.kwargs["rag_storage_dir"], 'nodes.pkl')):
        #     print("[MSG] Loading nodes from disk...")
        #     with open(os.path.join(self.kwargs["rag_storage_dir"], 'nodes.pkl'), 'rb') as f:
        #         content = pickle.load(f)
        #         self.all_nodes = content["all_nodes"]
        #         self.all_nodes_dict = content["all_nodes_dict"]

        #     return

        print("[MSG] Building nodes from documents...")

        with open(self.knowledge_base_pkl_path, 'rb') as f:
            content = pickle.load(f)
            knowledge_base = content["knowledge_base"]
            meta_data = content["meta_data"]

        node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=16)

        documents = [Document(text=sentence, extra_info=mdata) for sentence, mdata in zip(knowledge_base, meta_data)]
        base_nodes = node_parser.get_nodes_from_documents(documents)
        self.all_nodes = base_nodes

        print("nodes size:", len(self.all_nodes))
        self.all_nodes_dict = {n.node_id: n for n in self.all_nodes}

        # Path(self.kwargs["rag_storage_dir"]).mkdir(parents=True, exist_ok=True)
        # with open(os.path.join(self.kwargs["rag_storage_dir"], 'nodes.pkl'), 'wb') as f:
        #     pickle.dump({"all_nodes": self.all_nodes, "all_nodes_dict": self.all_nodes_dict}, f)        

    def build_dense_retriever(self, similarity_top_k, reindex=False):
        """
            source: https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/
            source: https://github.com/run-llama/llama_index/issues/6977
            source: https://github.com/run-llama/llama_index/issues/10631
            source: https://app.readytensor.ai/publications/retrieval-augmented-generation-using-llamaindex-faiss-and-openai-gpt-4-SfLlZniaZJ9C
        """

        if os.path.exists(os.path.join(self.kwargs["dense_retriever_storage"], 'index_store.json')) and not reindex:
            print(f"[MSG] Loading Dense (k={similarity_top_k}) retriever...")
            
            vector_store = FaissVectorStore.from_persist_dir(self.kwargs["dense_retriever_storage"])
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.kwargs["dense_retriever_storage"])
            index = load_index_from_storage(storage_context)
        else:
            print(f"[MSG] Building Dense (k={similarity_top_k}) retriever...")

            faiss_index = faiss.IndexFlatL2(384)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(self.all_nodes, storage_context=storage_context, show_progress=False)
            index.storage_context.persist(persist_dir=self.kwargs["dense_retriever_storage"])

        retriever = VectorIndexRetriever(
            index=index, 
            similarity_top_k=similarity_top_k
        )

        return retriever

    def build_bm25_retriever(self, similarity_top_k, reindex=False):
        """
            reference: https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/
        """
        
        if os.path.exists(self.kwargs["bm25_retriever_storage"]) and not reindex:
            print(f"[MSG] Loading BM25 (k={similarity_top_k}) retriever...")
            
            retriever = BM25Retriever.from_persist_dir(self.kwargs["bm25_retriever_storage"])
        else:
            print(f"[MSG] Building BM25 (k={similarity_top_k}) retriever...")

            retriever = BM25Retriever.from_defaults(
                nodes=self.all_nodes,
                similarity_top_k=similarity_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )
            retriever.persist(self.kwargs["bm25_retriever_storage"])

        return retriever

    def build_query_engine(self, retriever, similarity_posprocess=False, reranker=False, prompt_edit=False):
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

        if prompt_edit == False:
            prompt = self.kwargs["rag_fact_verif_prompt"]  
        else:
            prompt = self.kwargs[f"prompt_{prompt_edit}"]

        query_engine = RetrieverQueryEngine.from_args(
            retriever, 
            response_mode="compact_accumulate",
            text_qa_template=PromptTemplate(prompt),
            node_postprocessors=node_postprocessors
        )

        print(f"[MSG] RAG query engine is ready to go. (reranker={reranker}, prompt_edit={prompt_edit})")

        return query_engine
    
    def to_question(self, query_str):
        message = f"""
        Convert the following claim into a yes/no question. 
        Do not start the question with 'Is it true that ...'.
        Contain every piece of information from the claim in your question. 
        Do not use any information from your knowlede in your question, only use the claim text.

        {query_str}
        """

        response = Settings.llm.chat([ChatMessage(role="user", content=message)])
        return response.message.blocks[0].text

    def query(self, query_str, params={}, add_entity_triplets=False, consistency_check=False, entailment_check=False, failure_check=False):
        retriever_type = params.get("retriever", "bm25")
        topk = params.get("topk", 5)
        reranker = params.get("reranker", False)
        prompt_edit = params.get("prompt_edit", False)
        reindex = params.get("reindex", False)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated()
        tic = time.perf_counter()

        if reindex:
            self.build_nodes()
            if retriever_type == "dense":
                retriever = self.build_dense_retriever(topk)
            elif retriever_type == "bm25":
                retriever = self.build_bm25_retriever(topk)
            
            query_engine = self.build_query_engine(retriever, reranker=reranker, prompt_edit=prompt_edit)
        else:
            query_engine = self.query_engines[f"{retriever_type}_{topk}_{reranker}_{prompt_edit}"]
        
        claim_question = self.to_question(query_str)
        response_object = query_engine.query(claim_question)
        response = response_object.response.strip()
        if response == "Empty Response":
            # In case no node found by the retriever:
            # https://github.com/run-llama/llama_index/blob/fe72a2f5dbefb92d8c91cb460d4299de5637aa5a/llama-index-core/llama_index/core/response_synthesizers/base.py#L284

            response = {"prediction": "NOTENOUGHINFO"}
        else:
            response = json.loads(repair_json(response))

        latency = time.perf_counter() - tic # seconds
        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated()
        vram_usage = (peak_after - start_mem) / 1048576 # MB, 1024 * 1024

        response["raw_response"] = response_object
        response["question"] = claim_question
        response["claim"] = query_str
        response["retrieved_context"] = [{"text": n.node.text, 'doc_id': n.node.metadata['doc_id'], 'node_id': n.id_, 'score': n.score} for n in response_object.source_nodes]

        if self.knowledge_graph and consistency_check and "response" in response:
            response["consistency_check"] = self.knowledge_graph.consistency_check(response["response"])

        if entailment_check and "response" in response:
            response["entailment_check"] = self.entailment_checker.check(query_str, response["response"], response["retrieved_context"])

        response["latency"] = latency
        response["vram_usage"] = vram_usage

        if add_entity_triplets and "response" in response:
            response["entity_relations"] = self.triplet_extractor.extract_triplets(response["response"])

        return response