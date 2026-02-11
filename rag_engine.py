import torch
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate
import pickle
import faiss
import json
from triplet_extractor import TripletExtractor
import os
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

    def __init__(self, output_path, knowledgebase, knowledge_graph=None, **kwargs):
        self.output_path = output_path
        self.kwargs = kwargs
        self.knowledge_graph = knowledge_graph
        self.knowledgebase = knowledgebase
        self.knowledgebase.reset()

        self.build_nodes()

        self.first_time_dense_index = True
        self.first_time_bm25_index = True

        self.query_engines = {}
        for retriever_type in ["dense", "bm25"]:
            for topk in [10, 5, 20]:
                for reranker in [False, True]:
                    for prompt_edit in ["simple_qa", "paraphrase_qa", "simplify_qa"]:
                        if retriever_type == "dense":
                            retriever = self.build_dense_retriever(topk)
                        elif retriever_type == "bm25":
                            retriever = self.build_bm25_retriever(topk)
                        
                        self.query_engines[f"{retriever_type}_{topk}_{reranker}_{prompt_edit}"] = self.build_query_engine(retriever, reranker=reranker, prompt_edit=prompt_edit)

        self.triplet_extractor = TripletExtractor(**kwargs)
        self.entailment_checker = EntailmentChecker(**kwargs)

    def build_nodes(self):
        """
            source: https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
        """

        print("[MSG] Building nodes from documents...")
        node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=16)

        documents = [Document(text=text, extra_info={}) for text in self.knowledgebase]
        base_nodes = node_parser.get_nodes_from_documents(documents)
        self.all_nodes = base_nodes

        print("nodes size:", len(self.all_nodes))
        self.all_nodes_dict = {n.node_id: n for n in self.all_nodes}

    def build_dense_retriever(self, similarity_top_k, reindex=False):
        """
            source: https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/
            source: https://github.com/run-llama/llama_index/issues/6977
            source: https://github.com/run-llama/llama_index/issues/10631
            source: https://app.readytensor.ai/publications/retrieval-augmented-generation-using-llamaindex-faiss-and-openai-gpt-4-SfLlZniaZJ9C
        """

        if self.first_time_dense_index:
            reindex = True
            self.first_time_dense_index = False

        if os.path.exists(os.path.join(self.output_path, self.kwargs["dense_retriever_storage"], 'index_store.json')) and not reindex:
            print(f"[MSG] Loading Dense (k={similarity_top_k}) retriever...")
            
            vector_store = FaissVectorStore.from_persist_dir(os.path.join(self.output_path, self.kwargs["dense_retriever_storage"]))
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=os.path.join(self.output_path, self.kwargs["dense_retriever_storage"]))
            index = load_index_from_storage(storage_context)
        else:
            print(f"[MSG] Building Dense (k={similarity_top_k}) retriever...")

            faiss_index = faiss.IndexFlatL2(384)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            index = VectorStoreIndex(self.all_nodes, storage_context=storage_context, show_progress=False)
            index.storage_context.persist(persist_dir=os.path.join(self.output_path, self.kwargs["dense_retriever_storage"]))

        retriever = VectorIndexRetriever(
            index=index, 
            similarity_top_k=similarity_top_k
        )

        return retriever

    def build_bm25_retriever(self, similarity_top_k, reindex=False):
        """
            reference: https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/
        """

        if self.first_time_bm25_index:
            reindex = True
            self.first_time_bm25_index = False
        
        if os.path.exists(os.path.join(self.output_path, self.kwargs["bm25_retriever_storage"])) and not reindex:
            print(f"[MSG] Loading BM25 (k={similarity_top_k}) retriever...")
            
            retriever = BM25Retriever.from_persist_dir(os.path.join(self.output_path, self.kwargs["bm25_retriever_storage"]))
        else:
            print(f"[MSG] Building BM25 (k={similarity_top_k}) retriever...")

            retriever = BM25Retriever.from_defaults(
                nodes=self.all_nodes,
                similarity_top_k=similarity_top_k,
                stemmer=Stemmer.Stemmer("english"),
                language="english",
            )
            retriever.persist(os.path.join(self.output_path, self.kwargs["bm25_retriever_storage"]))

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

        prompt = self.kwargs["prompts"][prompt_edit]
        query_engine = RetrieverQueryEngine.from_args(
            retriever, 
            response_mode="compact_accumulate",
            text_qa_template=PromptTemplate(prompt),
            node_postprocessors=node_postprocessors
        )

        print(f"[MSG] RAG query engine is ready to go. (reranker={reranker}, prompt_edit={prompt_edit})")

        return query_engine
    
    def query(self, query_str, params={}, consistency_check=False, entailment_check=False):
        retriever_type = params.get("retriever", "bm25")
        topk = params.get("topk", 5)
        reranker = params.get("reranker", False)
        prompt_edit = params.get("prompt_edit", False)
        reindex = params.get("reindex", False)
        response_obj = {}

        if self.kwargs["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
        else:
            start_mem = 0.
        tic = time.perf_counter()

        if reindex:
            self.knowledgebase.evolve()
            self.build_nodes()
            if retriever_type == "dense":
                retriever = self.build_dense_retriever(topk)
            elif retriever_type == "bm25":
                retriever = self.build_bm25_retriever(topk)
            
            query_engine = self.build_query_engine(retriever, reranker=reranker, prompt_edit=prompt_edit)
        else:
            query_engine = self.query_engines[f"{retriever_type}_{topk}_{reranker}_{prompt_edit}"]
        
        response_object = query_engine.query(query_str)
        response = response_object.response.strip()
        if response == "Empty Response":
            # In case no node found by the retriever:
            # https://github.com/run-llama/llama_index/blob/fe72a2f5dbefb92d8c91cb460d4299de5637aa5a/llama-index-core/llama_index/core/response_synthesizers/base.py#L284

            response = {"label": "NOTENOUGHINFO", "response": "Context is not sufficient."}
        else:
            response = json.loads(repair_json(response))

        response_obj.update(response)

        latency = time.perf_counter() - tic # seconds
        if self.kwargs["device"] == "cuda":
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated()
            vram_usage = (peak_after - start_mem) / 1048576 # MB, 1024 * 1024
        else:
            vram_usage = 0.

        response_obj["query"] = query_str
        response_obj["retrieved_context"] = [{"text": n.node.text, 'node_id': n.id_, 'score': n.score} for n in response_object.source_nodes]

        if self.knowledge_graph and consistency_check:
            response_obj["consistency_check"] = self.knowledge_graph.consistency_check(response_obj["response"])

        if entailment_check:
            response_obj["entailment_check"] = self.entailment_checker.check(query_str, response_obj["response"], response_obj["retrieved_context"])

        response_obj["latency"] = latency
        response_obj["vram_usage"] = vram_usage

        return response_obj
    
    def query_shortanswer(self, query_str, params={}, consistency_check=False, entailment_check=False):
        retriever_type = params.get("retriever", "bm25")
        topk = params.get("topk", 5)
        reranker = params.get("reranker", False)
        prompt_edit = params.get("prompt_edit", False)
        reindex = params.get("reindex", False)
        response_obj = {}

        if self.kwargs["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_mem = torch.cuda.memory_allocated()
        else:
            start_mem = 0.
        tic = time.perf_counter()

        if reindex:
            self.knowledgebase.evolve()
            self.build_nodes()
            if retriever_type == "dense":
                retriever = self.build_dense_retriever(topk)
            elif retriever_type == "bm25":
                retriever = self.build_bm25_retriever(topk)
            
            query_engine = self.build_query_engine(retriever, reranker=reranker, prompt_edit=prompt_edit)
        else:
            query_engine = self.query_engines[f"{retriever_type}_{topk}_{reranker}_{prompt_edit}"]
        
        response_object = query_engine.query(query_str)
        response = response_object.response.strip()
        if response == "Empty Response":
            # In case no node found by the retriever:
            # https://github.com/run-llama/llama_index/blob/fe72a2f5dbefb92d8c91cb460d4299de5637aa5a/llama-index-core/llama_index/core/response_synthesizers/base.py#L284

            response = {"response": "The provided context is insufficient."}
        else:
            response = {"response": response.replace("Response 1: ", "")}

        response_obj.update(response)

        latency = time.perf_counter() - tic # seconds
        if self.kwargs["device"] == "cuda":
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated()
            vram_usage = (peak_after - start_mem) / 1048576 # MB, 1024 * 1024
        else:
            vram_usage = 0.

        response_obj["query"] = query_str
        response_obj["retrieved_context"] = [{"text": n.node.text, 'node_id': n.id_, 'score': n.score} for n in response_object.source_nodes]

        if self.knowledge_graph and consistency_check:
            response_obj["consistency_check"] = self.knowledge_graph.consistency_check(response_obj["response"])

        if entailment_check:
            response_obj["entailment_check"] = self.entailment_checker.check(query_str, response_obj["response"], response_obj["retrieved_context"])

        response_obj["latency"] = latency
        response_obj["vram_usage"] = vram_usage

        return response_obj
    
