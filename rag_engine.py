from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import IndexNode
import pickle
import faiss
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever, RecursiveRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate


class RAGEngine:

    def __init__(self, knowledge_base_pkl_path, **kwargs):
        self.kwargs = kwargs

        with open(knowledge_base_pkl_path, 'rb') as f:
            content = pickle.load(f)
            self.knowledge_base = content["knowledge_base"]
            self.meta_data = content["meta_data"]

        self.build_nodes()
        self.prepare_storage()
        self.build_query_engine()

    def build_nodes(self):
        """
            source: https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
        """

        documents = [Document(text=sentence, extra_info=mdata) for sentence, mdata in zip(self.knowledge_base, self.meta_data)]

        # build parent chunks via NodeParser
        node_parser = SentenceSplitter(chunk_size=1024)
        base_nodes = node_parser.get_nodes_from_documents(documents)

        # define smaller child chunks
        sub_chunk_sizes = [128, 256, 512]
        sub_node_parsers = [
            SentenceSplitter(chunk_size=c, chunk_overlap=32) for c in sub_chunk_sizes
        ]
        self.all_nodes = []
        for base_node in base_nodes:
            for n in sub_node_parsers:
                sub_nodes = n.get_nodes_from_documents([base_node])
                sub_inodes = [
                    IndexNode.from_text_node(sn, base_node.node_id) for sn in sub_nodes
                ]
                self.all_nodes.extend(sub_inodes)

            # also add original node to node
            original_node = IndexNode.from_text_node(base_node, base_node.node_id)
            self.all_nodes.append(original_node)

    def prepare_storage(self):
        """
            source: https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/usage_documents/
            source: https://github.com/run-llama/llama_index/issues/6977
            source: https://github.com/run-llama/llama_index/issues/10631
            source: https://app.readytensor.ai/publications/retrieval-augmented-generation-using-llamaindex-faiss-and-openai-gpt-4-SfLlZniaZJ9C
        """

        faiss_index = faiss.IndexFlatL2(384)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        self.index = VectorStoreIndex(self.all_nodes, storage_context=storage_context)

    def build_query_engine(self):
        """
            source: https://www.llamaindex.ai/blog/evaluating-rag-with-deepeval-and-llamaindex
            source: https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b
        """

        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=self.kwargs["similarity_top_k"])

        # build RecursiveRetriever
        all_nodes_dict = {n.node_id: n for n in self.all_nodes}
        retriever_chunk = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": retriever},
            node_dict=all_nodes_dict,
            verbose=False,
        )

        self.query_engine = RetrieverQueryEngine.from_args(
            retriever_chunk,
            text_qa_template=PromptTemplate(self.kwargs["fact_prompt"]),
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=self.kwargs["similarity_cutoff"], 
                                        filter_empty=True,
                                        filter_duplicates=True,
                                        filter_similar=True,)
                ],
        )

    def query(self, query_str):
        return self.query_engine.query(query_str).response.strip()