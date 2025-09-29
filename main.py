from create_knowledge_base import WikipagesKnowledgeBase
from knowledge_graph import WikiMoviesKnowledgeGraph
from utils import setup_settings, load_fever
from rag_engine import RAGEngine
import tqdm


if __name__ == "__main__":
    # WikipagesKnowledgeBase().build("./shared_task_dev.jsonl", "./wiki-pages", 5)
    
    setup = setup_settings()

    kgraph = WikiMoviesKnowledgeGraph(**setup)
    kgraph.build("./movieqa")
    kgraph.plot()

    rag = RAGEngine("./wikipages_knowledge_base.pkl", **setup)

    print("Prediction:", rag.query("Telemundo is a English-language television network."))
    print("Prediction:", rag.query("Damon Albarn's debut album was released in 2011."))
    print("Prediction:", rag.query("There is a capital called Mogadishu."))
    print("Prediction:", rag.query("Happiness in Slavery is a gospel song by Nine Inch Nails."))
    print("Prediction:", rag.query("Soroush Hashemifar is an artificial intelligence."))

    print("Prediction:", rag.query("Telemundo is a English-language television network.", add_entity_triplets=True))
    print("Prediction:", rag.query("Damon Albarn's debut album was released in 2011.", add_entity_triplets=True))
    print("Prediction:", rag.query("There is a capital called Mogadishu.", add_entity_triplets=True))
    print("Prediction:", rag.query("Happiness in Slavery is a gospel song by Nine Inch Nails.", add_entity_triplets=True))
    print("Prediction:", rag.query("Soroush Hashemifar is an artificial intelligence.", add_entity_triplets=True))

    ### EXTRACT TRIPLETS FROM RAG RESPONSE

    ### CHECK KG CONSISTENCY



    # fever_dataset = load_fever("./shared_task_dev.jsonl")

    # results = []
    # for claim, label in tqdm.tqdm(fever_dataset):
    #     pred = rag.query(claim)
    #     if pred.lower() in ["supports", "refutes", "notenoughinfo"]:
    #         results.append(pred == label)

    #     if len(results) == 100:
    #         break

    # accuracy = sum(results) / len(results)   
    # print(accuracy)