from create_knowledge_base import WikipagesKnowledgeBase
from knowledge_graph import WikiMoviesKnowledgeGraph
from utils import setup_settings, load_fever
from rag_engine import RAGEngine
import tqdm


if __name__ == "__main__":
    # WikipagesKnowledgeBase().build("./shared_task_dev.jsonl", "./wiki-pages", 1)

    setup = setup_settings()

    kgraph = WikiMoviesKnowledgeGraph(**setup)
    kgraph.build("./movieqa", "./wikipages_knowledge_base.pkl", 100)
    kgraph.plot()
    # kgraph = None

    prompt = "Mitchell Altieri directed The Violent Kind."
    print(kgraph.consistency_check(prompt))

    prompt = "Mitchell Altieri wrote The Violent Kind."
    print(kgraph.consistency_check(prompt))

    prompt = "Soroush Hashemifar directed The Violent Kind."
    print(kgraph.consistency_check(prompt))

    prompt = "Mitchell Altieri directed Golabiha."
    print(kgraph.consistency_check(prompt))

    prompt = "Soroush Hashemifar directed Golabiha."
    print(kgraph.consistency_check(prompt))

    # prompt = "Return ONLY 'True' if the following statement is true, otherwise just 'False': Mitchell Gholami directed The Violent Kind."
    # print(kgraph.query_engine.query(prompt).response)

    rag = RAGEngine("./wikipages_knowledge_base.pkl", knowledge_graph=kgraph, **setup)

    # print("Prediction:", rag.query("Telemundo is a English-language television network."))
    # print("Prediction:", rag.query("Damon Albarn's debut album was released in 2011."))
    # print("Prediction:", rag.query("There is a capital called Mogadishu."))
    # print("Prediction:", rag.query("Happiness in Slavery is a gospel song by Nine Inch Nails."))
    # print("Prediction:", rag.query("Soroush Hashemifar is an artificial intelligence."))

    # # # print("Prediction:", rag.query("Telemundo is a English-language television network.", add_entity_triplets=True))
    # # # print("Prediction:", rag.query("Damon Albarn's debut album was released in 2011.", add_entity_triplets=True))
    # # # print("Prediction:", rag.query("There is a capital called Mogadishu.", add_entity_triplets=True))
    # # # print("Prediction:", rag.query("Happiness in Slavery is a gospel song by Nine Inch Nails.", add_entity_triplets=True))
    # # # print("Prediction:", rag.query("Soroush Hashemifar is an artificial intelligence.", add_entity_triplets=True))

    # print("Prediction:", rag.query("Telemundo is a English-language television network.", consistency_check=True))
    # print("Prediction:", rag.query("Damon Albarn's debut album was released in 2011.", consistency_check=True))
    # print("Prediction:", rag.query("There is a capital called Mogadishu.", consistency_check=True))
    # print("Prediction:", rag.query("Happiness in Slavery is a gospel song by Nine Inch Nails.", consistency_check=True))
    # print("Prediction:", rag.query("Soroush Hashemifar is an artificial intelligence.", consistency_check=True))
    # print("Prediction:", rag.query("Mitchell Altieri and Phil Flores directed The Violent Kind.", consistency_check=True))

    print("Prediction:", rag.query("Telemundo is a English-language television network.", consistency_check=True, entailment_check=True))
    print("Prediction:", rag.query("Damon Albarn's debut album was released in 2011.", consistency_check=True, entailment_check=True))
    print("Prediction:", rag.query("There is a capital called Mogadishu.", consistency_check=True, entailment_check=True))
    print("Prediction:", rag.query("Happiness in Slavery is a gospel song by Nine Inch Nails.", consistency_check=True))
    print("Prediction:", rag.query("Soroush Hashemifar is an artificial intelligence.", consistency_check=True, entailment_check=True))
    print("Prediction:", rag.query("Mitchell Altieri and Phil Flores directed The Violent Kind.", consistency_check=True, entailment_check=True))



    # fever_dataset = load_fever("./shared_task_dev.jsonl")

    # results = []
    # for claim, label in tqdm.tqdm(fever_dataset):
    #     pred = rag.query(claim, consistency_check=True)
    #     print(pred)
        
    #     if pred.lower() in ["supports", "refutes", "notenoughinfo"]:
    #         results.append(pred == label)

    #     if len(results) == 100:
    #         break

    # accuracy = sum(results) / len(results)   
    # print(accuracy)