import os 
# prevent JAX from using gpu to avoid bm25s eat up the gpu memory
os.environ['JAX_PLATFORMS'] = 'cpu'

from knowledge_graph import WikiMoviesKnowledgeGraph
from utils import setup_settings, load_fever
from rag_engine import RAGEngine
import tqdm


if __name__ == "__main__":
    setup = setup_settings()

    kgraph = WikiMoviesKnowledgeGraph(**setup)
    # kgraph.manual_check_triplets("./movieqa", "./wikipages_knowledge_base.pkl", 1000)
    kgraph.build("./movieqa", "./wikipages_knowledge_base.pkl", 100)
    # kgraph.plot()
    # kgraph = None

    # prompt = "Mitchell Altieri directed The Violent Kind."
    # print(kgraph.consistency_check(prompt))
    # print("######")

    # prompt = "Mitchell Altieri wrote The Violent Kind."
    # print(kgraph.consistency_check(prompt))
    # print("######")

    # prompt = "Soroush Hashemifar directed The Violent Kind."
    # print(kgraph.consistency_check(prompt))
    # print("######")

    # prompt = "Mitchell Altieri directed Golabiha."
    # print(kgraph.consistency_check(prompt))
    # print("######")

    # prompt = "Soroush Hashemifar directed Golabiha."
    # print(kgraph.consistency_check(prompt))
    # print("######")

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

    # print("Prediction:", rag.query("Whole population of Lithuania identify themselves as Lithuanians.", consistency_check=True))
    # print("Prediction:", rag.query("Half of the population of Lithuania identify themselves as Lithuanians.", consistency_check=True))
    # print("Prediction:", rag.query("People in Lithuania identify themselves as Americans.", consistency_check=True))
    # print("Prediction:", rag.query("People in Armenia identify themselves as Lithuanians.", consistency_check=True))
    # print("Prediction:", rag.query("People in Armenia identify themselves as European.", consistency_check=True))

    # print("Prediction:", rag.query("Whole population of Lithuania identify themselves as Lithuanians.", entailment_check=True))
    # print("Prediction:", rag.query("Half of the population of Lithuania identify themselves as Lithuanians.", entailment_check=True))
    # print("Prediction:", rag.query("People in Lithuania identify themselves as Americans.", entailment_check=True))
    # print("Prediction:", rag.query("People in Armenia identify themselves as Lithuanians.", entailment_check=True))
    # print("Prediction:", rag.query("People in Armenia identify themselves as European.", entailment_check=True))

    # print("Prediction:", rag.query("Whole population of Lithuania identify themselves as Lithuanians.", consistency_check=True, entailment_check=True))
    # print("Prediction:", rag.query("Half of the population of Lithuania identify themselves as Lithuanians.", consistency_check=True, entailment_check=True))
    # print("Prediction:", rag.query("People in Lithuania identify themselves as Americans.", consistency_check=True, entailment_check=True))
    # print("Prediction:", rag.query("People in Armenia identify themselves as Lithuanians.", consistency_check=True, entailment_check=True))
    # print("Prediction:", rag.query("People in Armenia identify themselves as European.", consistency_check=True, entailment_check=True))

    print("Prediction:", rag.query("Whole population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("Half of the population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Lithuania identify themselves as Americans."))
    print("Prediction:", rag.query("People in Armenia identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Armenia identify themselves as European."))
    
    rag.build_dense_retriever(5, reindex=True)
    print("Prediction:", rag.query("Whole population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("Half of the population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Lithuania identify themselves as Americans."))
    print("Prediction:", rag.query("People in Armenia identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Armenia identify themselves as European."))

    rag.build_dense_retriever(10, reindex=True)
    print("Prediction:", rag.query("Whole population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("Half of the population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Lithuania identify themselves as Americans."))
    print("Prediction:", rag.query("People in Armenia identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Armenia identify themselves as European."))

    rag.build_query_engine(reranker=True)
    print("Prediction:", rag.query("Whole population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("Half of the population of Lithuania identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Lithuania identify themselves as Americans."))
    print("Prediction:", rag.query("People in Armenia identify themselves as Lithuanians."))
    print("Prediction:", rag.query("People in Armenia identify themselves as European."))
    exit()





    fever_dataset = load_fever("./shared_task_dev.jsonl")

    results = []
    for claim, label in tqdm.tqdm(fever_dataset):
        pred = rag.query(claim, consistency_check=True, entailment_check=True, failure_check=True)
        print(pred)
        
    #     if pred.lower() in ["supports", "refutes", "notenoughinfo"]:
    #         results.append(pred == label)

    #     if len(results) == 100:
    #         break

    # accuracy = sum(results) / len(results)   
    # print(accuracy)