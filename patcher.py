from bandits import LinUCB, ThompsonSampling
import random
import numpy as np
import cloudpickle
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core import Settings
from sentence_transformers import SentenceTransformer
import torch


class BanditPatcher:

    def __init__(self, latency_budget=None, vram_budget=None, method="linucb", alpha=1.):
        self.reward_binary_threshold = 0.6
        self.train_exploration_rate = 0.5

        self.latency_budget = latency_budget
        self.vram_budget = vram_budget
        self.method = method
        assert method in ["linucb", "thompsonsampling"]
        self.init_cfg()

        # idx = 0
        # self.possible_actions = []
        # for retriever_type in ["dense", "bm25"]:
        #     for topk in [10, 5, 20]:
        #         for reranker in [False, True]:
        #             for prompt_edit in [False, "WP", "WR"]:
        #                 for reindex in [False, True]:
        #                     self.possible_actions.append((idx, {"retriever": retriever_type, "topk": topk, "reranker": reranker, "prompt_edit": prompt_edit, "reindex": reindex}))
        #                     idx += 1

        idx = 0
        self.possible_actions = []
        for retriever_type in ["dense", "bm25"]:
            for topk in [5, 20]:
                for reindex in [False, True]:
                    self.possible_actions.append((idx, {"retriever": retriever_type, "topk": topk, "reindex": reindex}))
                    idx += 1

        for reranker in [True]:
            self.possible_actions.append((idx, {"reranker": reranker}))
            idx += 1

        for prompt_edit in ["WP", "WR"]:
            self.possible_actions.append((idx, {"prompt_edit": prompt_edit}))
            idx += 1

        if method == "linucb":
            self.bandit = LinUCB(len(self.possible_actions), self.context_dim, alpha=alpha)
        elif method == "thompsonsampling":
            self.bandit = ThompsonSampling(len(self.possible_actions))

    def get_failure_label(self, rag_response):
        response = rag_response.get("response", "None")
        rag_prediction = rag_response.get("prediction", "NOTENOUGHINFO")
        kg_result = rag_response.get("consistency_check", "MISSING")
        nli_response_result = rag_response.get("entailment_check", {}).get("response", "NEUTRAL")
        nli_claim_result = rag_response.get("entailment_check", {}).get("claim", "NEUTRAL")
        retrieved_context = rag_response.get("retrieved_context", [None])

        if kg_result == "CONFLICT":
            failure_label = "WRONG_PREDICATE_FAILURE" # generation problem
        elif len(retrieved_context) == 0 or len(response) == 0:
            failure_label = "RETRIEVER_FAILURE" # retriever problem
        elif nli_claim_result == "NEUTRAL":
            failure_label = "NOTENOUGHINFO_FAILURE" # retriever problem
        elif nli_response_result in ["CONTRADICTION", "NEUTRAL"]:
            failure_label = "WRONG_RESPONSE_FAILURE" # generation problem
        elif nli_response_result in "ENTAILMENT":
            if rag_prediction == "SUPPORTS" and nli_claim_result == "ENTAILMENT":
                failure_label = "NO_FAILURE"
            elif rag_prediction == "REFUTES" and nli_claim_result == "CONTRADICTION":
                failure_label = "NO_FAILURE"
            elif rag_prediction == "NOTENOUGHINFO":
                failure_label = "NO_FAILURE"
            else:
                failure_label = "LABEL_RESPONSE_MISMATCH_FAILURE" # generation problem
        
        return failure_label

    def get_context(self, claim, context_len, failure_label, consistency_check, claim_entailment_check, response_entailment_check, rag_response):
        claim_embedding = self.embedding_model.encode(claim)
        claim_embedding = torch.avg_pool1d(torch.from_numpy(claim_embedding[None, ...]), 8)[0].numpy().tolist()
        
        failure_onehot = self.failure_map[failure_label] 
        consistency_vector = self.consistency_map[consistency_check] 
        claim_entailment_vector = self.entailment_map[claim_entailment_check] 
        response_entailment_vector = self.entailment_map[response_entailment_check] 

        metrics_vector = [
            self.evaluator_faithfulness.evaluate_response(response=rag_response).score,
        ]

        context_vector = np.array([context_len] + claim_embedding + failure_onehot + consistency_vector + claim_entailment_vector + response_entailment_vector + metrics_vector)

        return context_vector

    def calculate_reward(self, failure_label, rag_response, rag_label, gt_label, consistency_check, entailment_check, action_latency, action_vram_usage):
        if failure_label == "NO_FAILURE":
            failure_reward = 1.
        else:
            failure_reward = 0.

        # faithfulness_reward = self.evaluator_faithfulness.evaluate_response(response=rag_response).score

        if rag_label == gt_label:
            prediction_reward = 1.
        else:
            prediction_reward = 0.

        if consistency_check == "CONSISTENT":
            consistency_reward = 1.
        elif consistency_check == "CONFLICT":
            consistency_reward = 0.
        else:
            consistency_reward = 0.

        if entailment_check == "ENTAILMENT":
            factuality_reward = 1.
        else:
            factuality_reward = 0.

        latency_budget_strict = (0. if action_latency > self.latency_budget else 1.) if self.latency_budget is not None else 1.
        vram_budget_strict = (0. if action_vram_usage > self.vram_budget else 1.) if self.vram_budget is not None else 1.

        latency = (action_latency / self.latency_budget) if self.latency_budget is not None else 0.
        vram = (action_vram_usage / self.vram_budget) if self.vram_budget is not None else 0.

        # reward = (
        #     1.25 * failure_reward + \
        #     # 1. * faithfulness_reward + \
        #     1.25 * prediction_reward + \
        #     1.25 * consistency_reward + \
        #     1.25 * factuality_reward + \
        #     2.5 * latency_budget_reward + \
        #     2.5 * vram_budget_reward) / 10
        reward = (
            1. * failure_reward + \
            # 1. * faithfulness_reward + \
            2. * prediction_reward + \
            1. * consistency_reward + \
            2. * factuality_reward) / 6
        reward *= latency_budget_strict * vram_budget_strict
        reward *= (1 - latency) * (1 - vram)
        # reward -= 0.02 * num_retries
        # reward = float(max(-1.0, min(1.0, reward))) # [-1, 1]

        # reward = 0. if reward < self.reward_binary_threshold else 1. # {0, 1}

        if self.method == "linucb":
            # reward = (reward + 1) / 2 # [0, 1]
            # reward = 0. if reward < 0. else reward
            reward = {
                "total_reward": reward, 
                "failure_reward": failure_reward, 
                # "faithfulness_reward": faithfulness_reward,
                "prediction_reward": prediction_reward,
                "consistency_reward": consistency_reward,
                "factuality_reward": factuality_reward,
                "latency_budget_strict": latency_budget_strict,
                "vram_budget_strict": vram_budget_strict,
                "latency_rate": latency,
                "vram_rate": vram,
            }
            return reward
        elif self.method == "thompsonsampling":
            # reward = 0. if reward <= 0.5 else 1. # {0, 1}
            reward = 0. if reward < self.reward_binary_threshold else 1. # {0, 1}
            reward = {
                    "total_reward": reward, 
                    "failure_reward": failure_reward, 
                    # "faithfulness_reward": faithfulness_reward,
                    "prediction_reward": prediction_reward,
                    "consistency_reward": consistency_reward,
                    "factuality_reward": factuality_reward,
                    "latency_budget_strict": latency_budget_strict,
                    "vram_budget_strict": vram_budget_strict,
                    "latency_rate": latency,
                    "vram_rate": vram,
                }
            return reward

    def predict(self, context, failure_label=None, explore=False):
        prob = random.random()
        if explore and prob <= self.train_exploration_rate:
            predicted_action = random.choice(self.possible_actions)            
        else:
            predicted_action = self.possible_actions[self.bandit.select_arm(context)]

        return predicted_action
        
    def update_bandit(self, context, action, reward):
        self.bandit.update(action, context, reward)

    def save_bandit(self):
        cloudpickle.dump(self.bandit, open(f"out/{self.method}_patcher_{self.latency_budget}_{self.vram_budget}.pkl", "wb"))

    def load_bandit(self):
        self.bandit = cloudpickle.load(open(f"out/{self.method}_patcher_{self.latency_budget}_{self.vram_budget}.pkl", "rb"))

    def init_cfg(self):
        self.failure_map = {
            "NO_FAILURE": [1, 0, 0, 0, 0],
            "WRONG_PREDICATE_FAILURE": [0, 1, 0, 0, 0],
            "NOTENOUGHINFO_FAILURE": [0, 0, 1, 0, 0],
            "RETRIEVER_FAILURE": [0, 0, 0, 1, 0],
            "WRONG_RESPONSE_FAILURE": [0, 0, 0, 0, 1],
        }
        self.consistency_map = {
            "CONSISTENT": [1, 0, 0, 0, 0],
            "CONFLICT": [0, 1, 0, 0, 0],
            "MISSING": [0, 0, 1, 0, 0],
            "EMPTYINPUT": [0, 0, 0, 1, 0],
            "NOTRIPLETS": [0, 0, 0, 0, 1],
        }
        self.entailment_map = {
            "ENTAILMENT": [1, 0, 0],
            "CONTRADICTION": [0, 1, 0],
            "NEUTRAL": [0, 0, 1],
        }

        self.context_dim = len(self.failure_map["NO_FAILURE"]) + len(self.consistency_map["CONSISTENT"]) + 2 * len(self.entailment_map["ENTAILMENT"]) + 2 + 48
        
        self.evaluator_faithfulness = FaithfulnessEvaluator(llm=Settings.llm)
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


class BanditPatcherGR(BanditPatcher):

    def __init__(self, latency_budget=None, vram_budget=None, method="linucb", alpha=(1., 1.)):
        self.reward_binary_threshold = 0.6
        self.train_exploration_rate = 0.5
        
        self.latency_budget = latency_budget
        self.vram_budget = vram_budget
        self.method = method
        assert method in ["linucb", "thompsonsampling"]
        self.init_cfg()

        idx = 0
        self.possible_actions_retrieval = []
        for retriever_type in ["dense", "bm25"]:
            for topk in [5, 20]:
                for reindex in [False, True]:
                    self.possible_actions_retrieval.append((idx, {"retriever": retriever_type, "topk": topk, "reindex": reindex}))
                    idx += 1

        idx = 0
        self.possible_actions_generation = []
        # for reranker in [False, True]:
        #     for prompt_edit in [False, "WP", "WR"]:
        #         self.possible_actions_generation.append((idx, {"reranker": reranker, "prompt_edit": prompt_edit}))
        #         idx += 1
        for reranker in [True]:
            self.possible_actions_generation.append((idx, {"reranker": reranker}))
            idx += 1

        for prompt_edit in ["WP", "WR"]:
            self.possible_actions_generation.append((idx, {"prompt_edit": prompt_edit}))
            idx += 1

        if method == "linucb":
            self.bandit_retrieval = LinUCB(len(self.possible_actions_retrieval), self.context_dim, alpha=alpha[0])
            self.bandit_generation = LinUCB(len(self.possible_actions_generation), self.context_dim, alpha=alpha[1])
        elif method == "thompsonsampling":
            self.bandit_retrieval = ThompsonSampling(len(self.possible_actions_retrieval))
            self.bandit_generation = ThompsonSampling(len(self.possible_actions_generation))

    def detect_component(self, failure_label):
        if failure_label in ["WRONG_PREDICATE_FAILURE", "WRONG_RESPONSE_FAILURE", "LABEL_RESPONSE_MISMATCH_FAILURE"]:
            return "generation"
        elif failure_label in ["NOTENOUGHINFO_FAILURE", "RETRIEVER_FAILURE"]:
            return "retrieval"
        
        return None

    def predict(self, context, failure_label=None, explore=False):
        self.latest_component = self.detect_component(failure_label)

        prob = random.random()
        if explore and prob <= self.train_exploration_rate:
            if self.latest_component == "generation":
                predicted_action = random.choice(self.possible_actions_generation)
            elif self.latest_component == "retrieval":
                predicted_action = random.choice(self.possible_actions_retrieval)
            else:
                if random.random() > 0.5:
                    self.latest_component = "generation"
                    predicted_action = random.choice(self.possible_actions_generation)
                else:
                    self.latest_component = "retrieval"
                    predicted_action = random.choice(self.possible_actions_retrieval)
        else:
            if self.latest_component == "generation":
                predicted_action = self.possible_actions_generation[self.bandit_generation.select_arm(context)]
            elif self.latest_component == "retrieval":
                predicted_action = self.possible_actions_retrieval[self.bandit_retrieval.select_arm(context)]
            else:
                predicted_action = (-1, {})

        return predicted_action
    
    def update_bandit(self, context, action, reward):
        if action != -1 and self.latest_component == "generation":
            self.bandit_generation.update(action, context, reward)
        elif action != -1 and self.latest_component == "retrieval":
            self.bandit_retrieval.update(action, context, reward)
        
    def save_bandit(self):
        cloudpickle.dump(self.bandit_retrieval, open(f"out/{self.method}_retrieval_patcher_{self.latency_budget}_{self.vram_budget}.pkl", "wb"))
        cloudpickle.dump(self.bandit_generation, open(f"out/{self.method}_generation_patcher_{self.latency_budget}_{self.vram_budget}.pkl", "wb"))

    def load_bandit(self):
        self.bandit_retrieval = cloudpickle.load(open(f"out/{self.method}_retrieval_patcher_{self.latency_budget}_{self.vram_budget}.pkl", "rb"))
        self.bandit_generation = cloudpickle.load(open(f"out/{self.method}_generation_patcher_{self.latency_budget}_{self.vram_budget}.pkl", "rb"))