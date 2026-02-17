import random
import numpy as np
import cloudpickle
import torch
from transformers import AutoTokenizer, AutoModel

from src.bandits import LinUCB, ThompsonSampling


class BanditPatcher:

    def __init__(self, output_path, latency_budget=None, vram_budget=None, method="linucb", alpha=1., train_exploration_rate=0.2, reward_binary_threshold=0.4, with_gating=True, with_cost=True):
        self.output_path = output_path

        self.reward_binary_threshold = reward_binary_threshold
        self.train_exploration_rate = train_exploration_rate

        self.latency_budget = latency_budget
        self.vram_budget = vram_budget
        self.method = method
        assert method in ["linucb", "thompsonsampling"]
        self.init_cfg()

        self.with_gating = with_gating 
        self.with_cost = with_cost

        idx = 0
        self.retriever_possible_actions = []
        self.generation_possible_actions = []
        for retriever_type in ["dense", "bm25"]:
            for topk in [10, 20]:
                for reindex in [False, True]:
                    self.retriever_possible_actions.append((idx, {"retriever": retriever_type, "topk": topk, "reindex": reindex}))
                    idx += 1

        self.action_weights = [0.25 / len(self.retriever_possible_actions) for _ in range(len(self.retriever_possible_actions))]

        for reranker in [True]:
            self.generation_possible_actions.append((idx, {"reranker": reranker}))
            idx += 1

        for prompt_edit in ["paraphrase_qa", "simplify_qa"]:
            self.generation_possible_actions.append((idx, {"prompt_edit": prompt_edit}))
            idx += 1

        self.action_weights += [0.25 for _ in range(len(self.generation_possible_actions))]
        self.possible_actions = self.retriever_possible_actions + self.generation_possible_actions

        if method == "linucb":
            self.bandit = LinUCB(len(self.possible_actions), self.context_dim, alpha=alpha)
        elif method == "thompsonsampling":
            self.bandit = ThompsonSampling(len(self.possible_actions))

    def get_failure_label(self, rag_response):
        kg_result = rag_response.get("consistency_check", "MISSING")
        nli_response_result = rag_response.get("entailment_check", {}).get("response", "ENTAILMENT")
        nli_query_result = rag_response.get("entailment_check", {}).get("query", "ENTAILMENT")
        rag_label = rag_response.get("label", "SUPPORTS" if nli_query_result == "ENTAILMENT" else "REFUTES")

        if kg_result == "CONFLICT":
            failure_label = "WRONG_PREDICATE_FAILURE" # generation problem
        elif nli_query_result == "NEUTRAL":
            failure_label = "INSUFFICIENT_EVIDENCE_FAILURE" # retriever problem
            rag_label = "UNVERIFIED"
        elif nli_response_result in ["CONTRADICTION", "NEUTRAL"]:
            failure_label = "WRONG_RESPONSE_FAILURE" # generation problem
        elif rag_label == "SUPPORTS" and nli_query_result == "CONTRADICTION":
            failure_label = "LABEL_EVIDENCE_MISMATCH_FAILURE" # retriever problem
            rag_label = "UNVERIFIED"
        elif rag_label == "REFUTES" and nli_query_result == "ENTAILMENT":
            failure_label = "LABEL_EVIDENCE_MISMATCH_FAILURE" # retriever problem
            rag_label = "UNVERIFIED"
        else:
            failure_label = "NO_FAILURE"            

        return failure_label, rag_label

    def get_context(self, query, failure_label, consistency_check, query_entailment_check, response_entailment_check, action_latency):
        inputs = self.embedding_tokenizer(query, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy().tolist()

        failure_onehot = self.failure_map[failure_label] 
        consistency_vector = self.consistency_map[consistency_check] 
        query_entailment_vector = self.entailment_map[query_entailment_check] 
        response_entailment_vector = self.entailment_map[response_entailment_check] 

        if self.device == "cuda":
            vram_available, vram_total = torch.cuda.mem_get_info()
        else:
            vram_available, vram_total = 0., 0.
        budget_vector = [
            max(0., (self.vram_budget - (vram_total - vram_available) / 1048576) / 1e5),
            max(0., (self.latency_budget - action_latency) / 1e4)
        ]

        context_vector = np.array(query_embedding + failure_onehot + consistency_vector + query_entailment_vector + response_entailment_vector + budget_vector)

        return context_vector

    def calculate_reward(self, failure_label, consistency_check, entailment_check, action_latency, action_vram_usage):
        if failure_label == "NO_FAILURE":
            failure_reward = 1.
        else:
            failure_reward = 0.

        if consistency_check == "CONSISTENT":
            consistency_reward = 1.
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

        reward = (
            1. * failure_reward + \
            1. * consistency_reward + \
            2. * factuality_reward) / 4
        if self.with_gating:
            reward *= latency_budget_strict * vram_budget_strict
        if self.with_cost:
            reward *= (1 - latency) * (1 - vram)
        
        if self.method == "linucb":
            reward = {
                "total_reward": reward, 
                "failure_reward": failure_reward, 
                "consistency_reward": consistency_reward,
                "factuality_reward": factuality_reward,
                "latency_budget_strict": latency_budget_strict,
                "vram_budget_strict": vram_budget_strict,
                "latency_rate": latency,
                "vram_rate": vram,
            }
            return reward
        elif self.method == "thompsonsampling":
            reward = 0. if reward < self.reward_binary_threshold else 1. # {0, 1}
            reward = {
                    "total_reward": reward, 
                    "failure_reward": failure_reward, 
                    "consistency_reward": consistency_reward,
                    "factuality_reward": factuality_reward,
                    "latency_budget_strict": latency_budget_strict,
                    "vram_budget_strict": vram_budget_strict,
                    "latency_rate": latency,
                    "vram_rate": vram,
                }
            return reward

    def predict(self, context, failure_label=None, explore=False, patchset="all"):
        if patchset == "retriever":
            armset = list(range(8))
        elif patchset == "generation":
            armset = list(range(8, len(self.possible_actions)))
        else:
            armset = list(range(len(self.possible_actions)))
        
        prob = random.random()
        if explore and prob <= self.train_exploration_rate:
            predicted_action = random.choices(self.possible_actions, k=1, weights=self.action_weights)[0]
        else:
            predicted_action = self.possible_actions[self.bandit.select_arm(context, armset)]

        return predicted_action
        
    def update_bandit(self, context, action, reward):
        self.bandit.update(action, context, reward)

    def save_bandit(self, postfix):
        cloudpickle.dump(self.bandit, open(f"{self.output_path}/{self.method}_patcher{postfix}.pkl", "wb"))

    def load_bandit(self, patcher_filepath, postfix):
        self.bandit = cloudpickle.load(open(f"{patcher_filepath}/{self.method}_patcher{postfix}.pkl", "rb"))

    def init_cfg(self):
        self.failure_map = {
            "NO_FAILURE": [1, 0, 0, 0, 0],
            "WRONG_PREDICATE_FAILURE": [0, 1, 0, 0, 0],
            "INSUFFICIENT_EVIDENCE_FAILURE": [0, 0, 1, 0, 0],
            "WRONG_RESPONSE_FAILURE": [0, 0, 0, 1, 0],
            "LABEL_EVIDENCE_MISMATCH_FAILURE": [0, 0, 0, 0, 1],
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
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embedding_tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
        self.embedding_model = AutoModel.from_pretrained("distilbert/distilbert-base-uncased").to(self.device)
        self.embedding_model.eval()
        hidden_size = 768

        self.context_dim = len(self.failure_map["NO_FAILURE"]) + len(self.consistency_map["CONSISTENT"]) + 2 * len(self.entailment_map["ENTAILMENT"]) + 2 + hidden_size