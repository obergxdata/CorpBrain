import json
import hashlib
from collections import defaultdict
import random

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.market.agent import Market
    from agents.corporations.agent import Corporation

from agents.mdp_logger import logger


class StateDisc:

    def __init__(self, market: "Market"):
        self.market = market

    def bucket_price(self, corp: "Corporation"):
        min_price, max_price = self.market.min_max_price
        if min_price == max_price:
            return "flat"

        range_size = (max_price - min_price) / 3
        low_threshold = min_price + range_size
        high_threshold = min_price + 2 * range_size

        if corp.price <= low_threshold:
            bucket = "low"
        elif corp.price <= high_threshold:
            bucket = "mid"
        else:
            bucket = "high"

        return bucket

    def bucket_0_1(self, value: float):
        low_threshold = 0.37
        high_threshold = 0.67

        if value <= low_threshold:
            bucket = "low"
        elif value <= high_threshold:
            bucket = "mid"
        else:
            bucket = "high"

        return bucket

    def bucket_mpc(self):
        return self.bucket_0_1(self.market.avg_mpc)

    def bucket_profit_trend(self, corp: "Corporation"):
        return self.bucket_0_1(corp.profit_trend(zero=True))

    def bucket_market_share(self, corp: "Corporation"):
        return self.bucket_0_1(self.market.market_share(corp))

    def get_state(self, corp: "Corporation"):
        state = {
            "price_bucket": self.bucket_price(corp),
            "mpc_bucket": self.bucket_mpc(),
            "profit_trend_bucket": self.bucket_profit_trend(corp),
            "market_share_bucket": self.bucket_market_share(corp),
        }

        return state

    def hash_state(self, corp: "Corporation"):
        state = json.dumps(self.get_state(corp))
        return hashlib.sha1(state.encode("utf-8")).hexdigest()


class BellmanMDP:

    def __init__(self, state_disc: StateDisc, corp: "Corporation"):
        self.state_disc: StateDisc = state_disc
        self.corp: Corporation = corp
        self.env: dict = defaultdict(lambda: defaultdict(list))
        self.eval_queue: tuple = ()
        self.records: dict = defaultdict(lambda: defaultdict(list))
        self.max_iters: int = 10000
        self.gamma: float = 0.90
        self.theta: float = 1e-6
        self.V: dict = {}
        self.policy: dict = {}

    def step(self):
        if self.eval_queue:
            self._eval_action(self.corp.reward)
            self.value_iter()
            self.build_policy()

    def choose_action(self):
        epsilon = max(0.05, 1.0 - (self.corp.tick / self.corp.max_tick))
        state = self.state

        if random.random() < epsilon or state not in self.policy:
            # Explore: Random action
            action = None
        else:
            # Exploit: choose best action
            action = self.policy[state][0]

        self.record_action(action)

    def record_action(self, action: str):
        state_hash = self.state
        self.eval_queue = (state_hash, action)
        logger.info(f"Raw state is {self.state_disc.get_state(self.corp)}")
        logger.info(f"Performing {action} in state {state_hash} ")

    def _eval_action(self, reward: int):
        if self.eval_queue:
            prev_state, prev_action = self.eval_queue
            logger.info(f"Evaluating prev-state: {prev_state} for action {prev_action}")
            state_hash = self.state
            record = (state_hash, reward)
            self.records[prev_state][prev_action].append(record)

            self._update_env(prev_state, prev_action)

        self.eval_queue = ()

    def _update_env(self, prev_state: str, prev_action: str):

        #  = [(abc123, 4), (abc123, 2), (cde321, -20)]
        next_states = self.records[prev_state][prev_action]

        # Group states and collect rewards
        state_data = defaultdict(list)
        for state, reward in next_states:
            state_data[state].append(reward)

        # Calculate probabilities and average rewards
        total_count = len(next_states)
        result = []
        for state, rewards in state_data.items():
            probability = len(rewards) / total_count
            avg_reward = sum(rewards) / len(rewards)
            result.append((round(probability, 2), state, round(avg_reward, 2)))

        # Update the new enviroment
        self.env[prev_state][prev_action] = result
        return result

    def _q_value(self, state: str, action: str, oldV: dict, gamma: float):
        total = 0.0
        for prob, next_state, reward in self.env[state][action]:
            if next_state in oldV:
                total += prob * (reward + gamma * oldV[next_state])

        return total

    def value_iter(self):
        self.V = defaultdict(float, {s: 0.0 for s in (self.env.keys())})
        for _ in range(1, self.max_iters + 1):
            delta = 0.0
            oldV = dict(self.V)
            for state in self.env.keys():
                action_values = []
                for action in self.env[state]:
                    action_values.append(self._q_value(state, action, oldV, self.gamma))

                self.V[state] = round(max(action_values), 2)
                delta = max(delta, abs(self.V[state] - oldV[state]))

            if delta < self.theta:
                break

        return self.V

    def build_policy(self):
        for state in self.env.keys():
            best_action = None
            best_q = float("-inf")
            for action in self.env[state]:
                val = self._q_value(state, action, self.V, self.gamma)
                if val > best_q:
                    best_q = val
                    best_action = action
            self.policy[state] = (best_action, round(best_q, 2))

        return self.policy

    @property
    def state(self):
        return self.state_disc.hash_state(self.corp)
