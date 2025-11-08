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
        self.state_map: dict = {}

    def bucket_0_1(self, value: float, min_value: float = 0.0, max_value: float = 1.0):
        # Normalize value to 0-1 range
        if min_value == max_value:
            normalized = 0.5
        else:
            normalized = (value - min_value) / (max_value - min_value)

        # Apply bucketing to normalized value (10 buckets: 0-9)
        bucket = min(int(normalized * 20), 19)  # Ensures 1.0 goes to bucket 9
        return f"bucket_{bucket}"

    def bucket_price(self, corp: "Corporation"):
        min_price, max_price = self.market.min_max_price
        price = corp.price
        return self.bucket_0_1(price, min_value=min_price, max_value=max_price)

    def bucket_mpc(self):
        return self.bucket_0_1(self.market.avg_mpc)

    def bucket_relative_price(self, corp: "Corporation"):
        min_salary, max_salary = self.market.min_max_salary
        mpc = self.market.people[0].mpc
        min_budget = min_salary * mpc
        max_budget = max_salary * mpc

        # Use consumer budget range as reference (deterministic)
        # Prices below min_budget → negative buckets (discount pricing)
        # Prices above max_budget → high positive buckets (premium pricing)
        # Each bucket represents 5% of the budget range
        return self.bucket_0_1(
            value=corp.price, min_value=min_budget, max_value=max_budget
        )

    def bucket_profit_trend(self, corp: "Corporation"):
        return self.bucket_0_1(corp.profit_trend(zero=True))

    def profit_trend(self, corp: "Corporation"):

        profit = corp.history.tail("profit", 2)
        if len(profit) == 2:
            prev = profit[1]
            now = profit[0]
            if prev == now:
                return 0
            elif prev > now:
                return -1
            else:
                return 1
        else:
            return 0

    def bucket_market_share(self, corp: "Corporation"):
        min_market, max_market = self.market.min_max_market_share
        market_share = self.market.market_share(corp)

        return self.bucket_0_1(market_share, min_value=min_market, max_value=max_market)

    def get_state(self, corp: "Corporation"):
        state = {
            "bucket_price": self.bucket_price(corp),
            "bucket_relative_price": self.bucket_relative_price(corp),
            # "bucket_sales:": self.profit_trend(corp),
        }

        logger.debug(state)
        return state

    def hash_state(self, corp: "Corporation"):
        state = self.get_state(corp)
        str_state = json.dumps(state)
        hashed = hashlib.sha1(str_state.encode("utf-8")).hexdigest()
        self.state_map[hashed] = state
        return hashed


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
        epsilon = max(0.05, 1.0 - (self.corp.tick / (self.corp.max_tick * 0.75)))
        state = self.state

        if random.random() < epsilon or state not in self.policy:
            # Explore: Random action
            logger.info("Explore random action")
            random_key = random.choice(list(self.corp.actions().keys()))
            action_set = self.corp.actions()[random_key]
            action_func = action_set[0]  # self.change_price
            action_value = random.choice(action_set[1])  # [0.1,0.4,0.6] -> 0.4
            action_func(action_value)
            action = f"{random_key}_{action_value}"
        else:
            # Exploit: choose best action
            logger.info("Exploit policy")
            action = self.policy[state][0]

            # Parse and execute the policy action
            # Action format: "change_price_0.1"
            action_parts = action.split("_")
            action_name = "_".join(action_parts[:-1])  # "change_price"
            action_value = float(action_parts[-1])  # 0.1

            # Get the action function and execute it
            action_func = self.corp.actions()[action_name][0]
            action_func(action_value)

        logger.info(f"Action is {action}")
        self.record_action(action, state=state)

    def record_action(self, action: str, state: str):
        self.eval_queue = (state, action)
        logger.info(f"Performing {action} in state {state} ")

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
        logger.debug(
            f"Updating {prev_state[:8]}[{prev_action}] with {len(next_states)} observations"
        )

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

    def format_env(self) -> str:
        """Format the MDP environment in a readable way."""
        lines = []

        for state_hash, actions in self.env.items():
            # State header
            readable_state = self.state_disc.state_map[state_hash]
            # state_short = state_hash[:8]
            lines.append(
                f"\nState: {readable_state} ({len(actions)} actions): {state_hash[:8]}"
            )

            # Each action
            for action, transitions in actions.items():
                lines.append(f"  Action: {action}")

                # Each transition
                for prob, next_state, reward in transitions:
                    next_short = next_state[:8]
                    lines.append(
                        f"    -> {next_short} (prob: {prob:.2f}, reward: {reward:.2f})"
                    )

        # Summary at the end
        lines.append(f"\n{'='*50}")
        lines.append(f"Total unique states: {len(self.env)}")

        return "\n".join(lines)
