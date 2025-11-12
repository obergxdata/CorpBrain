# CorpBrain

A proof-of-concept economic simulation that uses **Reinforcement Learning (RL)** to optimize corporate pricing strategies in a competitive market environment.

## Overview

CorpBrain simulates a market economy where:
- **Corporations** compete by adjusting prices to maximize profit
- **Consumers** purchase products based on price and budget constraints
- **RL agents** learn optimal pricing strategies through Bellman equations and value iteration

The simulation uses **Markov Decision Processes (MDP)** with Q-learning/value iteration to find optimal policies for corporate decision-making.

### Implementation Approach
This is a **from-scratch implementation** using basic Python and numpy:
- **No external RL libraries** (e.g., no TensorFlow, PyTorch, Stable-Baselines)
- **No neural networks** - uses tabular Q-learning with discretized states
- **Independent learners** - each corporation learns independently (not Multi-Agent RL)
- Pure algorithmic approach focused on understanding RL fundamentals

## Key Features

### Reinforcement Learning
- **Bellman MDP Implementation**: Each corporation uses value iteration to learn optimal pricing
- **Epsilon-Greedy Exploration**: Balances exploration vs exploitation (ε decays over time)
- **State Discretization**: Market conditions bucketed into discrete states for Q-value computation
- **Policy Building**: Automatically constructs optimal policies from value functions

### Economic Realism
- **Imperfect Information**: Consumers only see 50% of available corporations (models limited market knowledge)
- **Price Competition**: Pure price-based competition in commodity markets
- **Income Distribution**: Normally distributed salaries across population
- **Marginal Propensity to Consume (MPC)**: Realistic consumer spending behavior

### Agent Types

#### Corporation Agent
- Manages inventory, employees, and cash flow
- Takes pricing actions: increase/decrease price by various amounts
- Receives rewards based on profit margins
- Learns pricing strategies via MDP/RL

#### Person Agent
- Receives salary each tick
- Spends based on MPC (marginal propensity to consume)
- Samples 50% of market (imperfect information)
- Buys from cheapest available options within budget

#### Market Agent
- Orchestrates simulation ticks
- Tracks market-wide statistics (avg price, profit, sales)
- Generates visualization charts
- Manages corporation lifecycle (bankruptcy detection)

## Installation

### Requirements
- Python 3.10+
- Dependencies: `numpy`, `pydantic`, `pytest`, `matplotlib` (for charts)

### Setup
```bash
# Clone the repository
git clone <repo-url>
cd CorpBrain

# Install dependencies (if requirements.txt exists)
pip install numpy pydantic pytest matplotlib
```

## Usage

### Running the Simulation

```bash
python main.py
```

This will:
1. Initialize market with configured corporations and consumers
2. Run the simulation for specified number of ticks (default: 5000)
3. Generate charts in the `charts/` directory showing:
   - Average market prices over time
   - Corporate profit trends
   - Individual corporation pricing strategies

### Configuration

Edit `main.py` to adjust simulation parameters:

```python
sim_config = SimConfig(
    nr_of_ticks=5000,      # Simulation length
    nr_of_people=1500,      # Number of consumers
    nr_of_corps=3,          # Number of competing corporations
    min_base_salary=50,     # Minimum consumer salary
    max_base_salary=125,    # Maximum consumer salary
)

corp_seed = CorpSeed(
    balance=50000,          # Starting capital
    price=20,               # Initial product price
    salary=100,             # Employee wages
    nr_of_employees=10,     # Initial workforce
    upe=25,                 # Units produced per employee
)

person_seed = PersonSeed(
    mpc=0.5,                # Marginal propensity to consume (50%)
    balance=0,              # Starting savings
)
```

## Testing

Run tests using the Makefile:

```bash
# Run all agent tests
make test-agents

# Run MDP/RL tests specifically
make test-mdp

# Run full market simulation test (slow)
make test-sim

# Show all available targets
make help
```

Or use pytest directly:
```bash
pytest agents/*/test.py -v
pytest agents/*/RL/tests.py -v
```

## How It Works

### Simulation Loop

Each tick:
1. **Corporation Step**:
   - Produce inventory (employees × UPE)
   - Choose pricing action using RL policy
   - Record current state

2. **Consumer Step**:
   - Receive salary
   - Sample 50% of corporations (imperfect info)
   - Sort by price (cheapest first)
   - Purchase products until budget exhausted

3. **Corporation Finish**:
   - Calculate profit/loss
   - Pay employee salaries
   - Update MDP with rewards
   - Trigger value iteration (ε-greedy)
   - Detect bankruptcy

4. **Recording**: Track market statistics and generate time-series data

### RL Implementation Details

**State Space**:
- `bucket_price`: Corporation's price relative to market min/max (normalized 0-1)
- `bucket_relative_price`: Price relative to consumer budgets
- States are discretized into buckets (0.1 step size) and hashed for lookup

**Action Space**:
- Price changes: +/- various percentages
- Actions are evaluated after observing rewards (sales, profit)

**Reward Signal**:
- Based on profit: `revenue - costs`
- Sparse rewards encourage exploration early on

**Value Iteration**:
- Bellman equation: `V(s) = max_a Σ P(s'|s,a)[R(s,a,s') + γV(s')]`
- Converges when `max|V_new - V_old| < θ`
- Gamma (discount): 0.90
- Theta (convergence): 1e-6

**Policy**:
- Greedy policy: `π(s) = argmax_a Q(s,a)`
- Extracted after each value iteration

## Project Structure

```
CorpBrain/
├── agents/
│   ├── corporations/
│   │   ├── agent.py           # Corporation logic
│   │   ├── RL/
│   │   │   ├── MDP.py         # Bellman MDP, value iteration
│   │   │   └── tests.py       # RL unit tests
│   │   └── test.py            # Corporation tests
│   ├── people/
│   │   ├── agent.py           # Consumer behavior
│   │   └── test.py            # Consumer tests
│   └── market/
│       ├── agent.py           # Market orchestration
│       └── test_sim.py        # Integration tests
├── stats/
│   └── history.py             # Time-series tracking & charting
├── main.py                    # Entry point
├── settings.py                # Configuration models
├── seeds.py                   # Initial state definitions
├── Makefile                   # Test commands
└── README.md                  # This file
```

## Key Insights

### Why RL for Pricing?
Traditional economic models assume rational actors with perfect information. Real markets are:
- **Dynamic**: Competitor actions change optimal strategies
- **Uncertain**: Consumer behavior has randomness
- **Multi-agent**: Each corporation's actions affect others

RL naturally handles these properties by learning from experience rather than assuming equilibria.

### Market Dynamics Observed
- **Price Wars**: Early exploration often triggers undercutting
- **Convergence**: Prices stabilize as policies converge
- **Bankruptcy Risk**: Aggressive pricing can deplete capital
- **Market Concentration**: Weaker firms exit, survivors gain market share

### Imperfect Information Impact
By limiting consumers to 50% market visibility:
- Even non-cheapest corporations get sales
- Price dispersion persists longer
- More realistic than perfect competition models

## Future Work

### Near-term Enhancements
- [ ] **Workforce management actions**: Hire/fire employees dynamically
- [ ] **Production decisions**: Adjust output levels based on demand


