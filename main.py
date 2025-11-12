from agents.market.agent import Market
from settings import SimConfig
from seeds import CorpSeed, PersonSeed

sim_config = SimConfig(
    nr_of_ticks=5000,
    nr_of_people=1500,
    nr_of_corps=3,
    min_base_salary=50,
    max_base_salary=125,
)

corp_seed = CorpSeed(
    balance=50000,
    price=20,
    salary=100,
    nr_of_employees=10,
    upe=25,
)

person_seed = PersonSeed(
    mpc=0.5,
    balance=0,
)


if __name__ == "__main__":
    market = Market(
        sim_config=sim_config,
        corp_seed=corp_seed,
        person_seed=person_seed,
    )
    market.run()
    market.generate_charts()
