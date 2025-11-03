import logging
import sys

logger = logging.getLogger("mdp")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Only set up handlers once
if not logger.handlers:
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.WARNING)
    console.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] MDP: %(message)s", datefmt="%H:%M:%S"
        )
    )
    logger.addHandler(console)

    file = logging.FileHandler("logs/MDP.log", mode="w")
    file.setLevel(logging.DEBUG)
    file.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(file)
