.PHONY: test help

help:
	@echo "Available targets:"
	@echo "  test    - Run all tests except test_sim"

test-agents:
	pytest agents/*/test.py -v

test-mdp:
	pytest agents/*/RL/tests.py -v

test-sim:
	pytest agents/market/test_sim.py -v