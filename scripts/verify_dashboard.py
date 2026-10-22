"""Check the dashboard picks the right training script per algorithm and that
the run/train templates render for a DQN run."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jinja2 import Environment, FileSystemLoader

from dashboard.server.train_api import ALGORITHM_SCRIPTS

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "templates"
)

print("=== algorithm -> script dispatch ===")
for algorithm, script in ALGORITHM_SCRIPTS.items():
    print(f"  {algorithm:<6} -> {script}")

assert ALGORITHM_SCRIPTS["dqn"] == "scripts/train_dqn.py"
assert ALGORITHM_SCRIPTS["ppo"] == "scripts/train.py"
print("  OK")

print()
print("=== template rendering ===")
env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))

# A DQN run: metrics carry loss/td_error instead of policy_loss/value_loss.
dqn_metrics = [
    {"step": 1000, "loss": 0.5, "td_error": 1.2, "epsilon": 0.9,
     "mean_distance": 400.0, "mean_reward": 20.0},
    {"step": 2000, "loss": 0.3, "td_error": 0.9, "epsilon": 0.8,
     "mean_distance": 700.0, "mean_reward": 30.0},
]
run = {
    "id": "dqn_20261022_120000",
    "metrics": dqn_metrics,
    "episodes": [],
    "config": {"algorithm": "dqn"},
}

run_html = env.get_template("run.html").render(
    run=run, plots={}, algorithms=["ppo", "dqn"],
)
assert "loss" in run_html
assert "td_error" in run_html
print("  run.html renders for a DQN run: OK")

train_html = env.get_template("train.html").render(
    presets={
        "ppo": {"algorithm": "ppo", "updates": 100, "rollout_size": 512},
        "dqn": {"algorithm": "dqn", "total_steps": 1000, "batch_size": 32},
    },
    algorithms=["ppo", "dqn"],
)
assert "total_steps" in train_html
assert "DQN_FIELDS" in train_html
print("  train.html renders with DQN fields: OK")

print()
print("All dashboard checks passed.")
