import json

from src.evaluation.evaluation import evaluate, format_evaluation, save_evaluation
from src.policies.baselines import ConstantPolicy, RandomPolicy
from tests.fakes import FakeEnvironment


def test_evaluate_completing_agent():
    environment = FakeEnvironment(episode_length=10, flag_at_end=True, x_step=5)
    results = evaluate(environment, ConstantPolicy(1), num_episodes=3)

    assert results["num_episodes"] == 3
    assert results["completion_rate"] == 1.0
    assert results["mean_distance"] == 90  # 40 + 5 * 10
    assert results["mean_episode_length"] == 10
    assert results["steps_per_second"] > 0
    assert len(results["episodes"]) == 3


def test_evaluate_failing_agent_and_timeout():
    environment = FakeEnvironment(episode_length=100, flag_at_end=False)
    results = evaluate(environment, ConstantPolicy(1), num_episodes=2, timeout=7)

    assert results["completion_rate"] == 0.0
    assert results["mean_episode_length"] == 7


def test_format_and_save(tmp_path):
    environment = FakeEnvironment()
    results = evaluate(environment, RandomPolicy(seed=0), num_episodes=2)
    assert "completion_rate" in format_evaluation("random", results)

    path = tmp_path / "results.json"
    save_evaluation(str(path), {"random": results})
    assert json.loads(path.read_text())["random"]["num_episodes"] == 2


def test_random_policy_is_seedable_and_in_range():
    a, b = RandomPolicy(seed=1), RandomPolicy(seed=1)
    actions = [a.act(None) for _ in range(50)]
    assert actions == [b.act(None) for _ in range(50)]
    assert all(0 <= x < 7 for x in actions)
