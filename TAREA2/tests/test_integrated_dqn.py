"""Behavior checks for the shared A/B/C agent without running notebook experiments."""
import ast
import copy
import json
import random
from pathlib import Path
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_agent():
    namespace = {}
    notebook = json.loads((ROOT/'lander.ipynb').read_text())
    exec(''.join(notebook['cells'][4]['source']), namespace)
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.ClassDef) and node.name in {'Agente', 'AgenteDQN'}:
                exec(compile(ast.Module(body=[node], type_ignores=[]), 'lander.ipynb', 'exec'), namespace)
    return namespace['AgenteDQN']


class IntegratedAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent_class = load_agent()
        torch.set_num_threads(1)

    def agent(self, **kwargs):
        options = dict(semilla=7, batch_size=4, learning_starts=8,
                       train_frequency=4, target_sync_steps=12)
        options.update(kwargs)
        return self.agent_class(gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32),
                                gym.spaces.Discrete(4), **options)

    def feed(self, agent, steps):
        state = np.zeros(8, dtype=np.float32)
        for i in range(steps):
            agent.aprender(state, i % 4, float(i+1), state+.1, i % 2 == 0)

    def test_no_replay_uses_only_current_transition_with_matched_update_schedule(self):
        full, no_replay = self.agent(), self.agent(use_replay=False)
        initial = copy.deepcopy(no_replay.red_q.state_dict())
        with patch.object(no_replay, '_actualizar_red', wraps=no_replay._actualizar_red) as update:
            self.feed(no_replay, 12)
        self.feed(full, 12)
        self.assertEqual(full.optimizer_updates, 2)
        self.assertEqual(no_replay.optimizer_updates, full.optimizer_updates)
        self.assertEqual(len(no_replay.memoria), 0)
        self.assertEqual(no_replay.effective_batch_size, 1)
        self.assertEqual([call.args[0][0][2] for call in update.call_args_list], [8., 12.])
        self.assertTrue(all(len(call.args[0]) == 1 for call in update.call_args_list))
        self.assertTrue(any(not torch.equal(initial[k], value) for k, value in no_replay.red_q.state_dict().items()))

    def test_disabled_exploration_stays_greedy_and_does_not_consume_randomness(self):
        for schedule in ['linear_episodes', 'linear_steps', 'exponential_episodes']:
            agent = self.agent(use_exploration=False, epsilon_schedule=schedule)
            state = np.zeros(8, dtype=np.float32)
            expected = agent.elegir_accion(state, explorar=False)
            with patch.object(random, 'random', side_effect=AssertionError('exploration RNG used')):
                self.assertEqual(agent.elegir_accion(state), expected)
            self.feed(agent, 12)
            agent.fin_episodio()
            self.assertEqual(agent.epsilon, 0.)
            self.assertEqual(agent.epsilon_en(500, 50000), 0.)

    def test_schedule_units_and_floors(self):
        linear = self.agent(epsilon_decay_episodes=2)
        self.feed(linear, 12)
        self.assertEqual(linear.epsilon, 1.)
        linear.fin_episodio()
        self.assertAlmostEqual(linear.epsilon, .525)
        linear.fin_episodio()
        self.assertAlmostEqual(linear.epsilon, .05)
        steps = self.agent(epsilon_schedule='linear_steps', epsilon_decay_steps=16)
        self.feed(steps, 8)
        self.assertAlmostEqual(steps.epsilon, .525)
        steps.fin_episodio()
        self.assertAlmostEqual(steps.epsilon, .525)
        self.feed(steps, 12)
        self.assertAlmostEqual(steps.epsilon, .05)
        exponential = self.agent(epsilon_schedule='exponential_episodes', epsilon_decay_factor=.5)
        exponential.fin_episodio()
        self.assertEqual(exponential.epsilon, .5)
        for _ in range(10):
            exponential.fin_episodio()
        self.assertEqual(exponential.epsilon, .05)

    def test_soft_target_updates_only_after_optimizer_and_remains_frozen(self):
        agent = self.agent(target_update='soft', tau=.25, target_sync_steps=1)
        initial = copy.deepcopy(agent.red_objetivo.state_dict())
        self.feed(agent, 7)
        self.assertTrue(all(torch.equal(initial[k], v) for k, v in agent.red_objetivo.state_dict().items()))
        self.feed(agent, 1)
        for key, target in agent.red_objetivo.state_dict().items():
            torch.testing.assert_close(target, .75*initial[key] + .25*agent.red_q.state_dict()[key])
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in agent.red_objetivo.parameters()))

    def test_target_off_bootstraps_online_with_terminal_mask_and_no_gradient(self):
        agent = self.agent(use_target=False, use_replay=False)
        self.assertFalse(hasattr(agent, 'red_objetivo'))
        with torch.no_grad():
            for p in agent.red_q.parameters():
                p.zero_()
            agent.red_q[-1].bias.fill_(3)
        targets = agent._calcular_objetivos(torch.zeros(2, 8), torch.tensor([2., 2.]), torch.tensor([True, False]))
        torch.testing.assert_close(targets, torch.tensor([2., 4.97]))
        self.assertFalse(targets.requires_grad)
        self.feed(agent, 12)
        self.assertEqual(agent.optimizer_updates, 2)

    def test_greedy_evaluation_has_no_training_or_rng_side_effects(self):
        for options in [{}, {'use_replay': False}, {'use_exploration': False}, {'use_target': False}]:
            agent = self.agent(**options)
            self.feed(agent, 12)
            before = (agent.epsilon, agent.env_steps, agent.optimizer_updates, len(agent.memoria))
            weights = copy.deepcopy(agent.red_q.state_dict())
            rng = random.getstate()
            for _ in range(5):
                agent.elegir_accion(np.zeros(8, dtype=np.float32), explorar=False)
            self.assertEqual(before, (agent.epsilon, agent.env_steps, agent.optimizer_updates, len(agent.memoria)))
            self.assertEqual(rng, random.getstate())
            self.assertTrue(all(torch.equal(weights[k], v) for k, v in agent.red_q.state_dict().items()))

    def test_invalid_switches_and_schedules_rejected(self):
        for options in [{'use_replay': 'False'}, {'use_exploration': 0}, {'epsilon_schedule': 'unknown'},
                        {'epsilon_decay_steps': 0}, {'epsilon_decay_factor': 1.1},
                        {'target_update': 'unknown'}, {'tau': 0}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.agent(**options)


if __name__ == '__main__':
    unittest.main()
