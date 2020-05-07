#!/usr/bin/env python3
import os
import gym
import ptan
import argparse
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lib import environ, data, models, common, validation

GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4

BARS_COUNT = 10
REWARD_STEPS = 2
DEFAULT_STOCKS = "data/YNDX_160101_161231.csv"
DEFAULT_VAL_STOCKS = "data/YNDX_150101_151231.csv"

CHECKPOINT_EVERY_STEP = 10000
VALIDATION_EVERY_STEP = 1000

class PGN(nn.Module):
	def __init__(self, input_size, n_actions):
		super(PGN, self).__init__()

		self.net = nn.Sequential(
			nn.Linear(input_size, 128),
			nn.ReLU(),
			nn.Linear(128, n_actions)
		)

	def forward(self, x):
		return self.net(x)


def calc_qvals(rewards):
	res = []
	sum_r = 0.0
	for r in reversed(rewards):
		sum_r *= GAMMA
		sum_r += r
		res.append(sum_r)
	return list(reversed(res))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
	parser.add_argument("--data", default=DEFAULT_STOCKS, help="Stocks file or dir to train on, default=" + DEFAULT_STOCKS)
	parser.add_argument("--year", type=int, help="Year to be used for training, if specified, overrides --data option")
	parser.add_argument("--valdata", default=DEFAULT_VAL_STOCKS, help="Stocks data for validation, default=" + DEFAULT_VAL_STOCKS)
	parser.add_argument("-r", "--run", required=True, help="Run name")
	args = parser.parse_args()
	device = torch.device("cuda" if args.cuda else "cpu")

	saves_path = os.path.join("saves", args.run)
	os.makedirs(saves_path, exist_ok=True)

	if args.year is not None or os.path.isfile(args.data):
		if args.year is not None:
			stock_data = data.load_year_data(args.year)
		else:
			stock_data = {"YNDX": data.load_relative(args.data)}
		env = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False, volumes=False)
		env_tst = environ.StocksEnv(stock_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
	elif os.path.isdir(args.data):
		env = environ.StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
		env_tst = environ.StocksEnv.from_dir(args.data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
	else:
		raise RuntimeError("No data to train on")
	env = gym.wrappers.TimeLimit(env, max_episode_steps=1000)

	val_data = {"YNDX": data.load_relative(args.valdata)}
	env_val = environ.StocksEnv(val_data, bars_count=BARS_COUNT, reset_on_close=True, state_1d=False)
	
	#env = gym.make("CartPole-v0")
	writer = SummaryWriter(comment="-Stocks-PG")

	net = PGN(env.observation_space.shape[0], env.action_space.n).to(device)
	print(net)
	
	selector = ptan.actions.ProbabilityActionSelector()
	agent = ptan.agent.PolicyAgent(net, action_selector=selector, preprocessor=ptan.agent.float32_preprocessor,
								   apply_softmax=True)
	exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

	optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

	total_rewards = []
	step_idx = 0
	done_episodes = 0

	batch_episodes = 0
	batch_states, batch_actions, batch_qvals = [], [], []
	cur_rewards = []

	with common.RewardTracker(writer, np.inf, group_rewards=100) as reward_tracker:
		for step_idx, exp in enumerate(exp_source):
			batch_states.append(exp.state)
			batch_actions.append(int(exp.action))
			cur_rewards.append(exp.reward)

			if exp.last_state is None:
				batch_qvals.extend(calc_qvals(cur_rewards))
				cur_rewards.clear()
				batch_episodes += 1

			# handle new rewards
			#new_rewards = exp_source.pop_total_rewards()
			new_rewards = exp_source.pop_rewards_steps()
			if new_rewards:
				done_episodes += 1
				reward = new_rewards[0]
				total_rewards.append(reward)
				reward_tracker.reward(new_rewards[0], step_idx)
				mean_rewards = float(np.mean(total_rewards[-100:]))
				#print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d" % (
					#step_idx, reward, mean_rewards, done_episodes))
				#writer.add_scalar("reward", reward, step_idx)
				writer.add_scalar("reward_100", mean_rewards, step_idx)
				writer.add_scalar("episodes", done_episodes, step_idx)
				#if mean_rewards > 195:
					#print("Solved in %d steps and %d episodes!" % (step_idx, done_episodes))
					#break

			if batch_episodes < EPISODES_TO_TRAIN:
				continue

			optimizer.zero_grad()
			states_v = torch.FloatTensor(batch_states)
			batch_actions_t = torch.LongTensor(batch_actions)
			batch_qvals_v = torch.FloatTensor(batch_qvals)

			logits_v = net(states_v)
			log_prob_v = F.log_softmax(logits_v, dim=1)
			log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
			loss_v = -log_prob_actions_v.mean()

			loss_v.backward()
			optimizer.step()

			batch_episodes = 0
			batch_states.clear()
			batch_actions.clear()
			batch_qvals.clear()
			
			if step_idx % CHECKPOINT_EVERY_STEP == 0:
				idx = step_idx // CHECKPOINT_EVERY_STEP
				torch.save(net.state_dict(), os.path.join(saves_path, "checkpoint-%3d.data" % idx))

			if step_idx % VALIDATION_EVERY_STEP == 0:
				res = validation.validation_run(env_tst, net, device=device)
				for key, val in res.items():
					writer.add_scalar(key + "_test", val, step_idx)
				res = validation.validation_run(env_val, net, device=device)
				for key, val in res.items():
					writer.add_scalar(key + "_val", val, step_idx)

	writer.close()
