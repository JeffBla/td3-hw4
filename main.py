import argparse

from td3_agent_CarRacing import CarRacingTD3Agent

parse = argparse.ArgumentParser()
parse.add_argument('--gpu', action='store_true', help='use gpu or not')
parse.add_argument('--training_steps',
                   type=int,
                   default=1e8,
                   help='total training steps')
parse.add_argument('--gamma',
                   type=float,
                   default=0.99,
                   help='discounted factor')
parse.add_argument('--tau',
                   type=float,
                   default=0.005,
                   help='soft update parameter')
parse.add_argument('--batch_size',
                   type=int,
                   default=32,
                   help='batch size for sampling from replay buffer')
parse.add_argument('--warmup_steps',
                   type=int,
                   default=1000,
                   help='steps for the warmup stage')
parse.add_argument('--total_episode',
                   type=int,
                   default=100000,
                   help='total training episodes')
parse.add_argument('--lra',
                   type=float,
                   default=4.5e-5,
                   help='learning rate for actor network')
parse.add_argument('--lrc',
                   type=float,
                   default=4.5e-5,
                   help='learning rate for critic network')
parse.add_argument('--replay_buffer_capacity',
                   type=int,
                   default=5000,
                   help='capacity of replay buffer')
parse.add_argument('--logdir',
                   type=str,
                   default='log/CarRacing/td3_test/',
                   help='directory to save logs')
parse.add_argument('--update_freq',
                   type=int,
                   default=2,
                   help='frequency of updating networks')
parse.add_argument('--eval_interval',
                   type=int,
                   default=10,
                   help='interval (in episodes) for evaluating the agent')
parse.add_argument('--eval_episode',
                   type=int,
                   default=10,
                   help='number of episodes for each evaluation')
parse.add_argument('--seed', type=int, default=42, help='random seed')
args = parse.parse_args()

if __name__ == '__main__':
    config = vars(args)
    agent = CarRacingTD3Agent(config)
    agent.train()
