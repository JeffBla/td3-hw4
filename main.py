import argparse

from td3_agent_CarRacing import CarRacingTD3Agent

parse = argparse.ArgumentParser()
parse.add_argument('--gpu', action='store_true', help='use gpu or not')
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
                   default=1000000,
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
parse.add_argument('--noise_clip',
                   type=float,
                   default=0.5,
                   help='noise clip for target policy smoothing')
parse.add_argument('--seed', type=int, default=42, help='random seed')
parse.add_argument('--twin', action='store_true', help='use twin Q-networks')
parse.add_argument('--target_policy_smoothing',
                   action='store_true',
                   help='use target policy smoothing')
parse.add_argument('--exploration_noise_std',
                   type=float,
                   default=0.1,
                   help='standard deviation of Gaussian exploration noise')
# reward shaping parameters
parse.add_argument('--use_shaping',
                   action='store_true',
                   help='use reward shaping or not')
parse.add_argument('--w_onroad',
                   type=float,
                   default=0.5,
                   help='weight for on road reward')
parse.add_argument('--w_antigrass',
                   type=float,
                   default=0.3,
                   help='weight for anti grass reward')
parse.add_argument('--w_smooth',
                   type=float,
                   default=0.05,
                   help='weight for smooth steering reward')
parse.add_argument('--w_brake',
                   type=float,
                   default=0.02,
                   help='weight for brake penalty')
parse.add_argument('--brake_thresh',
                   type=float,
                   default=0.2,
                   help='brake threshold')
parse.add_argument('--w_throttle',
                   type=float,
                   default=0.05,
                   help='weight for throttle reward')
parse.add_argument('--time_penalty',
                   type=float,
                   default=0.001,
                   help='time penalty per step')
parse.add_argument('--road_min_for_ok',
                   type=int,
                   default=10,
                   help='minimum road pixel count to be considered on road')
parse.add_argument('--heavy_offroad_penalty',
                   type=float,
                   default=60.0,
                   help='penalty for heavy offroad situation')
args = parse.parse_args()

if __name__ == '__main__':
    config = vars(args)
    agent = CarRacingTD3Agent(config)
    agent.train()
