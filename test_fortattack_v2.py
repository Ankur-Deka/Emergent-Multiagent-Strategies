import sys
# sys.path.append('./mape')
import os
import json
import datetime
import numpy as np
import torch
import utils
from utils import normalize_obs

import random
from copy import deepcopy
from arguments import get_args
from tensorboardX import SummaryWriter      # use pip install tensorboardX==1.8 not 1.7
from eval import evaluate
from learner import setup_master
from pprint import pprint
import gym, gym_fortattack
from gym.wrappers import Monitor
import time

np.set_printoptions(suppress=True, precision=4)


def test_fortattack(args, seed, policies_list, ob_rms):
    writer = SummaryWriter(args.log_dir)      # some issue in importing tensorboardX
    
    if seed is None: # ensure env eval seed is different from training seed
        seed = np.random.randint(0,100000)
    print("Evaluation Seed: ",seed)
    env = utils.make_single_env(args)
    master = setup_master(args, env) 
    obs = env.reset() # shape - num_agents (total) x obs_dim
    all_obs = [obs]


    # if ob_rms is not None:
    #     obs_mean, obs_std = ob_rms
    # else:
    #     obs_mean = None
    #     obs_std = None

    master.load_models(policies_list)       ## we are done setting the guards
    master.set_eval_mode()    

    n, n_guards, n_attackers = len(master.all_agents), master.env.world.numGuards, master.env.world.numAliveAttackers
    # final_rewards = torch.zeros([args.num_processes, n], device=args.device)

    num_attacker_ckpts = len(args.attacker_ckpts)
    stats = np.zeros((num_attacker_ckpts, 4+2+2))    # gameResult, num_alive for guards/acttackers, rewards for guards/attackers
    

    # start simulations
    start = datetime.datetime.now()
    
    for i,attacker_ckpt in enumerate(args.attacker_ckpts):      # iterate through attacker strategies
        print('Playing against attacker strategy {}, ckpt {}'.format(i, attacker_ckpt))
        master.select_attacker(attacker_ckpt)

        data = np.zeros((args.num_eval_episodes, 4+2+2))    # gameResult, num_alive for guards/acttackers, rewards for guards/attackers
    
        for j in range(args.num_eval_episodes):                                      # 3 test runs for each strategy
            episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
            
            master.initialize_obs(obs)
            step = 0

            done = False
            while not done:  # while episode is not over
                masks = torch.FloatTensor(obs[:,0])		##* check the values of masks, agent alive or dead
                if not args.no_cuda:
                    masks = masks.cuda()
                
                with torch.no_grad():
                    actions_list, attn_list = master.act(step, masks) ## IMPORTANT
                agent_actions = np.array(actions_list).reshape(-1)
                obs, reward, done, info = env.step(agent_actions)

                all_obs.append(obs)

                
                if args.render:
                    env.render(attn_list)	# attn_list = [[teamates_attn, opp_attn] for each team]
                    time.sleep(0.06)

                
                reward = torch.from_numpy(np.stack(reward)).float().to(args.device)

                master.update_rollout(obs, reward, masks)   ## adds the data point to the rolloutstorage
                episode_rewards += reward*masks
                
                step += 1
                if done:
                    print(master.env.world.gameResult)
                    data[j,:2] = master.env.world.gameResult[:2]
                    data[j,2] = data[j,0] + data[j,1]
                    data[j,3] = master.env.world.gameResult[2]
                    data[j,4] = master.env.world.numAliveGuards
                    data[j,5] = master.env.world.numAliveAttackers 
                    data[j,6] = np.average(episode_rewards[0,0:n_guards])
                    data[j,7] = np.average(episode_rewards[0,n_guards:]) 

                    obs = env.reset()
                    masks = torch.FloatTensor(obs[:,0])     ##* check the values of masks, agent alive or dead
                    if not args.no_cuda:
                        masks = masks.cuda()
                    if args.render:
                        time.sleep(2)
                    if not args.out_file is None:
                        all_obs = np.array(all_obs)
                        print('all_obs', all_obs.shape)
                        print('path', args.out_file)
                        np.save(args.out_file, all_obs)
                        break
            
            
            print('episode_rewards')
            print(episode_rewards)
            # time.sleep(2)
        stats[i] = data.mean(axis = 0)
    stats = stats.round(2)
    print('stats')
    print(stats)
    np.savetxt("marlsave/stats/stats_ensemble_strategies.csv", stats, delimiter=",")

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    torch.manual_seed(args.seed)    # set the random seed
    torch.set_num_threads(1)        # only one threah. ##* what's thread and what's process. I guess one process can have multiple treads 
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)   # GPU :)

    checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
    policies_list = checkpoint['models']
    ob_rms = checkpoint['ob_rms']
    test_fortattack(args, args.seed, policies_list, ob_rms)
    