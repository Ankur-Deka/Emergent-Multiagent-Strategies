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

    master.load_models(policies_list)
    master.set_eval_mode()
    n = len(master.all_agents)
    # final_rewards = torch.zeros([args.num_processes, n], device=args.device)

    # start simulations
    start = datetime.datetime.now()
    for j in range(1):   # iterations
        end_pts = []    # maintain a list of time end points of episodes that can be used in master.wrap_horizon. Each entry is (where the episode got over) + 1
        episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
        
        print('j (num update)', j)
        
        ## training
        print('training')
        print('end_pts at the begining', end_pts, 'train_fortattack.py')

        # print('obs', obs[:,0])
        master.initialize_obs(obs)
        step = 0
        while step < 300:  # data collection steps in each iteration
            # print('step', step, 'train_fortattack')
            masks = torch.FloatTensor(obs[:,0])		##* check the values of masks, agent alive or dead
            if not args.no_cuda:
                masks = masks.cuda()
            
            with torch.no_grad():
                actions_list, attn_list = master.act(step, masks) ## IMPORTANT
            agent_actions = np.array(actions_list).reshape(-1)
            obs, reward, done, info = env.step(agent_actions)

            all_obs.append(obs)

            # obs = normalize_obs(obs, obs_mean, obs_std) 
            # print('reward')
            # print(reward)

            env.render(attn_list)	# attn_list = [[teamates_attn, opp_attn] for each team]
            time.sleep(0.06)

            
            # obs, newdead = obs_newdead
            # print('obs', obs.shape, 'masks', masks.shape)
            # print('done', done)
            reward = torch.from_numpy(np.stack(reward)).float().to(args.device)

            ##* Don't know what final_reward means
            # print(masks.shape, episode_rewards.shape)
            # print(masks)
            # print(episode_rewards)
            # print(masks.dtype, episode_rewards.dtype)
            # print(((1 - masks) * episode_rewards).dtype, (1-masks).dtype,masks.dtype, episode_rewards.dtype)
            # episode_rewards *= masks
            # final_rewards += episode_rewards  # it is (1-masks)*.., but i think it should be masks*... 

            # print('episode_rewards')
            # print(episode_rewards)
            # final_rewards *= masks
            master.update_rollout(obs, reward, masks)   ## adds the data point to the rolloutstorage
            episode_rewards += reward*masks
            # print('step reward', reward)
            # print('done', done)
            ## once masks is used, we can update it
            ## it's just easier to use masks
            # masks = torch.FloatTensor(1-1.0*done).to(args.device)      # mask is to not use rewards for agents 
            
            step += 1
            ##* need to confirm this 
            if done:
                end_pts.append(step)
                # time.sleep(1)
                obs = env.reset()
                masks = torch.FloatTensor(obs[:,0])     ##* check the values of masks, agent alive or dead
                if not args.no_cuda:
                    masks = masks.cuda()
                master.initialize_new_episode(step, obs, masks)
                time.sleep(1)
                if not args.out_file is None:
                    all_obs = np.array(all_obs)
                    print('all_obs', all_obs.shape)
                    print('path', args.out_file)
                    np.save(args.out_file, all_obs)
                    break
        
        # print('')
        # if end_pts[-1] != args.num_steps:
        #     end_pts.append(args.num_steps)
        # master.wrap_horizon(end_pts) ## computes the return = (reward + gamam*value), IMPORTANT
        # # master.before_update()
        # vals = master.update()   ## PPO update, IMPORTANT. Multiple iterations of PPO on the last episode
        # value_loss = vals[:, 0]
        # action_loss = vals[:, 1]
        # dist_entropy = vals[:, 2]
        # master.after_update() ## IMPORTANT
        
        # for agent in master.all_agents:
        #     print('after_update', agent.rollouts.obs.shape, 'train_fortattack.py')


        print('episode_rewqards')
        print(episode_rewards)
    #     ## Saving trained model
    #     if j%args.save_interval == 0 and not args.test:
    #         print('saving')
    #         savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
    #         ob_rms = (None, None) if env.ob_rms is None else (env.ob_rms[0].mean, env.ob_rms[0].var)
    #         savedict['ob_rms'] = ob_rms
    #         savedir = args.save_dir+'/ep'+str(j)+'.pt'
    #         # print(savedir)
    #         torch.save(savedict, savedir)

    #     total_num_steps = (j + 1) * args.num_processes * args.num_steps

    #     # Logginhg data to tensorboardX
    #     if j%args.log_interval == 0:
    #         print('logging')
    #         end = datetime.datetime.now()
    #         seconds = (end-start).total_seconds()
    #         total_reward = episode_rewards.sum(dim=0).cpu().numpy()
    #         print('total_reward')
    #         print(total_reward)
    #         print("Updates {} | Num timesteps {} | Time {} | FPS {}\nTotal reward {}\nEntropy {:.4f} Value loss {:.4f} Policy loss {:.4f}\n".
    #               format(j, total_num_steps, str(end-start), int(total_num_steps / seconds), 
    #               total_reward, dist_entropy[0], value_loss[0], action_loss[0]))
    #         if not args.test:
    #             for idx in range(n):
    #                 writer.add_scalar('agent'+str(idx)+'/training_reward', total_reward[idx], j)
    #                 print('idx', idx, 'total_reward[idx]', total_reward[idx])

    #             writer.add_scalar('all/value_loss', value_loss[0], j)
    #             writer.add_scalar('all/action_loss', action_loss[0], j)
    #             writer.add_scalar('all/dist_entropy', dist_entropy[0], j)

    

    # writer.close()
    # if return_early:
    #     return savedir

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
    