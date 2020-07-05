import sys
# sys.path.append('./mape')
import os
import json
import datetime
import numpy as np
import torch
import utils
import random
from copy import deepcopy
from arguments import get_args
from tensorboardX import SummaryWriter      # use pip install tensorboardX==1.8 not 1.7
from eval import evaluate
from learner import setup_master
from pprint import pprint
import gym, gym_fortattack
import time

np.set_printoptions(suppress=True, precision=4)


def train(args, policies_list, return_early=False):
    writer = SummaryWriter(args.log_dir)      # some issue in importing tensorboardX
    # env = utils.make_parallel_envs(args) 
    env = utils.make_single_env(args)
    master = setup_master(args, env) 
    # used during evaluation only
    eval_master, eval_env = setup_master(args, return_env=True) 
    obs = env.reset() # shape - num_agents (total) x obs_dim
    
    if args.continue_training:
        master.load_models(policies_list)

    if args.train_guards_only:
        master.sample_attacker()

    n = len(master.all_agents)
    # final_rewards = torch.zeros([args.num_processes, n], device=args.device)

    # start simulations
    start = datetime.datetime.now()
    shift = int(args.ckpt)+1 if args.continue_training else 0   # for continuing training
    for j in range(shift, args.num_updates+shift):   # iterations
        end_pts = []    # maintain a list of time end points of episodes that can be used in master.wrap_horizon. Each entry is (where the episode got over) + 1
        episode_rewards = torch.zeros([args.num_processes, n], device=args.device)
        
        print('j (num update)', j)
        
        ## collecting samples
        print('collecting samples')
        
        master.initialize_obs(obs)
        step = 0
        while step < args.num_steps:  # data collection steps in each iteration
            # print('step', step, 'train_fortattack')
            masks = torch.FloatTensor(obs[:,0])		##* check the values of masks, agent alive or dead
            if not args.no_cuda:
                masks = masks.cuda()
           
            with torch.no_grad():
                actions_list, attn_list = master.act(step, masks) ## IMPORTANT
            agent_actions = np.array(actions_list).reshape(-1)
            obs, reward, done, info = env.step(agent_actions)
            
            # print('reward')
            # print(reward)

            if args.render:
                env.render(attn_list)	# attn_list = [[teamates_attn, opp_attn] for each team]
                time.sleep(0.01)


            
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
                if args.train_guards_only:
                    master.sample_attacker()    # choose a one pre-trained attacker policy
                # break
        
        # print('')
        if end_pts[-1] != args.num_steps:
            end_pts.append(args.num_steps)
        master.wrap_horizon(end_pts) ## computes the return = (reward + gamam*value), IMPORTANT
        # master.before_update()

        vals = master.update(train_guards_only = args.train_guards_only)   ## PPO update, IMPORTANT. Multiple iterations of PPO on the last episode
        value_loss = vals[:, 0]
        action_loss = vals[:, 1]
        dist_entropy = vals[:, 2]
        master.after_update() ## IMPORTANT
        
       

         ## Saving trained model
        if j%args.save_interval == 0 and not args.test:
            print('saving')
            savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
            ob_rms = (None, None) if env.ob_rms is None else (env.ob_rms[0].mean, env.ob_rms[0].var)
            savedict['ob_rms'] = ob_rms
            savedir = args.save_dir+'/ep'+str(j)+'.pt'
            torch.save(savedict, savedir)

        total_num_steps = (j + 1) * args.num_processes * args.num_steps

        # Logginhg data to tensorboardX
        if j%args.log_interval == 0:
            print('logging')
            end = datetime.datetime.now()
            seconds = (end-start).total_seconds()
            total_reward = episode_rewards.sum(dim=0).cpu().numpy()
            print('total_reward')
            print(total_reward)
            print("Updates {} | Num timesteps {} | Time {} | FPS {}\nTotal reward {}\nEntropy {:.4f} Value loss {:.4f} Policy loss {:.4f}\n".
                  format(j, total_num_steps, str(end-start), int(total_num_steps / seconds), 
                  total_reward, dist_entropy[0], value_loss[0], action_loss[0]))
            if not args.test:
                for idx in range(n):
                    writer.add_scalar('agent'+str(idx)+'/training_reward', total_reward[idx], j)
                    print('idx', idx, 'total_reward[idx]', total_reward[idx])

                writer.add_scalar('all/value_loss', value_loss[0], j)
                writer.add_scalar('all/action_loss', action_loss[0], j)
                writer.add_scalar('all/dist_entropy', dist_entropy[0], j)

        # ## evaluation/validation
        # if args.eval_interval is not None and j%args.eval_interval==0:
        #     print('evaluating')
        #     ob_rms = (None, None) if env.ob_rms is None else (env.ob_rms[0].mean, env.ob_rms[0].var)
        #     print('===========================================================================================')
        #     _, eval_perstep_rewards, final_min_dists, num_success, eval_episode_len = evaluate(args, None, master.all_policies,
        #                                                                                        ob_rms=ob_rms, env=eval_env,
        #                                                                                        render = args.render,
        #                                                                                        master=eval_master)
        #     print('Evaluation {:d} | Mean per-step reward {:.2f}'.format(j//args.eval_interval, eval_perstep_rewards.mean()))
        #     print('Num success {:d}/{:d} | Episode Length {:.2f}'.format(num_success, args.num_eval_episodes, eval_episode_len))
        #     if final_min_dists:
        #         print('Final_dists_mean {}'.format(np.stack(final_min_dists).mean(0)))
        #         print('Final_dists_var {}'.format(np.stack(final_min_dists).var(0)))
        #     print('===========================================================================================\n')

        #     if not args.test:
        #         writer.add_scalar('all/eval_success', 100.0*num_success/args.num_eval_episodes, j)
        #         writer.add_scalar('all/episode_length', eval_episode_len, j)
        #         for idx in range(n):
        #             writer.add_scalar('agent'+str(idx)+'/eval_per_step_reward', eval_perstep_rewards.mean(0)[idx], j)
        #             if final_min_dists:
        #                 writer.add_scalar('agent'+str(idx)+'/eval_min_dist', np.stack(final_min_dists).mean(0)[idx], j)
        #     # print('flag3')
        #     curriculum_success_thres = 0.9
        #     if return_early and num_success*1./args.num_eval_episodes > curriculum_success_thres:
        #         savedict = {'models': [agent.actor_critic.state_dict() for agent in master.all_agents]}
        #         ob_rms = (None, None) if env.ob_rms is None else (env.ob_rms[0].mean, env.ob_rms[0].var)
        #         savedict['ob_rms'] = ob_rms
        #         savedir = args.save_dir+'/ep'+str(j)+'.pt'
        #         torch.save(savedict, savedir)
        #         print('===========================================================================================\n')
        #         print('{} agents: training complete. Breaking.\n'.format(args.num_agents))
        #         print('===========================================================================================\n')
        #         break
        

    writer.close()
    if return_early:
        return savedir

if __name__ == '__main__':
    args = get_args()
    if args.seed is None:
        args.seed = random.randint(0,10000)
    args.num_updates = args.num_frames // args.num_steps // args.num_processes
    torch.manual_seed(args.seed)    # set the random seed
    torch.set_num_threads(1)        # only one threah. ##* what's thread and what's process. I guess one process can have multiple treads 
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)   # GPU :)

    pprint(vars(args))
    
    policies_list = None
    if args.continue_training:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
        policies_list = checkpoint['models']
        # ob_rms = checkpoint['ob_rms']   ## obs rms wasn't saved, so not used. Check later if it is required to be used/saved/loaded for this PPO implementation
    elif args.pretrained_guard:
        checkpoint = torch.load(args.guard_load_dir, map_location=lambda storage, loc: storage)
        policies_list = checkpoint['models']

    if not (args.test or args.continue_training):
        with open(os.path.join(args.save_dir, 'params.json'), 'w') as f:
            params = deepcopy(vars(args))
            params.pop('device')
            json.dump(params, f)
    train(args, policies_list)
