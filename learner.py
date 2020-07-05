import numpy as np
import torch
from rlcore.algo import JointPPO
from rlagent import Neo
from mpnn import MPNN
from utils import make_multiagent_env
import time


def setup_master(args, env=None, return_env=False):
    if env is None:
        env = make_multiagent_env(args.env_name, num_agents=args.num_agents, dist_threshold=args.dist_threshold, 
                                  arena_size=args.arena_size, identity_size=args.identity_size, num_steps = args.num_env_steps)
    policy1 = None
    policy2 = None
    team1 = []  ## List of Neo objects, one Neo object per agent, teams_list = [team1, teams2] 
    team2 = []

    num_adversary = 0
    num_friendly = 0
    for i,agent in enumerate(env.world.policy_agents):
        if agent.attacker:
            num_adversary += 1
        else:
            num_friendly += 1

    # share a common policy in a team
    # print('action space')
    # print(env.action_space)
    # time.sleep(5)
    # print(env.action_spaces)
    # print(type(env.action_spaces))
    # time.sleep(5)
    action_space = env.action_space[i]      ##* why on earth would you put i???
    entity_mp = args.entity_mp
    if args.env_name == 'simple_spread':
        num_entities = args.num_agents
    elif args.env_name == 'simple_formation':
        num_entities = 1    # probably should be args.num_agents
    elif args.env_name == 'simple_line':
        num_entities = 2
    elif args.env_name == 'fortattack-v1':
        num_entities = 0    ## obstacles/landmarks
    else:
        raise NotImplementedError('Unknown environment, define entity_mp for this!')


    if entity_mp:
        pol_obs_dim = env.observation_space[0].shape[0] - 2*num_entities    ## why is there i?? originally it was observation_space[i]
    else:
        pol_obs_dim = env.observation_space[0].shape[0]     ##* ensure 27 is correct 
        
    
    # index at which agent's position is present in its observation
    pos_index = args.identity_size + 2  ##* don't know why it's a const and +2

    for i, agent in enumerate(env.world.policy_agents):
        obs_dim = env.observation_space[i].shape[0]

        if not agent.attacker:   # first we have the guards and then the attackers
            if policy1 is None:
                policy1 = MPNN(input_size=pol_obs_dim, num_agents=num_friendly, num_opp_agents=num_adversary,num_entities=num_entities,action_space=env.action_spaces[i],
                               pos_index=pos_index, mask_dist=args.mask_dist, entity_mp=entity_mp, policy_layers=1).to(args.device)
            team1.append(Neo(args,policy1,(obs_dim,),action_space))     ## Neo adds additional features to policy such as loading model, update_rollout. Neo.actor_critic is the policy and is the same object instance within a team
        else:
            if policy2 is None:
                policy2 = MPNN(input_size=pol_obs_dim,num_agents=num_adversary, num_opp_agents=num_friendly, num_entities=num_entities,action_space=env.action_spaces[i],
                               pos_index=pos_index, mask_dist=args.mask_dist,entity_mp=entity_mp).to(args.device)
            team2.append(Neo(args,policy2,(obs_dim,),action_space))
    master = Learner(args, [team1, team2], [policy1, policy2], env)

    if args.continue_training:
        print("Loading pretrained model")
        master.load_models(torch.load(args.load_dir)['models'])
    ##* till now N2check pol_obs_dim, pol_obs_dim, MPNN and Neo. 4th Nov
    if return_env:
        return master, env
    return master


class Learner(object):
    # supports centralized training of agents in a team
    def __init__(self, args, teams_list, policies_list, env):
        self.teams_list = [x for x in teams_list if len(x)!=0]
        self.all_agents = [agent for team in teams_list for agent in team]
        self.policies_list = [x for x in policies_list if x is not None]    ## if there are 2 teams then policies_list will have 2 elememts, [MPNN, MPNN], first for guards and then for attackers
        self.trainers_list = [JointPPO(policy, args.clip_param, args.ppo_epoch, args.num_mini_batch, args.value_loss_coef,
                                       args.entropy_coef, lr=args.lr, max_grad_norm=args.max_grad_norm,
                                       use_clipped_value_loss=args.clipped_value_loss) for policy in self.policies_list]        # one JointPPO per MPNN policy, joint because takes care of all agents in the same team
        self.device = args.device
        self.env = env
        n = len(self.all_agents)
        self.masks = torch.zeros((args.num_processes, n), dtype = torch.float)
        if not args.no_cuda:
            self.masks = self.masks.cuda()    ## initially all agents are alive
        self.attacker_load_dir = args.attacker_load_dir ## these 2 lines if we want to train guards only (againt pre-trained attackers)
        self.attacker_ckpts = args.attacker_ckpts

    @property
    def all_policies(self):
        return [agent.actor_critic.state_dict() for agent in self.all_agents]

    @property
    def team_attn(self):
        return self.policies_list[0].attn_mat

    def initialize_obs(self, obs):
        # obs - num_processes x num_agents x obs_dim
        for i, agent in enumerate(self.all_agents):
            agent.initialize_obs(torch.from_numpy(obs[i]).float().to(self.device))
            agent.rollouts.to(self.device)

    def initialize_new_episode(self, step, obs, masks):
        # obs - num_processes x num_agents x obs_dim
        for i, agent in enumerate(self.all_agents):
            agent.initialize_new_episode(step, torch.from_numpy(obs[i]).float().to(self.device), masks[i])
            agent.rollouts.to(self.device)

    def sample_attacker(self):
        ckpt = np.random.choice(self.attacker_ckpts)
        self.select_attacker(ckpt)
        # path = self.attacker_load_dir + '/ep' + str(ckpt)+'.pt'
        # checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        # policies_list = checkpoint['models']    
        # attacker_list = self.teams_list[1]
        # numGuards = self.env.world.numGuards
        # for agent, policy in zip(attacker_list, policies_list[numGuards:]):
        #     agent.load_model(policy)
        # print('sampled attacker ckpt', ckpt, 'learner.py')

    def select_attacker(self, ckpt): # select specific attacker strategy
        path = self.attacker_load_dir + '/ep' + str(ckpt)+'.pt'
        print('path', path, 'learner.py')
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        policies_list = checkpoint['models']    
        attacker_list = self.teams_list[1]
        numGuards = self.env.world.numGuards
        for agent, policy in zip(attacker_list, policies_list[numGuards:]):
            agent.load_model(policy)
        print('sampled attacker ckpt', ckpt, 'learner.py')


    def act(self, step, all_masks):
        ## uses policies to choose actiuons
        ##* need to consider dead/alive here, masks denotes alive/dead 
        actions_list = []
        for i, team, policy in zip(np.arange(len(self.all_agents)), self.teams_list, self.policies_list):
            # print('TEAM', i, '\n')
            # concatenate all inputs
            all_obs = torch.cat([agent.rollouts.obs[step] for agent in team if agent.alive])   ##rollouts stores all the data for that episode, with he help of initialize_obs and update_rollout
            all_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[step] for agent in team if agent.alive]) # this alive is not doing anything AFAI think
            all_opp_obs = torch.cat([agent.rollouts.obs[step] for agent in self.teams_list[1-i] if agent.alive])
            # all_masks = torch.cat([agent.rollouts.masks[step] for agent in team if agent.alive])    
            # print('masks', masks) 
            # print('learner.py')
            # actual policy act            
            # i
            # print('masks', all_masks)
            masks = None ## to mask out dead agents during observation, not implemented yet
            props = policy.act(all_obs, all_hidden, all_opp_obs, masks, deterministic=False) # a single forward pass 

            # split all outputs
            n = len(team)
            all_value, all_action, all_action_log_prob, all_states = [torch.chunk(x, n) for x in props]
            for j in range(n):
                team[j].value = all_value[j]
                team[j].action = all_action[j]
                team[j].action_log_prob = all_action_log_prob[j]
                team[j].states = all_states[j]
                actions_list.append(all_action[j].cpu().numpy())
        attn_list = [[policy.attn_mat, policy.opp_attn_mat] for policy in self.policies_list]
        return actions_list, attn_list


    def update(self, train_guards_only = False):
        return_vals = []
        to_train_list = [self.trainers_list[0]] if  train_guards_only else self.trainers_list 
        # use joint ppo for training each team
        print('training, learner.py')
        for i, trainer in enumerate(to_train_list):
            print('trainier', i, 'learner.py')
            rollouts_list = [agent.rollouts for agent in self.teams_list[i]]
            opp_rollouts_list = [agent.rollouts for agent in self.teams_list[1-i]]
            vals = trainer.update(rollouts_list, opp_rollouts_list)
            return_vals.append([np.array(vals)]*len(rollouts_list))

        
        return np.stack([x for v in return_vals for x in v]).reshape(-1,3)

    ## computes the return (reward + gamam*value) for all states in the order T, T-1, ...., 2, 1
    def wrap_horizon(self, end_pts):
    ## end_pts are time end points of individual episodes since different episodes are concatenated
        start_pt = 0
        for end_pt in end_pts:
            for i, team, policy in zip(np.arange(len(self.all_agents)), self.teams_list,self.policies_list):  
                last_obs = torch.cat([agent.rollouts.obs[end_pt] for agent in team])
                last_opp_obs = torch.cat([agent.rollouts.obs[end_pt] for agent in self.teams_list[1-i]])
                last_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[end_pt] for agent in team])
                last_masks = torch.cat([agent.rollouts.masks[end_pt] for agent in team])
                
                with torch.no_grad():
                    next_value = policy.get_value(last_obs, last_hidden, last_opp_obs, last_masks) # tensor containing next values for each agent in the team

                all_value = torch.chunk(next_value,len(team)) ## splits next_value
                # print('next_value', next_value)
                # print('all_value', all_value)
                # print('len(team)', len(team))
                # print('learner.py')
                for i in range(len(team)):
                    team[i].wrap_horizon(all_value[i], start_pt, end_pt)
            start_pt = end_pt+1

        # for i, team, policy in zip(np.arange(len(self.all_agents)), self.teams_list,self.policies_list):  
        #     last_obs = torch.cat([agent.rollouts.obs[-1] for agent in team])
        #     last_opp_obs = torch.cat([agent.rollouts.obs[-1] for agent in self.teams_list[1-i]])
        #     last_hidden = torch.cat([agent.rollouts.recurrent_hidden_states[-1] for agent in team])
        #     last_masks = torch.cat([agent.rollouts.masks[-1] for agent in team])
            
        #     with torch.no_grad():
        #         next_value = policy.get_value(last_obs, last_hidden, last_opp_obs, last_masks) # tensor containing next values for each agent in the team

        #     all_value = torch.chunk(next_value,len(team)) ## splits next_value
        #     # print('next_value', next_value)
        #     # print('all_value', all_value)
        #     # print('len(team)', len(team))
        #     # print('learner.py')
        #     for i in range(len(team)):
        #         team[i].wrap_horizon(all_value[i], end_pts)

    def before_update(self):
        for agent in self.all_agents:
            agent.before_update()

    def after_update(self):
        for agent in self.all_agents:
            agent.after_update()

    ## adds current observations to the storage
    def update_rollout(self, obs, reward, masks):
        obs_t = torch.from_numpy(obs).float().to(self.device)
        for i, agent in enumerate(self.all_agents):
            agent_obs = obs_t[i, :]
            agent.update_rollout(agent_obs, reward[i], masks[i])

    def load_models(self, policies_list):
        for agent, policy in zip(self.all_agents, policies_list):
            agent.load_model(policy)

    def eval_act(self, obs, recurrent_hidden_states, mask):
        # used only while evaluating policies. Assuming that agents are in order of team!
        ## obs has agents of both teams
        # print('eval_act()')
        obs1 = []
        obs2 = []
        # all_obs = []
        for i in range(len(obs)):
            agent = self.env.world.policy_agents[i]
            # this thing needs correction 'adversary' -> 'attacker'?
            if hasattr(agent, 'attacker') and not agent.attacker:
                obs1.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
            else:
                obs2.append(torch.as_tensor(obs[i],dtype=torch.float,device=self.device).view(1,-1))
        # if len(obs1)!=0:
        #     all_obs.append(obs1)
        # if len(obs2)!=0:
        #     all_obs.append(obs2)

        all_obs = [obs1, obs2]
        # print(all_obs)

        actions = []
        for i,team,policy,obs in zip(np.arange(len(self.teams_list)), self.teams_list,self.policies_list,all_obs):
            if len(obs)!=0:
                oppObs = torch.cat(all_obs[1-i]).to(self.device)
                _,action,_,_ = policy.act(torch.cat(obs).to(self.device),None,oppObs,None,deterministic=True)
                actions.append(action.squeeze(1).cpu().numpy())

        return np.hstack(actions)

    def set_eval_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.eval()

    def set_train_mode(self):
        for agent in self.all_agents:
            agent.actor_critic.train()
