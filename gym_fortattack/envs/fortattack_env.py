import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_fortattack.core import World, Agent, EntityState
import numpy as np
import time

class FortAttackEnv(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self):
        # environment will have guards(green) and attackers(red)
        # red bullets - can hurt green agents, vice versa
        # single hit - if hit once agent dies
        self.world = World() 

        self.fortDim = 0.4
        self.numGuards = 2  # initial number of guards, attackers and bullets  
        self.numAttackers = 2
        self.numBullets = 0
        self.numAgents = self.numGuards + self.numAttackers
        landmarks = [] # as of now no obstacles, landmarks 

        self.world.agents = [Agent() for i in range(self.numAgents)] 
        
        for i, agent in enumerate(self.world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.attacker = False if i < self.numGuards else True
            agent.accel = 3  ## guards and attackers have same speed and accel
            agent.max_speed = 1   ## used in integrate_state() inside core.py. slowing down so that bullet can move fast and still it doesn't seem that the bullet is skipping steps 
            agent.max_rot = 0.17 ## approx 10 degree

        self.viewers = [None]
        self.render_geoms = None
        self.shared_viewer = True
        self.reset_world()        

    def reset_world(self):
        # light green for guards and light red for attackers
        self.world.bullets = [] ##
        for i, agent in enumerate(self.world.agents):
            agent.color = np.array([0, 1, 0]) if not agent.attacker else np.array([1, 0, 0])
            agent.state.p_vel = np.zeros(self.world.dim_p-1)    ##
            agent.state.c = np.zeros(self.world.dim_c)
            agent.state.p_ang = np.pi/2 if agent.attacker else 3*np.pi/2 

            # now we will set the initial positions
            # attackers start from far away
            if agent.attacker:
                agent.state.p_pos = np.concatenate((np.random.uniform(-1,1,1), np.random.uniform(-1,-0.8,1)))

            # guards start near the door
            else:
                agent.state.p_pos = np.concatenate((np.random.uniform(-0.8*self.fortDim/2,0.8*self.fortDim/2,1), np.random.uniform(0.8,1,1)))

            agent.numHit = 0         # overall in one episode
            agent.numWasHit = 0
            agent.hit = False        # in last time step
            agent.wasHit = False
        
        # random properties for landmarks
        for i, landmark in enumerate(self.world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        
        
        for i, landmark in enumerate(self.world.landmarks):
            if not landmark.boundary:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, self.world.dim_p)
                landmark.state.p_vel = np.zeros(self.world.dim_p)

    
 
    
    # # return all agents that are not adversaries
    # def guards(self):
    #     return [agent for agent in self.world.agents if not agent.attacker]

    # # return all adversarial agents
    # def attackers(self):
    #     return [agent for agent in self.world.agents if agent.attacker]



    def reward(self, agent):
        main_reward = self.attacker_reward(agent) if agent.attacker else self.guard_reward(agent)
        return main_reward

    def attacker_reward(self, agent):
        rew0, rew1, rew2, rew3, rew4 = 0,0,0,0,0
        doorLoc = np.array([0,1])

        # Attackers get reward for being close to the door
        distToDoor = np.sqrt(np.sum(np.square(agent.state.p_pos-doorLoc)))
        rew0 = -0.1*distToDoor

        # Attackers get very high reward for reaching the door
        th = 0.3
        sig = th/3
        rew1 = 10*np.exp(-(distToDoor/sig)**2)

        # attacker gets -ve reward for using laser
        if agent.action.shoot:
            rew2 = -0.1

        # gets positive reward for hitting a guard??
        if agent.hit:
            rew3 = +0.1

        # gets negative reward for being hit by a guard
        if agent.wasHit:
            rew4 = -1

        rew = rew0+rew1+rew2+rew3+rew4
        # print('attacker_reward', rew1, rew2, rew3, rew4, rew)
        return rew

    def guard_reward(self, agent):
        # guards get reward for keeping all attacker away
        rew0, rew1, rew2, rew3, rew4, rew5, rew6 = 0,0,0,0,0,0,0
        doorLoc = np.array([0,1])
        
        # high negative reward for leaving the fort
        selfDistToDoor = np.sqrt(np.sum(np.square(agent.state.p_pos-doorLoc)))
        if selfDistToDoor>0.3:
            rew0 = -2

        # negative reward for going away from the fort
        rew1 = -0.1*selfDistToDoor

        # negative reward if attacker comes closer
        # make it exponential
        minDistToDoor = np.min([np.sqrt(np.sum(np.square(attacker.state.p_pos-doorLoc))) for attacker in self.world.attackers])
        protectionRadius = 0.5
        sig = protectionRadius/3
        rew2 = -10*np.exp(-(minDistToDoor/sig)**2)

        # high negative reward if attacker reaches the fort
        if minDistToDoor<0.3:
            rew3 = -10

        # guard gets negative reward for using laser
        if agent.action.shoot:
            rew4 = -0.1

        # gets reward for hitting an attacker
        if agent.hit:
            rew5 = 0.1

        # guard gets -ve reward for being hit by laser
        if agent.wasHit:
            rew6 = -1

        rew = rew0+rew1+rew2+rew3+rew4+rew5+rew6
        # print('guard_reward', rew1, rew2, rew3, rew4, rew)
        return rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        orien = [np.array([np.cos(agent.state.p_ang), np.sin(agent.state.p_ang)])]

        comm = []
        other_pos = []
        other_vel = []
        other_orien = []
        other_shoot = [] 
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            ## if not other.attacker:
            other_vel.append(other.state.p_vel)
            rel_ang = other.state.p_ang - agent.state.p_ang
            other_orien.append(np.array([np.cos(rel_ang), np.sin(rel_ang)]))
            other_shoot.append(np.array([other.action.shoot]).astype(float))
        # print('obs')
        # print([agent.state.p_pos])
        # print([agent.state.p_vel])
        # print(orien)
        # print(entity_pos)
        # print(other_pos)
        # print(other_vel)
        # print(other_orien)
        # print(other_shoot)
        # print(len(other_orien), other_shoot.shape)
        # print(np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + orien + entity_pos + other_pos + other_vel + other_orien + other_shoot))
        return np.concatenate([agent.state.p_pos] + [agent.state.p_vel] + orien + entity_pos + other_pos + other_vel + other_orien + other_shoot)
