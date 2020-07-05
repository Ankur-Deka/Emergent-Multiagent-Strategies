import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_fortattack.core import World, Agent, EntityState
import numpy as np
import time

class FortAttackEnvV1(gym.Env):  
    metadata = {'render.modes': ['human']}   
    def __init__(self):
        # environment will have guards(green) and attackers(red)
        # red bullets - can hurt green agents, vice versa
        # single hit - if hit once agent dies
        self.world = World() 

        self.world.fortDim = 0.15   # radius
        self.world.doorLoc = np.array([0,0.8])
        self.world.numGuards = 5  # initial number of guards, attackers and bullets  
        self.world.numAttackers = 5
        self.world.numBullets = 0
        self.world.numAgents = self.world.numGuards + self.world.numAttackers
        self.world.numAliveGuards, self.world.numAliveAttackers, self.world.numAliveAgents = self.world.numGuards, self.world.numAttackers, self.world.numAgents
        self.world.atttacker_reached = False     ## did any attacker succeed to reach the gate?
        landmarks = [] # as of now no obstacles, landmarks 

        self.world.agents = [Agent() for i in range(self.world.numAgents)]  # first we have the guards and then we have the attackers
        for i, agent in enumerate(self.world.agents):
            agent.name = 'agent %d' % (i+1)
            agent.collide = True
            agent.silent = True
            agent.attacker = False if i < self.world.numGuards else True
            # agent.shootRad = 0.8 if i<self.world.numGuards else 0.6
            agent.accel = 3  ## guards and attackers have same speed and accel
            agent.max_speed = 3   ## used in integrate_state() inside core.py. slowing down so that bullet can move fast and still it doesn't seem that the bullet is skipping steps 
            agent.max_rot = 0.17 ## approx 10 degree

        self.viewers = [None]
        self.render_geoms = None
        self.shared_viewer = True
        self.world.time_step = 0
        self.world.max_time_steps = None #  set inside malib/environments/fortattack
        self.world.vizDead = False         # whether to visualize the dead agents
        self.world.vizAttn = True        # whether to visualize attentions
        self.world.gameResult = np.array([0,0,0]) #  [all attackers dead, max time steps, attacker reached fort]
        self.reset_world()        

    def reset_world(self):
        # light green for guards and light red for attackers
        self.world.time_step = 0
        self.world.bullets = [] ##
        self.world.numAliveAttackers = self.world.numAttackers
        self.world.numAliveGuards = self.world.numGuards
        self.world.numAliveAgents = self.world.numAgents
        self.world.gameResult[:] = 0
        for i, agent in enumerate(self.world.agents):
            agent.alive = True
            agent.color = np.array([0.0, 1.0, 0.0]) if not agent.attacker else np.array([1.0, 0.0, 0.0])
            agent.state.p_vel = np.zeros(self.world.dim_p-1)    ##
            agent.state.c = np.zeros(self.world.dim_c)
            agent.state.p_ang = np.pi/2 if agent.attacker else 3*np.pi/2 

            xMin, xMax, yMin, yMax = self.world.wall_pos
            # now we will set the initial positions
            # attackers start from far away
            if agent.attacker:
                agent.state.p_pos = np.concatenate((np.random.uniform(xMin,xMax,1), np.random.uniform(yMin,0.8*yMin,1)))

            # guards start near the door
            else:
                agent.state.p_pos = np.concatenate((np.random.uniform(-0.8*self.world.fortDim/2,0.8*self.world.fortDim/2,1), np.random.uniform(0.8*yMax,yMax,1)))

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

    def reward(self, agent):
        if agent.alive or agent.justDied:
            main_reward = self.attacker_reward(agent) if agent.attacker else self.guard_reward(agent)
        else:
            main_reward = 0
        return main_reward

    def attacker_reward(self, agent):
        rew0, rew1, rew2, rew3, rew4, rew5 = 0,0,0,0,0,0
        
        # dead agents are not getting reward just when they are dead
        # # Attackers get reward for being close to the door
        distToDoor = np.sqrt(np.sum(np.square(agent.state.p_pos-self.world.doorLoc)))
        if agent.prevDist is not None:
            rew0 = 2*(agent.prevDist - distToDoor)
            # print('rew0', rew0, 'fortattack_env_v1.py')
        # Attackers get very high reward for reaching the door
        th = self.world.fortDim
        if distToDoor<th:
            rew1 = 10
            self.world.atttacker_reached = True

        # attacker gets -ve reward for using laser
        if agent.action.shoot:
            rew2 = -1

        # gets positive reward for hitting a guard??
        if agent.hit:
            rew3 = +3

        # gets negative reward for being hit by a guard
        if agent.wasHit:
            rew4 = -3

        # high negative reward if all attackers are dead
        if self.world.numAliveAttackers == 0:
            rew5 = -10

        rew = rew0+rew1+rew2+rew3+rew4+rew5
        agent.prevDist = distToDoor.copy()
        # print('attacker_reward', rew1, rew2, rew3, rew4, rew)
        return rew

    def guard_reward(self, agent):
        # guards get reward for keeping all attacker away
        rew0, rew1, rew2, rew3, rew4, rew5, rew6, rew7, rew8 = 0,0,0,0,0,0,0,0,0
        
        # # high negative reward for leaving the fort
        selfDistToDoor = np.sqrt(np.sum(np.square(agent.state.p_pos-self.world.doorLoc)))
        # if selfDistToDoor>0.3:
        #     rew0 = -2

        # negative reward for going away from the fort
        if agent.prevDist is not None:
            if selfDistToDoor>0.3 and agent.prevDist<=0.3:
                rew0 = -1
            elif selfDistToDoor<=0.3 and agent.prevDist>0.3:
                rew0 = 1

            # rew1 = 20*(agent.prevDist - selfDistToDoor)

            # print('rew1', rew1, 'fortattack_env_v1.py')
        # rew1 = -0.1*selfDistToDoor

        # negative reward if attacker comes closer
        # make it exponential
        if self.world.numAliveAttackers != 0:
            minDistToDoor = np.min([np.sqrt(np.sum(np.square(attacker.state.p_pos-self.world.doorLoc))) for attacker in self.world.alive_attackers])
            # protectionRadius = 0.5
            # sig = protectionRadius/3
            # rew2 = -10*np.exp(-(minDistToDoor/sig)**2)

            # high negative reward if attacker reaches the fort
            th = self.world.fortDim
            if minDistToDoor<th:
                rew3 = -10

        # guard gets negative reward for using laser
        if agent.action.shoot:
            rew4 = -0.1

        # gets reward for hitting an attacker
        if agent.hit:
            rew5 = 3


        # guard gets -ve reward for being hit by laser
        if agent.wasHit:
            rew6 = -3

        # high positive reward if all attackers are dead
        if self.world.numAliveAttackers == 0:
            # if agent.hit:
            rew7 = 10

        # # small positive reward at every time step
        # rew8 = 10/self.world.max_time_steps

        rew = rew0+rew1+rew2+rew3+rew4+rew5+rew6+rew7+rew8
        # print('guard_reward', rew1, rew2, rew3, rew4, rew)
        agent.prevDist = selfDistToDoor.copy()
        return rew


    def observation(self, agent, world):
        # print('agent name', agent.name)
        # if not agent.alive:
        #     return(np.array([]))
        # else:
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        
        orien = [[agent.state.p_ang]]
        # [np.array([np.cos(agent.state.p_ang), np.sin(agent.state.p_ang)])]
        
        # communication of all other agents
        # comm = []
        # other_pos = []
        # other_vel = []
        # other_orien = []
        # other_shoot = [] 
        # for other in world.agents:
        #     if other is agent: continue
        #     comm.append(other.state.c)
        #     other_pos.append(other.state.p_pos - agent.state.p_pos)
        #     ## if not other.attacker:
        #     other_vel.append(other.state.p_vel)
        #     rel_ang = other.state.p_ang - agent.state.p_ang
        #     other_orien.append(np.array([np.cos(rel_ang), np.sin(rel_ang)]))
        #     other_shoot.append(np.array([other.action.shoot]).astype(float))
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
        # [[int(agent.alive)]]+
        
        # print(np.shape(np.concatenate([[int(agent.alive)]]+[agent.state.p_pos] + [agent.state.p_vel] + orien + entity_pos + other_pos + other_vel + other_orien + other_shoot)))
    
        # return np.concatenate([[int(agent.alive)]]+[agent.state.p_pos] + [agent.state.p_vel] + orien + entity_pos + other_pos + other_vel + other_orien + other_shoot)
        
        ## self.alive, self.pos, self.orien, self.vel
        return(np.concatenate([[agent.alive]]+[agent.state.p_pos] + orien + [agent.state.p_vel] +entity_pos))

    