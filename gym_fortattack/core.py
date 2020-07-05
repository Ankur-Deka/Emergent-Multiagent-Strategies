import numpy as np
import time

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical orientation
        self.p_ang = None ## extra angle attribute, can shoot only towards angle, can move along any direction
        # physical velocity
        self.p_vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None   ## first two components for x,y. third component for rotation
        self.shoot = False   ## number greater than 0 means to shoot
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self, size = 0.05 ,color = None):
        # name 
        self.name = ''
        # properties:
        self.size = size
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = color
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0
        
    @property
    def mass(self):
        return self.initial_mass

## bullet is an entity
class Bullet(Entity):
    def __init__(self, bulletType):
        super(Bullet, self).__init__(size = 0.01)
        self.name = 'bullet'
        self.movable = True
        self.type = bulletType    # 'attacker' or 'guard' bullet
        self.color = np.array([0, 0.85, 0]) if self.type == 'guard' else np.array([0.85, 0.35, 0.35])


# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        ## number of bullets hit
        self.numHit = 0         # overall
        self.numWasHit = 0
        self.hit = False        # in last time
        self.wasHit = False
        ## shooting cone's radius and width (in radian)
        self.shootRad = 0.8 # default value (same for guards and attackers, can be changed in fortattack_env_v1)
        self.shootWin = np.pi/4
        self.alive = True   # alive/dead
        self.justDied = False   # helps compute reward for agent when it just died
        self.prevDist = None


# multi-agent world
class World(object):
    def __init__(self):
        ## lists of agents, entities and bullets (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.bullets = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 3  ## x, y, angle
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-10 # 1e-3
        ## wall positions
        self.wall_pos = [-1,1,-0.8,0.8] # (xmin, xmax) vertical and  (ymin,ymax) horizontal walls
    
    # return all alive agents
    @property
    def alive_agents(self):
        return [agent for agent in self.agents if agent.alive]

    # return all agents that are not adversaries
    @property
    def alive_guards(self):
        return [agent for agent in self.agents if (agent.alive and not agent.attacker)]

    # return all agents that are not adversaries
    @property
    def guards(self):
        return [agent for agent in self.agents if not agent.attacker]

    # return all adversarial agents
    @property
    def alive_attackers(self):
        return [agent for agent in self.agents if (agent.alive and agent.attacker)]


    # return all adversarial agents
    @property
    def attackers(self):
        return [agent for agent in self.agents if agent.attacker]

    # return all active in the world
    @property
    def active_entities(self):
        return [agent for agent in self.agents if agent.alive] + self.landmarks + self.bullets ## now bullets are also entities


    # return all entities in the world
    @property
    def entities(self):
        return [agent for agent in self.agents] + self.landmarks + self.bullets ## now bullets are also entities

    # return all agents controllable by external policies
    @property
    def alive_policy_agents(self):
        return [agent for agent in self.agents if (agent.alive and agent.action_callback is None)]


    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]


    # return all agents controlled by world scripts
    @property
    def active_scripted_agents(self):
        return [agent for agent in self.agents if (agent.alive and agent.action_callback is not None)]


    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # print('second step')
        # set actions for scripted agents
        ## IGNORE FOLLOWING: scripted agents are probably non-learning heuristic agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        ## The following is where are actions are actually applied for learning agents
        
        ## -------- apply effects of laser ------------- ##
        self.apply_laser_effect()      ## calling it effect as it might apply force, kill, etc...  
        
        # ------------- Calculate total physical (p_force) on each agent ------------- #
        p_force = [None] * len(self.active_entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        ## apply wall collision forces
        p_force = self.apply_wall_collision_force(p_force)
        # integrate physical state
        # calculates new state based on forces
        self.integrate_state(p_force)

        ## The following is for communication - IGNORE --------------- ##
        # update agent communication state
        for agent in self.alive_agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.alive_agents):
            force_dim = agent.action.u.shape[0]-1 ## 3rd u dimension is for rotation
            if agent.movable:
                noise = np.random.randn(*force_dim) * agent.u_noise if agent.u_noise else 0.0     ##
                p_force[i] = agent.action.u[:2] + noise           ##    
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.active_entities):
            for b,entity_b in enumerate(self.active_entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    ## apply wall collision force
    def apply_wall_collision_force(self, p_force):
        for a,agent in enumerate(self.alive_agents):
            f = self.get_wall_collision_force(agent)
            if(f is not None):
                if(p_force[a] is None): p_force[a] = 0.0
                p_force[a] = f + p_force[a] 
        return p_force

    def apply_laser_effect(self):
        ## reset bullet hitting states

        for i,entity in enumerate(self.alive_agents):
            entity.hit = False
            entity.wasHit = False

        for i,entity in enumerate(self.alive_agents):
            if entity.action.shoot:
                # let's use lasers - instantaneously hit entities within a cone (approximated by triangle)
                # compute cone
                A = self.get_tri_pts_arr(entity)

                for b,entity_b in enumerate(self.alive_agents):
                    if entity.attacker == entity_b.attacker:    # lasers don't affect agents of the same team
                        continue

                    if self.laser_hit(A, entity_b):
                        # print(entity.name, entity.attacker, entity_b.name, entity_b.attacker, 'final', entity.attacker == entity_b.attacker)
                        # # apply some force
                        # direc = entity_b.state.p_pos-entity.state.p_pos
                        # direc /= np.linalg.norm(direc)
                        # power = 10       # making it constant and much more than max accel
                        # force = direc*power
                        # p_force[b] += force
                        
                        # update hit states and counters
                        entity.hit = True
                        entity.numHit += 1
                        entity_b.wasHit = True
                        entity_b.numWasHit += 1
        
        # update just died state of dead agents
        for agent in self.agents:
            if not agent.alive:
                agent.justDied = False


        ## laser directly kills with one shot
        for agent in self.alive_agents:
            if agent.wasHit:
                agent.alive = False
                agent.justDied = True
                agent.color *= 0.5
                        
                if agent.attacker:
                    self.numAliveAttackers -= 1
                else:
                    self.numAliveGuards -= 1

    # integrate physical state
    def integrate_state(self, p_force):
        ## Step 0 - check bullet hitting agents
        # usedBullets = []
        # for i, bullet in enumerate(self.bullets):
        #     bulletHit = False           # flag to check if bullet hit
        #     opponents = self.attackers if bullet.type == 'guard' else self.guards   # bullet can kill the opponent team only 
        #     for opponent in opponents:
        #         if self.is_collision(bullet, opponent):
        #             bulletHit = True
        #             opponent.alive = False
        #             print('bullet laga')
        #             # time.sleep(2)
        #     if bulletHit:
        #         usedBullets.append(i)

        # usedBullets.reverse()
        # for i in usedBullets:
        #     self.bullets.pop(i)

        for i,entity in enumerate(self.active_entities):
            if not entity.movable: continue
            if not 'bullet' in entity.name:
                entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
                if (p_force[i] is not None):
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                    if speed > entity.max_speed:
                        entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
                ## simple model for rotation
                entity.state.p_ang += entity.action.u[2]%(2*np.pi)    ##
            
            entity.state.p_pos += entity.state.p_vel * self.dt

            # shoot?
            # if 'agent' in entity.name:  # only agents can shoot
            #     if entity.action.shoot: # and they should want to shoot
            #         # bulletType = 'attacker' if entity.attacker else 'guard'
            #         # newBullet = Bullet(bulletType = bulletType)
            #         # ang = entity.state.p_ang
            #         # direc = np.array([np.cos(ang), np.sin(ang)])
            #         # newBullet.state.p_pos = entity.state.p_pos + entity.size*direc
            #         # newBullet.state.p_vel = direc
            #         # self.bullets.append(newBullet)

            #         # let's use laser bullets - instantaneously hit entities within a cone (approximated by triangle)
            #         # compute cone
            #         A = self.get_tri_pts_arr(entity)

            #         opponents = self.guards if entity.attacker  else self.attackers   # bullet can kill the opponent team only 
            #         for opponent in opponents:
            #             if self.laser_hit(A, opponent):
            #                 entity.hit = True
            #                 entity.numHit += 1
            #                 opponent.wasHit = True
            #                 opponent.numWasHit += 1



    def svd_sol(self, A, b):
        U, sigma, Vt = np.linalg.svd(A)
        sigma[sigma<1e-10] = 0
        sigma_reci = [(1/s if s!=0 else 0) for s in sigma]
        sigma_reci = np.diag(sigma_reci)
        x = Vt.transpose().dot(sigma_reci).dot(U.transpose()).dot(b)
        return(x)

    def get_tri_pts_arr(self, agent):
        ang = agent.state.p_ang
        pt1 = agent.state.p_pos+agent.size*np.array([np.cos(ang), np.sin(ang)])
        pt2 = pt1 + agent.shootRad*np.array([np.cos(ang+agent.shootWin/2), np.sin(ang+agent.shootWin/2)])
        pt3 = pt1 + agent.shootRad*np.array([np.cos(ang-agent.shootWin/2), np.sin(ang-agent.shootWin/2)])
        
        A = np.array([[pt1[0], pt2[0], pt3[0]],
                      [pt1[1], pt2[1], pt3[1]],
                      [     1,      1,      1]])       
        return(A)

    def laser_hit(self, A, agent):
        b = np.array([[agent.state.p_pos[0]],[agent.state.p_pos[1]],[1]])
        x = self.svd_sol(A,b)
        if np.all(x>=0):
            return(True)
        else:
            return(False)

    def in_triangle(self, pts, pt):
        # 1- checks if agent is within the threshold of all opponents 
        # 2- checks if agent is in the convex hull of the agents
        posAdv = np.array([adv.state.p_pos for adv in adversaries])
        posAgent = np.array(agent.state.p_pos)
        dists = np.sqrt(np.sum(np.square(posAdv-posAgent), axis=1))
        maxDist = np.max(dists)
        
        rew = 0
        # first condition
        if maxDist<=th:
            A = np.concatenate((posAdv.transpose(), np.ones((1,3))), axis = 0)
            b = np.concatenate((posAgent, np.ones(1))).reshape(3,1)
            alpha = self.svd_sol(A,b)

            # 2nd condition
            if all(alpha>=0) and all(alpha<=1):
                # now we will give it some reward
                sig = th/3
                rew = np.exp(-(maxDist/sig)**2)
        
        return(rew)

    # def lineParams(pt, ang):
    #     a = -np.sin(ang)
    #     b = np.cos(ang)
    #     c = pt[0]*np.sin(ang)-pt[1]*np.cos(ang)
    #     return([a,b,c])

    # def line(x,lineParams):
    #     a,b,c = lineParams
    #     return(a*x+b*y+c)

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]

    # collision force with wall
    def get_wall_collision_force(self, entity):
        if not entity.collide:
            return([None]) # not a collider
        xmin,xmax,ymin,ymax = self.wall_pos
        x,y = entity.state.p_pos
        size = entity.size
        dists = np.array([x-size-xmin, xmax-x-size, y-size-ymin, ymax-y-size])

        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -dists/k)*k
        fx1,fx2,fy1,fy2 = self.contact_force * penetration
        force = [fx1-fx2,fy1-fy2]
        return force