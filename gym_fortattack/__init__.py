from gym.envs.registration import register
 
register(id='fortattack-v0', 
    entry_point='gym_fortattack.envs:FortAttackEnv', 
)
register(id='fortattack-v1', 
    entry_point='gym_fortattack.envs:FortAttackEnvV1', 
)