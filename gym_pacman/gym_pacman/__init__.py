import gym
try:
	gym.envs.registry.spec('BerkeleyPacman-v0')
except:
	from gym.envs.registration import register
	register(
	    id='BerkeleyPacman-v0',
	    entry_point='gym_pacman.gym_pacman.envs:PacmanEnv',
	)
