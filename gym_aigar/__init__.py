import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='AigarPellet-v0',
    entry_point='gym_aigar.envs:AigarPelletEnv',
)

register(
    id='AigarGreedy-v0',
    entry_point='gym_aigar.envs:AigarGreedyEnv',
)



