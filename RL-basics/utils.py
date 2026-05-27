from gymnasium.envs.registration import register

def setup():
    register(
        id="CliffWalking-RLSS-v0",
        entry_point="envs.CustomCliff:CliffWalkingEnv",
    )