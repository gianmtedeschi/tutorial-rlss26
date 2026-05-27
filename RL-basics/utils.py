from gymnasium.envs.registration import register

def setup():
    register(
        id="CliffWalking",
        entry_point="envs.CustomCliff:CliffWalkingEn",
    )