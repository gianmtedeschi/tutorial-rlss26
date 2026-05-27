from gymnasium.envs.registration import register

def setup():
    register(
        id="Cliff-v0",
        entry_point="envs.CustomCliff:CliffWalkingEn",
    )