from gymnasium.envs.registration import register
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display
import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from matplotlib import animation
from abc import ABC, abstractmethod

random.seed(1)
np.random.seed(1)

UP    = 0
RIGHT = 1
DOWN  = 2
LEFT  = 3

def setup():
    register(
        id="CliffWalking-RLSS-v0",
        entry_point="envs.CustomCliff:CliffWalkingEnv",
    )

def animate_frames(frames, interval=200):
    """
    Anima una lista di frame RGB ottenuti da un env Gymnasium.

    Args:
        frames:   lista di array numpy (H, W, 3) ottenuti con env.render()
        interval: ms tra un frame e l'altro
    """
    h, w = frames[0].shape[:2]
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    img_plot = ax.imshow(frames[0])

    def update(frame_idx):
        img_plot.set_data(frames[frame_idx])
        return [img_plot]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        repeat=True
    )

    plt.close(fig)
    display(HTML(ani.to_jshtml(default_mode='loop')))

def plot_cumulative_reward(mean_cumulative_rewards: list, std_cumulative_rewards: list, n_runs: int):
    # Convert to numpy arrays for element-wise math operations
    mean_cumulative_rewards = np.array(mean_cumulative_rewards)
    std_cumulative_rewards = np.array(std_cumulative_rewards)

    # Calculate the 95% Confidence Interval margin of error
    # Formula: 1.96 * (Standard Deviation / sqrt(N))
    margin_of_error = 1.96 * (std_cumulative_rewards / np.sqrt(n_runs))

    plt.figure(figsize=(10, 6))
    episodes = range(len(mean_cumulative_rewards))
    
    # Plot the mean line
    plt.plot(episodes, mean_cumulative_rewards, label='Mean Cumulative Reward', color='blue')
    
    # Plot the 95% Confidence Interval shaded region
    plt.fill_between(
        episodes,
        mean_cumulative_rewards - margin_of_error,
        mean_cumulative_rewards + margin_of_error,
        alpha=0.3,
        color='blue',
        label='95% Confidence Interval'
    )
    
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Agent Performance Over Time (with 95% CI)')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
