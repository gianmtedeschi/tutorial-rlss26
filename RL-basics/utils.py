from gymnasium.envs.registration import register

def setup():
    register(
        id="CliffWalking-RLSS-v0",
        entry_point="envs.CustomCliff:CliffWalkingEnv",
    )

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML, display

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