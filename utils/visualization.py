import matplotlib.pyplot as plt
import torch

def show_image(image, label):
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()


def show_spike_frames(
    spikes: torch.Tensor,
    sample_index: int = 0,
    num_frames: int = 5,
) -> None:
    if spikes.ndim != 5:
        raise ValueError(f"Spike frame must be 5D tensor: [time, batch, channel, height, width]")

    num_frames = min(num_frames, spikes.shape[0])

    figure, axes = plt.subplots(
        1,
        num_frames,
        figsize=(12, 3),
    )
    if num_frames == 1:
        axes = [axes]

    for time_step in range(num_frames):
        frame = spikes[
            time_step,
            sample_index,
            0,
        ]
        axes[time_step].imshow(
            frame.cpu(),
            cmap='gray',
        )
        axes[time_step].set_title(
            f"Time: {time_step}",
        )
        axes[time_step].axis('off')

    plt.tight_layout()
    plt.show()
