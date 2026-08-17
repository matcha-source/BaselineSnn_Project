# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
from datasets.data_loader import create_dataloader
from configs.config import RANDOM_SEED, TIME_STEPS
from utils.reproducibility import set_seed
#from utils.visualization import show_image, show_spike_frames
from utils.encoding import poisson_encoder, calculate_spike_rate
from models.snn_model import BaselineSNN
from utils.decoding import decode_spike_count

set_seed(RANDOM_SEED)

def main() -> None:

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_loader = create_dataloader(train=True)
    #test_loader = create_dataloader(train=False)

    images, labels = next(iter(train_loader))
    #test_images, test_labels = next(iter(test_loader))

    images = images.to(device)
    labels = labels.to(device)

    spikes = poisson_encoder(
        images=images,
        time_steps=TIME_STEPS,
    )

    model = BaselineSNN().to(device)
    output_spikes = model(spikes)

    predictions = decode_spike_count(output_spikes)

    spike_rate = calculate_spike_rate(spikes)

    #show_image(images[0], labels[0].item())

    # show_spike_frames(
    #     spikes=spikes,
    #     sample_index=0,
    #     num_frames=5,
    # )

    accuracy = (predictions == labels).float().mean()

    print(f" Input image shape: {images.shape}")
    print(f" labels shape: {labels.shape}")
    print(f" Input spike shape: {spikes.shape}")
    print(f" Output spike shape: {output_spikes.shape}")
    print(f"Average spike rate: {spike_rate}")

    print(f"Predictions: {predictions[:10]}")
    print(f"Labels: {labels[:10]}")
    print(f"Initial accuracy: {accuracy.item() * 100: .2f}%")
    #print(test_images.shape)

    # print("=" * 50)
    # print("PyTorch Version :", torch.__version__)
    # print("CUDA Available :", torch.cuda.is_available())
    # print("CUDA Version   :", torch.version.cuda)
    # print("GPU Count      :", torch.cuda.device_count())
    #
    # if torch.cuda.is_available():
    #     print("GPU Name       :", torch.cuda.get_device_name(0))
    # else:
    #     print("Running on CPU")
    #
    # print("=" * 50)

#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print('Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# this is just to commit


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Sprint_hi('PyCharm')
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
