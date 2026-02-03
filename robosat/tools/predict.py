import argparse
import os
import sys

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from tqdm import tqdm
from PIL import Image

import rootutils
rootutils.setup_root(__file__, indicator="robosat", pythonpath=True)

from robosat.transforms import ConvertImageMode, ImageToTensor
from robosat.datasets import BufferedSlippyMapDirectory
from robosat.unet import UNet
from robosat.config import load_config
from robosat.colors import continuous_palette_for_color

from torch.utils.data._utils.collate import default_collate

def collate(batch):
    images = default_collate([b[0] for b in batch])
    coords = [b[1] for b in batch]  # оставляем как список кортежей (x,y,z)
    return images, coords

def add_parser(subparser):
    parser = subparser.add_parser(
        "predict",
        help="predicts probability masks for slippy map tiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--batch_size", type=int, default=4, help="images per batch")
    parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    parser.add_argument("--overlap", type=int, default=32, help="tile pixel overlap to predict on")
    parser.add_argument("--tile_size", type=int, required=True, help="tile size for slippy map tiles")
    parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")
    parser.add_argument("tiles", type=str, help="directory to read slippy map image tiles from")
    parser.add_argument("probs", type=str, help="directory to save slippy map probability masks to")
    parser.add_argument("--model", type=str, required=True, help="path to model configuration file")
    parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")

    parser.set_defaults(func=main)


def main(args):
    model = load_config(args.model)
    dataset = load_config(args.dataset)

    cuda = model["common"]["cuda"]

    device = torch.device("cuda" if cuda else "cpu")

    def map_location(storage, _):
        return storage.cuda() if cuda else storage.cpu()

    if cuda and not torch.cuda.is_available():
        sys.exit("Error: CUDA requested but not available")

    num_classes = len(dataset["common"]["classes"])

    # https://github.com/pytorch/pytorch/issues/7178
    chkpt = torch.load(args.checkpoint, map_location=map_location)

    net = UNet(num_classes).to(device)
    net = nn.DataParallel(net)

    if cuda:
        torch.backends.cudnn.benchmark = True

    net.load_state_dict(chkpt["state_dict"])
    net.eval()

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    transform = Compose([ConvertImageMode(mode="RGB"), ImageToTensor(), Normalize(mean=mean, std=std)])

    directory = BufferedSlippyMapDirectory(args.tiles, transform=transform, size=args.tile_size, overlap=args.overlap)
    assert len(directory) > 0, "at least one tile in dataset"

    loader = DataLoader(directory, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)

    # don't track tensors with autograd during prediction
    with torch.no_grad():
        for batch_idx, (images, tiles) in enumerate(tqdm(loader, desc="Eval", unit="batch", ascii=True)):
            images = images.to(device)
            outputs = net(images)

            # manually compute segmentation mask class probabilities per pixel
            probs = nn.functional.softmax(outputs, dim=1).data.cpu().numpy()

            for tile, prob in zip(tiles, probs):
                # Дополнительная защита на случай неожиданных форм
                if isinstance(tile, (tuple, list)):
                    x, y, z = map(int, tile)
                else:
                    # tile может быть тензором; приведём к списку
                    if hasattr(tile, "detach"):
                        tlist = tile.detach().cpu().tolist()
                    else:
                        try:
                            tlist = list(tile)
                        except Exception:
                            raise ValueError(f"Unexpected tile type: {type(tile)}")
                    if len(tlist) == 3:
                        x, y, z = map(int, tlist)
                    else:
                        raise ValueError(f"Unexpected tile format in batch {batch_idx}: {tlist}")

                # we predicted on buffered tiles; now get back probs for original image
                prob = directory.unbuffer(prob)

                # Quantize the floating point probabilities in [0,1] to [0,255] and store
                # a single-channel `.png` file with a continuous color palette attached.

                assert prob.shape[0] == 2, "single channel requires binary model"
                assert np.allclose(np.sum(prob, axis=0), 1.), "single channel requires probabilities to sum up to one"
                foreground = prob[1:, :, :]

                anchors = np.linspace(0, 1, 256)
                quantized = np.digitize(foreground, anchors).astype(np.uint8)

                palette = continuous_palette_for_color("pink", 256)

                out = Image.fromarray(quantized.squeeze(), mode="P")
                out.putpalette(palette)

                os.makedirs(os.path.join(args.probs, str(z), str(x)), exist_ok=True)
                path = os.path.join(args.probs, str(z), str(x), str(y) + ".png")

                out.save(path, optimize=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict probability masks for slippy map tiles")
    subparsers = parser.add_subparsers(dest="command", help="available commands")

    # Добавляем парсер для команды predict
    predict_parser = subparsers.add_parser("predict", help="run prediction")

    # Копируем аргументы из функции add_parser
    predict_parser.add_argument("--batch_size", type=int, default=1, help="images per batch")
    predict_parser.add_argument("--checkpoint", type=str, required=True, help="model checkpoint to load")
    predict_parser.add_argument("--overlap", type=int, default=32, help="tile pixel overlap to predict on")
    predict_parser.add_argument("--tile_size", type=int, required=True, help="tile size for slippy map tiles")
    predict_parser.add_argument("--workers", type=int, default=0, help="number of workers pre-processing images")
    predict_parser.add_argument("tiles", type=str, help="directory to read slippy map image tiles from")
    predict_parser.add_argument("probs", type=str, help="directory to save slippy map probability masks to")
    predict_parser.add_argument("--model", type=str, required=True, help="path to model configuration file")
    predict_parser.add_argument("--dataset", type=str, required=True, help="path to dataset configuration file")

    args = parser.parse_args()

    if args.command == "predict":
        main(args)
    else:
        parser.print_help()