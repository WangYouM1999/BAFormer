import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# for f in file:
#     target_path = os.path.join(path, f)
#     img = Image.open(target_path)
#     img = np.array(img)
#     print(img)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 128]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 255]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, default='./config/mapcup/unetformer.py',
        help="Path to  config")
    arg("-o", "--output_path", type=Path, default='fig_results/mapcup/unetformer',
        help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("--rgb", default='--rgb', help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    seed_everything(42)

    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda(config.gpus[0])
    # 评估代码
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90(angles=[90]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False)
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    predict_dataset = config.predict_dataset

    with torch.no_grad():
        predict_loader = DataLoader(
            predict_dataset,
            batch_size=2,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(predict_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda(config.gpus[0]))

            image_ids = input["img_id"]
            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
