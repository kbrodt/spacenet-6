import argparse
from pathlib import Path

import numpy as np
import torch
import tqdm
import pandas as pd
import PIL.Image as Image

from dataset import dev_transform
from utils import read_img_ski
from mask import mask_to_poly_geojson


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to data')
    parser.add_argument('--exp', type=str, required=True,
                        help='Path to models checkpoints in jit format')
    parser.add_argument('--to-save', type=str, required=True,
                        help='Folder path to save test masks')
    parser.add_argument('--submit-path', type=str, required=True)
    
    parser.add_argument('--n-parts', type=int, default=1)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--tta', type=int, default=1)
    parser.add_argument('--watershed', action='store_true')
    parser.add_argument('--thresh', type=float, default=None)
    
    parser.add_argument('--res', type=int, default=900,
                        help='Image resolution')
    parser.add_argument('--batch-size', type=int, default=32)

    return parser.parse_args()


def get_input(pred_msk):
    pred_msk = pred_msk[0] * (1 - pred_msk[2])
    pred_msk = 1 * (pred_msk > 0.45)

    pred_msk = pred_msk.astype(np.uint8)

    return pred_msk


class DS(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        path = self.df.iloc[index].image
        
        img = read_img_ski(path).astype('float32')/255
        mask = np.zeros_like(img)[..., [0]]

        return dev_transform(img, mask)[0], path.split('/')[-1]


def collate(x):
    x, y = list(zip(*x))

    return torch.stack(x), y


def main():
    args = parse_args()
    print(args)
    
    test_anns = pd.read_csv(args.csv)

    n = len(test_anns)
    k = n//args.n_parts
    start = args.part*k
    end = k*(args.part + 1) if args.part + 1 != args.n_parts else n
    test_anns = test_anns.iloc[start:end].copy()
    print(f'test size: {len(test_anns)}')
    
    batch_size = args.batch_size
    ds = DS(test_anns)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=batch_size,
        shuffle=False,
        collate_fn=collate,
        pin_memory=True,
    )

    models = [
        torch.jit.load(str(p)).cuda().eval()
        for p in Path(args.exp).rglob('*.pt')
    ]
    print(f'#models: {len(models)}')
    n_models = len(models)
    watershed = args.watershed
    n_augs = n_models * args.tta
    
    to_save = Path(args.to_save)
    print(f'save path: {to_save}')
    if not to_save.exists():
        to_save.mkdir(parents=True)

    def get_submit():
        masks = torch.zeros((batch_size, 4 if args.thresh is None else 3 , args.res, args.res), dtype=torch.float32, device='cuda')
        submit = pd.DataFrame()
        with torch.no_grad():
            with tqdm.tqdm(loader, mininterval=2) as pbar:
                for img, anns in pbar:
                    bs = img.size(0)
                    img = img.cuda()

                    masks.zero_()
                    for model in models:
                        mask = model(img)
                        if args.thresh is not None:
                            masks[:bs] += torch.sigmoid(mask)[..., 62:-62, 62:-62]
                        else:
                            masks[:bs] += torch.softmax(mask, dim=1)[..., 62:-62, 62:-62]

                        # vertical flip
                        if args.tta > 1:
                            mask = model(torch.flip(img, dims=[-1]))
                            if args.thresh is not None:
                                masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-1])[..., 62:-62, 62:-62]
                            else:
                                masks[:bs] += torch.flip(torch.softmax(mask, dim=1), dims=[-1])[..., 62:-62, 62:-62]

                        # horizontal flip
                        if args.tta > 2:
                            mask = model(torch.flip(img, dims=[-2]))
                            if args.thresh is not None:
                                masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-2])[..., 62:-62, 62:-62]
                            else:
                                masks[:bs] += torch.flip(torch.softmax(mask, dim=1), dims=[-2])[..., 62:-62, 62:-62]

                        if args.tta > 3:
                            # vertical + horizontal flip
                            mask = model(torch.flip(img, dims=[-1, -2]))
                            if args.thresh is not None:
                                masks[:bs] += torch.flip(torch.sigmoid(mask), dims=[-1, -2])[..., 62:-62, 62:-62]
                            else:
                                masks[:bs] += torch.flip(torch.softmax(mask, dim=1), dims=[-1, -2])[..., 62:-62, 62:-62]

                    masks /= n_augs
                    for mask, annotation in zip(masks, anns):
                        mask = mask.cpu().numpy().astype('float32')
                        
#                         np.save(str(to_save / annotation), mask)
                        
                        if args.thresh is None:
                            m = mask.argmax(0) == 1
                        elif not watershed:
                            m = mask[0] > args.thresh
                        else:
                            m = get_input(mask)
                            
                        m = Image.fromarray(((m*255).astype('uint8')/255).astype('uint8'))
                        m.save(str(to_save / annotation), compression='tiff_deflate')

                        m = np.array(m).astype('uint8')*255
                        vectordata = mask_to_poly_geojson(
                            m,
                            output_path=None,
                            output_type='csv',
                            min_area=0,
                            bg_threshold=128,
                            do_transform=False,
                            simplify=True
                        )

                        poly = vectordata['geometry']
                        if poly.empty:
                            poly = ['POLYGON EMPTY']

                        sub = pd.DataFrame({
                            'ImageId': '_'.join(annotation[:-4].split('_')[-4:]),
                            'BuildingId': 0,
                            'PolygonWKT_Pix': poly,
                            'Confidence': 1
                        })

                        sub['BuildingId'] = range(len(sub))
                        submit = submit.append(sub)

        return submit                

    submit = get_submit()
    submit.to_csv(args.submit_path, index=False)
    
    
if __name__ == '__main__':
    main()
