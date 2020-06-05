import argparse
from pathlib import Path

import apex
import numpy as np
import segmentation_models_pytorch as smp
import torch
import tqdm

from dataset import CloudsDS, dev_transform, collate_fn
from metric import Dice, JaccardMicro
from utils import get_data_groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data')
    parser.add_argument('--load', type=str, required=True,
                        help='Load model')
    parser.add_argument('--save', type=str, default='',
                        help='Save predictions')
    parser.add_argument('--tta', type=int, default=0,
                        help='Test time augmentations')

    return parser.parse_args()
    

def epoch_step(loader, desc, model, metrics):
    model.eval()

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_targets, loc_preds = [], []
    if args.cls:
        loc_preds_cls = []
    
    
    for x, y in loader:
        x, y = x.cuda(args.gpu), y.cuda(args.gpu).float()
        
        masks = []
        if args.cls:
            clsss = []

        logits = model(x)
        if args.cls:
            logits, cls = logits
            clsss.append(torch.sigmoid(cls).cpu().numpy())
        if not args.use_softmax:
            masks.append(torch.sigmoid(logits).cpu().numpy()[..., 62:-62, 62:-62])
        else:
            masks.append(torch.softmax(logits, dim=1).cpu().numpy()[..., 62:-62, 62:-62])
        
        if args.tta > 0:
            logits = model(torch.flip(x, dims=[-1]))
            if args.cls:
                logits, cls = logits
                clsss.append(torch.sigmoid(cls).cpu().numpy())
            if not args.use_softmax:
                masks.append(torch.flip(torch.sigmoid(logits), dims=[-1]).cpu().numpy()[..., 62:-62, 62:-62])
            else:
                masks.append(torch.flip(torch.softmax(logits, dim=1), dims=[-1]).cpu().numpy()[..., 62:-62, 62:-62])

        if args.tta > 1:
            logits = model(torch.flip(x, dims=[-2]))
            if args.cls:
                logits, cls = logits
                clsss.append(torch.sigmoid(cls).cpu().numpy())
            if not args.use_softmax:
                masks.append(torch.flip(torch.sigmoid(logits), dims=[-2]).cpu().numpy()[..., 62:-62, 62:-62])
            else:
                masks.append(torch.flip(torch.softmax(logits, dim=1), dims=[-2]).cpu().numpy()[..., 62:-62, 62:-62])

        if args.tta > 2:
            logits = model(torch.flip(x, dims=[-1, -2]))
            if args.cls:
                logits, cls = logits
                clsss.append(torch.sigmoid(cls).cpu().numpy())
            if not args.use_softmax:
                masks.append(torch.flip(torch.sigmoid(logits), dims=[-1, -2]).cpu().numpy()[..., 62:-62, 62:-62])
            else:
                masks.append(torch.flip(torch.softmax(logits, dim=1), dims=[-1, -2]).cpu().numpy()[..., 62:-62, 62:-62])

        trg = y.cpu().numpy()[..., 62:-62, 62:-62]
        loc_targets.extend(trg)
        preds = np.mean(masks, 0)
        loc_preds.extend(preds)
    
        for metric in metrics.values():
            metric.update(preds, trg)
        
        if args.cls:
            loc_preds_cls.extend(np.mean(clsss, 0))

        torch.cuda.synchronize()

        if args.local_rank == 0:
            pbar.set_postfix(**{
                k: f'{metric.evaluate():.4}' for k, metric in metrics.items()
            })
            pbar.update()

    pbar.close()
    
    if args.cls:
        return loc_targets, loc_preds, loc_preds_cls
    
    return loc_targets, loc_preds

    
def main():
    global args
    
    args = parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True

    args.gpu = 0
    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'
    
    path_to_load = Path(args.load)
    if path_to_load.is_file():
        print(f"=> Loading checkpoint '{path_to_load}'")
        checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda(args.gpu))
        print(f"=> Loaded checkpoint '{path_to_load}'")
    else:
        raise

    tta = args.tta
    args = checkpoint['args']
    args.tta = tta
    print(args)

    if args.cls:
        print('With classification')
    else:
        print('Without classification')

    model = smp.Unet(encoder_name=args.encoder,
                     encoder_weights='imagenet' if 'dpn92' not in args.encoder else 'imagenet+5k',
                     classes=args.n_classes,
                     in_channels=args.in_channels,
                     decoder_attention_type=args.attention_type,
                     activation=None)
    
    model.cuda()
     
    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    args.fp16 = False
    assert args.fp16 == False, "torch script doesn't work with amp"
    if args.fp16:
        model = apex.amp.initialize(model,
                                    opt_level=args.opt_level,
                                    keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                    loss_scale=args.loss_scale
                                   )
    
    work_dir = path_to_load.parent
    
    import copy
    state_dict = copy.deepcopy(checkpoint['state_dict'])
    for p in checkpoint['state_dict']:
        if p.startswith('module.'):
            state_dict[p[len('module.'):]] = state_dict.pop(p)

    model.load_state_dict(state_dict)
    
    x = torch.rand(2, 4, 512, 512).cuda()
    model = model.eval()
    if 'efficientnet' in args.encoder:
        model.encoder.set_swish(memory_efficient=False)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)

    traced_model.save(str(work_dir / f'model_{path_to_load.stem}.pt'))


if __name__ == '__main__':
    main()
