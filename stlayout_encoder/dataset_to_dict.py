'''
Saving box dataset tensors to a file with video filename as dict keys.
For use in video-box pretraining.
'''
import json
import ipdb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_moma import BBox2ActivityDataset

def get_dict(data, device='cpu'):
    d = dict()
    for bbox_set, bbox_mask, sact_label_index, video_fname in data:
        bbox_set = bbox_set.to(device)
        bbox_mask = bbox_mask.to(device)
        sact_label_index = sact_label_index.to(device)
        d[video_fname[0]] = [bbox_set[0], bbox_mask[0], sact_label_index[0]]
    
    return d

def main(args):
    # Set the device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    
    train_data = get_dict(DataLoader(BBox2ActivityDataset(split='train', dir_moma=args.data_dir), batch_size=args.batch_size))
    val_data = get_dict(DataLoader(BBox2ActivityDataset(split='val', dir_moma=args.data_dir), batch_size=args.batch_size))
    train_val_data = train_data.update(val_data)

    torch.save(train_data, f'data_VideoSTLayoutPT/train_sact.pt')
    torch.save(val_data, f'data_VideoSTLayoutPT/val_sact.pt')
    torch.save(train_val_data, f'data_VideoSTLayoutPT/train_val_sact.pt')

    train_data, val_data, train_val_data = 0,0,0

    test_data = get_dict(DataLoader(BBox2ActivityDataset(split='test', dir_moma=args.data_dir), batch_size=args.batch_size))
    torch.save(test_data, f'data_VideoSTLayoutPT/test_sact.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='/Users/akashaj/Work/Datasets/MOMA-LRG')

    args = parser.parse_args()

    print(args)

    main(args)