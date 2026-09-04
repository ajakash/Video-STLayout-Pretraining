import json
import ipdb
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_moma import BBox2ActivityDataset

def save_data(data, device, filename):
    bbox_set_full, bbox_mask_full, sact_label_indices_full = [], [], []
    for bbox_set, bbox_mask, sact_label_indices in data:
        bbox_set = bbox_set.to(device)
        bbox_mask = bbox_mask.to(device)
        sact_label_indices = sact_label_indices.to(device)

        bbox_set_full.append(bbox_set)
        bbox_mask_full.append(bbox_mask)
        sact_label_indices_full.append(sact_label_indices)

    d = {"bbox_sets": torch.cat(bbox_set_full, dim=0),
         "bbox_masks": torch.cat(bbox_mask_full, dim=0),
         "sact_label_indices": torch.cat(sact_label_indices_full, dim=0)}
    torch.save(d, f'data/'+filename)

def main(args):
    # Set the device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    
    train_data = DataLoader(BBox2ActivityDataset(split='train', dir_moma=args.data_dir), batch_size=args.batch_size)
    val_data = DataLoader(BBox2ActivityDataset(split='val', dir_moma=args.data_dir), batch_size=args.batch_size)
    test_data = DataLoader(BBox2ActivityDataset(split='test', dir_moma=args.data_dir), batch_size=args.batch_size)

    save_data(train_data, device, "train.pt")
    save_data(val_data, device, "val.pt")
    save_data(test_data, device, "test.pt")    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='MOMA-LRG')

    args = parser.parse_args()

    print(args)

    main(args)