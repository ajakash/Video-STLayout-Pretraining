import ipdb
import sys
import argparse
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# from models import BBox2Activity
# from dataset import BBox2ActivityDataset

def get_dataset(split, data_dir):
    data = torch.load(f'{data_dir}/{split}.pt')
    dataset = TensorDataset(data['bbox_sets'], 
                            data['bbox_masks'], 
                            data['sact_label_indices'])
    
    # print(f"Loaded {split} data with {data['bbox_sets'].size(0)} items.")

    return dataset


def compute_loss(model, tgt, data, criterion, device, split='train'):
    start_time = time.time()
    epoch_loss = 0.0
    accuracy_top1 = 0.0
    accuracy_top5 = 0.0

    print_output = True

    with torch.no_grad():
        for bbox_set, bbox_mask, sact_label_indices in data:
            bbox_set = bbox_set.to(device)
            bbox_mask = bbox_mask.to(device)
            sact_label_indices = sact_label_indices.to(device)

            output = model(bbox_set, tgt, bbox_mask)
            # ipdb.set_trace()
            output = output.view(-1, args.output_dim)
            target = sact_label_indices.view(-1)
            if print_output:
                print(output)
                print_output = False
            # sys.exit()
            # ipdb.set_trace()

            # Top-1 accuracy
            correct = (output.max(1)[1] == target).sum().item()
            accuracy_top1 += (correct / target.size(0)) * 100.0
            
            # Top-5 accuracy
            _, predicted = output.topk(5, 1, True, True)
            correct = predicted.eq(target.view(-1, 1).expand_as(predicted))
            accuracy_top5 += correct.sum(1).float().mean().item() * 100.0

            loss = criterion(output, target)
            epoch_loss += loss.item()

        print(f'{split} loss: {epoch_loss / len(data):.4f},  acc1: {accuracy_top1 / len(data):.2f},  \
              acc5: {accuracy_top5 / len(data):.2f}, \t Time: {time.time() - start_time:.2f}')

def main(args): 
    torch.manual_seed(args.seed)

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Load and preprocess the dataset
    # train_data = DataLoader(get_dataset('train', args.data_dir), batch_size=args.batch_size)
    val_data = DataLoader(get_dataset('val', args.data_dir), batch_size=args.batch_size)
    # test_data = DataLoader(get_dataset('test', args.data_dir), batch_size=args.batch_size)
    tgt = torch.zeros([args.batch_size, 1], dtype=torch.int, device=device)

    # with open(args.models_file) as fp:
    #     Lines = fp.readlines()
    #     for line in Lines:
    #         # Create the model instance
    #         model = torch.load(f'saved_models/{line.rstrip()}').to(device)

    #         print(line.rstrip(), end=",  ")
    #         model.eval()
    #         # compute_loss(model, tgt, train_data, criterion, device, split='train')
    #         # compute_loss(model, tgt, val_data, criterion, device, split='val')
    #         compute_loss(model, tgt, test_data, criterion, device, split='test')

    # model = torch.load(f'saved_models/H4_L3_D256_LR0001v2/bbox2activity_best.pt').to(device)
    model = torch.load(f'saved_models/{args.model_name}/bbox2activity_best.pt').to(device)
    best_epoch_file = f'saved_models/{args.model_name}/best_epoch.txt'
    with open(best_epoch_file, 'r') as f:
        for line in f:
            print(f"best epoch : {line}")
    model.eval()
    # compute_loss(model, tgt, train_data, criterion, device, split='train')
    compute_loss(model, tgt, val_data, criterion, device, split='val')
    # compute_loss(model, tgt, test_data, criterion, device, split='test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dim', type=int, default=91, \
                    help='number of classes for activity recognition')
    parser.add_argument('--model_path', type=str, default='saved_models/bbox2activity_best.pt',
                        help='path to the model to be loaded')
    parser.add_argument('--models_file', type=str, default='test_models.txt',
                        help='txt file with list of models in saved_models folder')
    parser.add_argument('--model_name', type=str, default='H4_L3_D256_LR0001v2')
    parser.add_argument('--data_dir', type=str, default='data')

    args = parser.parse_args()

    # print(args)
    print(args.model_name)

    main(args)