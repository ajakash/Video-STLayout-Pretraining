# import ipdb
import os
import argparse
import torch
import time
import wandb
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import BBox2Activity
from dataset import BBox2ActivityDataset

def printlog(args, line):
    print(line)
    with open(args.checkpoint_dir+'/log.txt', 'a') as file:
        file.write(line+'\n')

def main(args):
    torch.manual_seed(args.seed)

    wandb.init(
        # set the wandb project where this run will be logged
        project="video-stlayout-pretraining",

        # track hyperparameters and run metadata
        config=args
    )
    wandb.run.name = args.output_dir
    args.checkpoint_dir = os.path.join('saved_models', args.output_dir)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set the device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device

    # Create the model instance
    model = BBox2Activity(args.input_dim, args.output_dim, args.hidden_dim, \
                          args.num_heads, args.num_layers, args.dropout).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Load and preprocess the dataset
    train_data = DataLoader(BBox2ActivityDataset(split='train', dir_moma=args.data_dir), batch_size=args.batch_size)
    val_data = DataLoader(BBox2ActivityDataset(split='val', dir_moma=args.data_dir), batch_size=args.batch_size)
    tgt = torch.zeros([args.batch_size, 1], dtype=torch.int, device=device)
    
    # Training loop
    min_val_loss = float('inf')
    for epoch in range(1, args.epochs):
        epoch_loss = 0.0
        model.train()
        start_time = time.time()

        for bbox_set, bbox_mask, sact_label_indices in train_data:
            bbox_set = bbox_set.to(device)
            bbox_mask = bbox_mask.to(device)
            sact_label_indices = sact_label_indices.to(device)
            optimizer.zero_grad()

            output = model(bbox_set, tgt, bbox_mask)
            output = output.view(-1, args.output_dim)
            target = sact_label_indices.view(-1)

            loss = criterion(output, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(train_data)
        printlog(args, f'Epoch {epoch} Loss: {epoch_loss:.4f} \
              \t Time: {time.time() - start_time:.2f}')
        wandb.log({"Train loss": epoch_loss}, step=epoch)

        if epoch % args.check_val_epoch == 0:
            # torch.save(model, f'saved_models/bbox2activity_{epoch}.pt') 
            model.eval()
            val_loss = 0.0
            start_time = time.time()

            with torch.no_grad():
                for bbox_set, bbox_mask, sact_label_indices in val_data:
                    bbox_set = bbox_set.to(device)
                    bbox_mask = bbox_mask.to(device)
                    sact_label_indices = sact_label_indices.to(device)
                    
                    # ipdb.set_trace()
                    output = model(bbox_set, tgt, bbox_mask)
                    output = output.view(-1, args.output_dim)
                    target = sact_label_indices.view(-1)

                    loss = criterion(output, target)

                    val_loss += loss.item()
                
                val_loss = val_loss / len(val_data)
                printlog(args, f'Epoch {epoch} Validation Loss: {val_loss:.4f} \
                      \t Time: {time.time() - start_time:.2f}')
                wandb.log({"Validation loss": val_loss}, step=epoch)
                
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                torch.save(model, f'{args.checkpoint_dir}/bbox2activity_best_{epoch}.pt')
                with open(f'{args.checkpoint_dir}/best_epoch.txt', 'w') as f:
                    f.write(f'{epoch}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='predictor')
    parser.add_argument('--input_dim', type=int, default=289, \
                        help='size of concatenated input for each frame: \
                            frame number encoding + bbox encoding + bbox label')
    parser.add_argument('--output_dim', type=int, default=91, \
                        help='number of classes for activity recognition')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--grad_clip', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--check_val_epoch', type=int, default=1)
    parser.add_argument('--log_dir', type=str, default='',
                        help='base directory to save logs')
    parser.add_argument('--data_dir', type=str, default='MOMA-LRG')

    args = parser.parse_args()

    print(args)

    main(args)