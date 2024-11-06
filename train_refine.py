import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
from datasets.kitti import KITTI

from timesformer.models.refinement import PoseRefinementModule as PRM, create_graph
import argparse
import time

def compute_loss(y_hat, y, criterion, weighted_loss=1.0):
    y = torch.reshape(y, (y.shape[0], 1, 6))
    gt_angles = y[:, :, :3].flatten()
    gt_translation = y[:, :, 3:].flatten()

    # predict pose
    y_hat = torch.reshape(y_hat, (y_hat.shape[0], 1,  6))
    estimated_angles = y_hat[:, :, :3].flatten()
    estimated_translation = y_hat[:, :, 3:].flatten()

    # compute custom loss
    k = weighted_loss
    loss_angles = k * criterion(estimated_angles, gt_angles.float())
    loss_translation = criterion(estimated_translation, gt_translation.float())
    loss =  loss_angles + loss_translation   
    return loss

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=256, help='Dimension of model')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--pred_path', type=str, 
                        default='/home/wangdongzhihan/codes/TSformer-VO/checkpoints/Exp7/checkpoint_best/pred_poses_00.npy', 
                        help='Path to predicted poses file')
    parser.add_argument('--weighted_loss', type=float, default=1.0, help='Weighted loss factor')
    parser.add_argument('--window_size', type=int, default=2, help='Number of consecutive poses to use for each graph')
    parser.add_argument('--checkpoint_path', type=str, default='/home/wangdongzhihan/codes/BrightVO/checkpoints/PoseRefinement', help='Path to save checkpoints')
    args = parser.parse_args()

    # Parameters
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    pred_path = args.pred_path
    weighted_loss = args.weighted_loss
    window_size = args.window_size

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # create checkpoints folder
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    checkpoint_path = args.checkpoint_path

    # tensorboard writer
    TensorBoardWriter = SummaryWriter(log_dir=args.checkpoint_path)


    dataset = KITTI()
    train_size = int(0.8 * len(dataset))
    # val_size = int(len(dataset) - train_size)
    # train_data, val_data = random_split(dataset, [train_size, val_size]) #generator=torch.Generator().manual_seed(2))
    train_data = Subset(dataset, range(train_size))
    val_data = Subset(dataset, range(train_size, len(dataset)))
    train_loader = DataLoader(train_data, 
                              batch_size=1, 
                              shuffle=True)
    val_loader = DataLoader(val_data, 
                            batch_size=1, 
                            shuffle=False)

    # Model, loss, optimizer
    model = PRM(d_model=d_model, num_heads=num_heads, num_layers=num_layers)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        pred_poses = np.load(pred_path).squeeze(1)  # Load once outside the loop
        iter = (epoch - 1) * len(train_loader) + 1
        with tqdm(train_loader, unit="batch") as tepoch:
            for idx, (images, gt_pose, tstmp) in enumerate(tepoch):
                images, gt_pose = images.to(device), gt_pose.to(device)

                graph = create_graph(pred_poses, idx, window_size)
                pose, edge_idx = graph.x.to(device), graph.edge_index.to(device)

                batch_dict = {
                    'pose': pose,
                    'images': images,
                    'head_mask': None,
                    'output_attentions': False,
                    'output_hidden_states': True,
                }
                refined_pose = model(batch_dict)
                refined_pose = refined_pose.to(device)

                loss = compute_loss(refined_pose, gt_pose, criterion, weighted_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                iter += 1
                train_loss += loss.item()
                TensorBoardWriter.add_scalar('training_loss', loss.item(), iter)

        train_loss /= len(train_loader)  # Move outside the loop
        TensorBoardWriter.add_scalar('train_loss', train_loss, epoch)
        # Save model weights
        torch.save(model.state_dict(), os.path.join(checkpoint_path, f'checkpoint_{epoch+1}.pth'))

        if val_loader:
            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for idx, (_, gt_pose, tstmp) in enumerate(tqdm(val_loader, desc="Validation")):
                    gt_pose = gt_pose.cuda()  # [1, batch_size, 6]

                    graph = create_graph(pred_poses, idx, window_size)
                    x, edge_idx = graph.x.to(device), graph.edge_index.to(device)   

                    refined_pose = model(x, edge_idx, tstmp)
                    refined_pose = refined_pose.to(device)
                    loss = criterion(refined_pose, gt_pose) * weighted_loss
                    val_loss += loss.item()        
            avg_loss = val_loss / len(val_loader)
            TensorBoardWriter.add_scalar('avg_validation_loss', avg_loss, epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")