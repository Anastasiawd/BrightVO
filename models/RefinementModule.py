import time
import numpy as np

import torch
from torch import nn

import pypose as pp
import pypose.optim.solver as ppos
import pypose.optim.kernel as ppok
import pypose.optim.corrector as ppoc
import pypose.optim.strategy as ppost
from pypose.optim.scheduler import StopOnPlateau
from datasets.utils import euler_to_rotation
 
class PoseGraph(nn.Module):
    def __init__(self, nodes, reproj=None):
        super().__init__()
        self.nodes = pp.Parameter(nodes.clone())

        self.reproj = reproj


    def forward(self, edges, vo_poses, imu_poses):
        nodes = self.nodes

        [x, y, z] = vo_poses[0, :3].cpu().numpy()
        t = vo_poses[0, 3:].cpu().numpy()
        R = np.asarray(euler_to_rotation(x, y, z, seq = 'xyz'))
        vo_poses  = np.concatenate((np.concatenate([R, np.reshape(t, (3,1))], axis=1) , [[0.0, 0.0, 0.0, 1.0]] ), axis=0)
        vo_poses = pp.from_matrix(vo_poses, ltype=pp.SE3_type).float().to(nodes.device)
        
        [x1, y1, z1] = imu_poses[0, :3].cpu().numpy()
        t1 = imu_poses[0, 3:].cpu().numpy()
        R1 = np.asarray(euler_to_rotation(x1, y1, z1, seq = 'xyz'))
        imu_poses  = np.concatenate((np.concatenate([R1, np.reshape(t1, (3,1))], axis=1) , [[0.0, 0.0, 0.0, 1.0]] ), axis=0)
        imu_poses = pp.from_matrix(imu_poses, ltype=pp.SE3_type).float().to(nodes.device)

        imu_poses = pp.SE3(imu_poses)
        # E = edges.size(0)
        # M = nodes.size(0) - 1
        # assert E == poses.size(0)
        # assert M == imu_drots.size(0) == imu_dtrans.size(0) == imu_dvels.size(0)
        
        # VO constraint
        node1 = nodes[edges[:, 0]]
        node2 = nodes[edges[:, 1]]
        error = vo_poses.Inv() @ node1.Inv() @ node2
        voerr = error.Log().tensor().float()


        # imu constraint
        node1 = nodes[edges[:, 0]]
        node2 = nodes[edges[:, 1]]
        error2 = imu_poses.Inv() @ node1.Inv() @ node2
        imuerr = error2.Log().tensor().float()


        if self.reproj is not None:
            node1 = nodes[ :-1]
            node2 = nodes[1:  ]
            motion = node1.Inv() @ node2
            motion[0] = 0.1
            reprojerr = self.reproj(motion)
            if len(reprojerr.shape) == 3:
                reprojerr = reprojerr.view(-1, self.reproj.N*2)
            return voerr, imuerr, reprojerr
        
        else:
            return voerr, imuerr
    def align_to(self, target, idx=0):
        # align nodes[idx] to target
        source = self.nodes[idx].detach()
        nodes = target @ source.Inv() @ self.nodes
        return nodes


def RefinementBlock(init_nodes, vo_motions, links, imu_dposes, 
                device='cuda:0', loss_weight = [1,0.1], radius=1e4, reproj=None, turned_on=True):

    if not turned_on:
        return vo_motions
    vo_poses_infos = np.ones(len(links)) * loss_weight[0]**2  
    imu_poses_infos = np.ones(len(init_nodes)) * loss_weight[1]**2

    vo_info_mats= [torch.diag(torch.tensor([vo_poses_infos[i]]*3))
                          for i in range(len(vo_poses_infos))]
    imu_poses_info_mats = [torch.diag(torch.tensor([imu_poses_infos[i]]*3)) 
                          for i in range(len(imu_poses_infos))]
    
    # init inputs
    edges = links.to(device)
    poses = vo_motions.detach().to(device)
    

    imu_dposes = imu_dposes.detach().to(device)

    vo_info_mats = torch.stack(vo_info_mats).to(torch.float32).to(device)
    imu_poses_info_mats = torch.stack(imu_poses_info_mats).to(torch.float32).to(device)
    # imu_trans_info_mats = torch.stack(imu_trans_info_mats).to(torch.float32).to(device)
    weights = [vo_info_mats, imu_poses_info_mats]
    if reproj is not None:
        reproj_info_mats = torch.stack(reproj_info_mats).to(torch.float32).to(device)
        weights.append(reproj_info_mats)

    # build graph and optimizer
    graph = PoseGraph(init_nodes, reproj).to(device)
    solver = ppos.Cholesky()
    strategy = ppost.TrustRegion(radius=radius)
    optimizer = pp.optim.LM(graph, solver=solver, strategy=strategy, min=1e-4, vectorize=True)
    scheduler = StopOnPlateau(optimizer, steps=10, patience=3, decreasing=1e-3, verbose=False)

    start_time = time.time()

    # optimization loop
    while scheduler.continual():
        loss = optimizer.step(input=(edges, poses, imu_dposes), weight=weights)
        # loss = optimizer.step(input=(edges, poses, imu_drots, imu_dtrans, imu_dvels, dts))
        scheduler.step(loss)
    end_time = time.time()
    print('refine time:', end_time - start_time)
    print('final loss:', loss)

    # # get loss for backpropagate
    # if target == 'vo':
    #     trans_loss, rot_loss = graph.vo_loss(edges, vo_motions)
    # elif target == 'imu':
    #     trans_loss, rot_loss = graph.imu_loss(imu_dposes_grad)

    # # for test
    # # trans_loss, rot_loss = graph.vo_loss_unroll(edges, data.poses_withgrad)

    # align nodes to the original first pose
    nodes = graph.align_to(init_nodes[0].to(device))
    nodes = nodes.detach().cpu()
    # vels = vels.detach().cpu()

    # covs = {'vo_rot':vo_rot_infos, 'imu_rot':imu_rot_infos,
    #         'vo_trans':vo_trans_infos, 'imu_vel':imu_vel_infos,
    #         'transvel':transvel_infos}
    # if reproj is not None:
    #     covs['reproj'] = reproj_infos

    return  nodes

def create_graph(pose_sequence, idx, window_size=2):
    # Create graph for a window of consecutive poses (from start_idx to start_idx + window_size)
    x = torch.tensor(pose_sequence[idx:idx + window_size], dtype=torch.float)  # Shape: (window_size, 6)

    # Define edges: connect each pose in the window bidirectionally
    edge_index = []
    for i in range(window_size - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # Shape: (2, num_edges)

    return edge_index