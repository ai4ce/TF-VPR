from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import math
import torchvision.models as models

class NetVLAD_Image(nn.Module):
    """NetVLAD layer implementation"""
    def __init__(self, num_clusters=64, dim=128, 
                normalize_input=True, vladv2=False):
            
        """
        Args:
        num_clusters : int
            The number of clusters
        
        dim : int
             Dimension of descriptors
        
        alpha : float
            Parameter of initialization. Larger value is harder assignment.
        
        normalize_input : bool
            If true, descriptor-wise L2 normalization is applied to input.
        
        vladv2 : bool
            If true, use vladv2 otherwise use vladv1
        """
        
        super(NetVLAD_Image, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = 0
        self.vladv2 = vladv2
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=vladv2)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
    
    def init_params(self, clsts, traindescs):
        
        #TODO replace numpy ops with pytorch ops
        if self.vladv2 == False:
            clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
            dots = np.dot(clstsAssign, traindescs.T)
            dots.sort(0)
            dots = dots[::-1, :] # sort, descending
            
            self.alpha = (-np.log(0.01) / np.mean(dots[0,:] - dots[1,:])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha*clstsAssign).unsqueeze(2).unsqueeze(3))
            self.conv.bias = None
        else:
            knn = NearestNeighbors(n_jobs=-1) #TODO faiss?
            knn.fit(traindescs)
            del traindescs
            dsSq = np.square(knn.kneighbors(clsts, 2)[1])
            del knn
            self.alpha = (-np.log(0.01) / np.mean(dsSq[:,1] - dsSq[:,0])).item()
            self.centroids = nn.Parameter(torch.from_numpy(clsts))
            del clsts, dsSq

            self.conv.weight = nn.Parameter(
                    (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
            )
            
            self.conv.bias = nn.Parameter(
                    - self.alpha * self.centroids.norm(dim=1)        
            )

    def forward(self, x):    
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)
        
        # calculate residuals to each clusters
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device)
        for C in range(self.num_clusters): # slower than non-looped, but lower memory usage 
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[C:C+1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:,C:C+1,:].unsqueeze(2)            
            vlad[:,C:C+1,:] = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten        
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad

class NetVLAD_Image2(nn.Module):
    """NetVLAD layer implementation"""
    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
            normalize_input=True):
        """
        Args:
        num_clusters : int
        The number of clusters
        dim : int
        Dimension of descriptors
        alpha : float
        Parameter of initialization. Larger value is harder assignment.
        normalize_input : bool
        If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD_Image, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()
    
    def _init_params(self):  
        self.conv.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)        
        )
        
        self.conv.bias = nn.Parameter(
                - self.alpha * self.centroids.norm(dim=1)        
        )
    
    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim
        
        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)
        x_flatten = x.view(N, C, -1)                                                                                                        

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        
        vlad = residual.sum(dim=-1)
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        return vlad

class NetVLADLoupe(nn.Module):
    def __init__(self, feature_size, max_samples, cluster_size, output_dim,
                 gating=True, add_batch_norm=True, is_training=True):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size
        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(torch.randn(
            feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.cluster_weights2 = nn.Parameter(torch.randn(
            1, feature_size, cluster_size) * 1 / math.sqrt(feature_size))
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim) * 1 / math.sqrt(feature_size))

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(
                cluster_size) * 1 / math.sqrt(feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm)

    def forward(self, x):
        x = x.transpose(1, 3).contiguous()
        x = x.view((-1, self.max_samples, self.feature_size))
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation = activation.view(-1,
                                         self.max_samples, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation)
        activation = activation.view((-1, self.max_samples, self.cluster_size))

        a_sum = activation.sum(-2, keepdim=True)
        a = a_sum * self.cluster_weights2

        activation = torch.transpose(activation, 2, 1)
        x = x.view((-1, self.max_samples, self.feature_size))
        vlad = torch.matmul(activation, x)
        vlad = torch.transpose(vlad, 2, 1)
        vlad = vlad - a

        vlad = F.normalize(vlad, dim=1, p=2)
        vlad = vlad.view((-1, self.cluster_size * self.feature_size))
        vlad = F.normalize(vlad, dim=1, p=2)

        vlad = torch.matmul(vlad, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(
            torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN2d(nn.Module):
    def __init__(self, num_points=256, k=2, use_bn=True):
        super(STN2d, self).__init__()
        self.k = k
        self.kernel_size = 2 if k == 2 else 1
        self.channels = 1 if k == 2 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)            
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)    
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        
        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
               1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class ObsFeatAVD(nn.Module):
    """Feature extractor for 2D organized point clouds"""
    def __init__(self, n_out=1024, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(ObsFeatAVD, self).__init__()
        self.n_out = n_out
        self.global_feature = global_feat
        self.feature_transform = feature_transform
        self.max_pool = max_pool
        k = 3
        p = int(np.floor(k / 2)) + 2
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.conv1 = nn.Conv2d(3,64,kernel_size=k,padding=p,dilation=3)        
        self.conv2 = nn.Conv2d(64,128,kernel_size=k,padding=p,dilation=3)
        self.conv3 = nn.Conv2d(128,256,kernel_size=k,padding=p,dilation=3)
        self.conv7 = nn.Conv2d(256,self.n_out,kernel_size=k,padding=p,dilation=3)
        self.amp = nn.AdaptiveMaxPool2d(1)
   
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans)
        x = x.view(batchsize, 1, -1, 3)
        x = x.permute(0,3,1,2)
        assert(x.shape[1]==3),"the input size must be <Bx3xHxW> "
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv7(x)
        if self.max_pool:
            x = self.amp(x) 
        #x = x.view(-1,self.n_out) #<Bxn_out>
        x = x.permute(0,2,3,1)
        return x

class ImageNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, max_pool=True):
        super(ImageNetfeat, self).__init__()
        encoder = models.vgg16(pretrained=True)
        layers = list(encoder.features.children())[:-2]

        self.base_model = nn.Sequential(*layers)

    def forward(self, x):
        B,_,g_x,g_y,_ = x.shape
        result = self.base_model(x.reshape(B,3,g_x,g_y))
        return result

class ImageNetVlad(nn.Module):
    def __init__(self, grid_x=256, grid_y=256, global_feat=True, feature_transform=False, max_pool=True, output_dim=1024):
        super(ImageNetVlad, self).__init__()
        
        self.obs_feat_extractor = ImageNetfeat(global_feat=global_feat,
                                               feature_transform=feature_transform, max_pool=max_pool)
        
        dim = list(self.obs_feat_extractor.parameters())[-1].shape[0]
        self.net_vlad = NetVLAD_Image(num_clusters=32, dim=512, vladv2=True)
        self.fc1 = nn.Linear(16384, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.obs_feat_extractor(x)
        x = self.net_vlad(x)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))

        return x


if __name__ == '__main__':
    num_points = 256
    sim_data = Variable(torch.rand(44, 1, num_points, 3))
    sim_data = sim_data.cuda()

    pnv = PointNetVlad.PointNetVlad(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=num_points).cuda()
    pnv.train()
    out3 = pnv(sim_data)
    print('pnv', out3.size())