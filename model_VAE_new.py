import torch
import numpy as np
import torch.nn as nn

def gaussian_reparameterization_var(means, var, train, times=1):
    if train:
        std = torch.sqrt(var)
        res = torch.zeros_like(means).to(means.device)
        for t in range(times):
            epi = std.data.new(std.size()).normal_()
            res += epi * std + means
        return res/times
    else:
        return means

class MLP(nn.Module):
    def __init__(self, in_dim,  out_dim,hidden_dim:list=[512,1024,1024,1024,512], act =nn.GELU,norm=nn.BatchNorm1d,dropout_rate=0.,final_act=True,final_norm=True):
        super(MLP, self).__init__()
        self.act = act
        self.norm = norm
        # init layers
        self.mlps =[]
        layers = []
        
        if len(hidden_dim)>0:
            layers.append(nn.Linear(in_dim, hidden_dim[0]))
            # layers.append(nn.Dropout(dropout_rate))
            layers.append(self.norm(hidden_dim[0]))
            layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
            ##hidden layer
            for i in range(len(hidden_dim)-1):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
                # layers.append(nn.Dropout(dropout_rate))
                layers.append(self.norm(hidden_dim[i+1]))
                layers.append(self.act())
                self.mlps.append(nn.Sequential(*layers))
                layers = []
            ##output layer
            layers.append(nn.Linear(hidden_dim[-1], out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            # layers.append(nn.Dropout(dropout_rate))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
            layers = []
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if final_norm:
                layers.append(self.norm(out_dim))
            if final_act:
                layers.append(self.act())
            self.mlps.append(nn.Sequential(*layers))
        self.mlps = nn.ModuleList(self.mlps)
    def forward(self, x):
        for layers in self.mlps:
            x = layers(x)
            # x = x + y
        return x

class sharedQz_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(sharedQz_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())

    def forward(self, x):
        hidden_features = self.mlp(x)
        z_mu = self.z_loc(hidden_features)
        z_sca = self.z_sca(hidden_features)
        return z_mu, z_sca

class specificQz_inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(specificQz_inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)
        self.z_loc = nn.Linear(out_dim, out_dim)
        self.z_sca = nn.Sequential(nn.Linear(out_dim, out_dim), nn.Softplus())

    def forward(self, x):
        hidden_features = self.mlp(x)
        z_mu = self.z_loc(hidden_features)
        z_sca = self.z_sca(hidden_features)
        return z_mu, z_sca

class inference_mlp(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=[1024]):
        super(inference_mlp, self).__init__()
        self.transfer_act = nn.ReLU
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim)

    def forward(self, x):
        hidden_features = self.mlp(x)
        return hidden_features

class Px_generation_mlp(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=[512]):
        super(Px_generation_mlp, self).__init__()
        self.mlp = MLP(in_dim, out_dim,hidden_dim=hidden_dim,final_act=False,final_norm=False)

    def forward(self, z):
        xr = self.mlp(z)
        return xr

class VAE_consist(nn.Module):
    def __init__(self, d_list,z_dim,class_num):
        super(VAE_consist, self).__init__()
        self.x_dim_list = d_list
        self.k = class_num
        self.z_dim = z_dim
        self.num_views = len(d_list)

        self.z_inference = []
        for v in range(self.num_views):
            self.z_inference.append(inference_mlp(self.x_dim_list[v], self.z_dim))
        self.qz_inference = nn.ModuleList(self.z_inference)
        self.qz_inference_header = sharedQz_inference_mlp(self.z_dim, self.z_dim)

        self.x_generation = []
        for v in range(self.num_views):
            self.x_generation.append(Px_generation_mlp(self.z_dim,self.x_dim_list[v]))
        self.px_generation = nn.ModuleList(self.x_generation)
        self.px_generation2 = nn.ModuleList(self.x_generation)

    def inference_z(self, x_list):
        uniview_mu_list = []
        uniview_sca_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            fea = self.qz_inference[v](x_list[v])
            if torch.sum(torch.isnan(fea)).item() > 0:
                print("zz:nan")
                pass
            z_mu_v, z_sca_v = self.qz_inference_header(fea)
            if torch.sum(torch.isnan(z_mu_v)).item() > 0:
                print("zzmu:nan")
                pass
            if torch.sum(torch.isnan(z_sca_v)).item() > 0:
                print("zzvar:nan")
                pass
            uniview_mu_list.append(z_mu_v)
            uniview_sca_list.append(z_sca_v)

        return uniview_mu_list, uniview_sca_list

    def generation_crossview_x(self, z):
        xr_dist = []
        for v in range(self.num_views):
            xr_list = []
            for j in range(self.num_views):
                xrs_loc = self.px_generation[j](z[v])
                xr_list.append(xrs_loc)
            xr_dist.append(xr_list)
        return xr_dist

    def poe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0,1).unsqueeze(-1)
        mask_matrix_new = torch.cat([torch.ones([1,mask_matrix.shape[1],mask_matrix.shape[2]]).cuda(),mask_matrix],dim=0)
        p_z_mu = torch.zeros([1,mu.shape[1],mu.shape[2]]).cuda()
        p_z_var = torch.ones([1,mu.shape[1],mu.shape[2]]).cuda()
        mu_new = torch.cat([p_z_mu,mu],dim=0)
        var_new = torch.cat([p_z_var,var],dim=0)
        exist_mu = mu_new * mask_matrix_new
        
        T = 1. / (var_new+eps)
        if torch.sum(torch.isnan(exist_mu)).item()>0:
            print('.')
        if torch.sum(torch.isinf(T)).item()>0:
            print('.')
        exist_T = T * mask_matrix_new
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / (aggregate_T + eps)
        if torch.sum(torch.isnan(aggregate_mu)).item()>0:
            print(',')
        return aggregate_mu, aggregate_var

    def forward(self, x_list, mask=None):
        uniview_mu_list, uniview_sca_list = self.inference_z(x_list)
        z_mu = torch.stack(uniview_mu_list,dim=0)
        z_sca = torch.stack(uniview_sca_list,dim=0)

        if torch.sum(torch.isnan(z_mu)).item() > 0:
            print("z:nan")
            pass

        fusion_mu, fusion_sca = self.poe_aggregate(z_mu, z_sca, mask)

        if torch.sum(torch.isnan(fusion_mu)).item() > 0:
            pass
        assert torch.sum(fusion_sca<0).item() == 0
        z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca,self.training,times=10)
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            print("z:nan")
            pass

        view_sample_list = []
        for v in range(self.num_views):
            view_sample_list.append(gaussian_reparameterization_var(uniview_mu_list[v], uniview_sca_list[v],self.training,times=10))
        xr_list = self.generation_crossview_x(view_sample_list)

        return z_sample, uniview_mu_list, uniview_sca_list, fusion_mu, fusion_sca, xr_list

class VAE_consist_encoder(nn.Module):
    def __init__(self, vae):
        super(VAE_consist_encoder, self).__init__()
        self.x_dim_list = vae.x_dim_list
        self.k = vae.k
        self.z_dim = vae.z_dim
        self.num_views = vae.num_views

        self.qz_inference = vae.qz_inference
        self.qz_inference_header = vae.qz_inference_header

    def inference_z(self, x_list):
        uniview_mu_list = []
        uniview_sca_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            fea = self.qz_inference[v](x_list[v])
            if torch.sum(torch.isnan(fea)).item() > 0:
                print("zz:nan")
                pass
            z_mu_v, z_sca_v = self.qz_inference_header(fea)
            if torch.sum(torch.isnan(z_mu_v)).item() > 0:
                print("zzmu:nan")
                pass
            if torch.sum(torch.isnan(z_sca_v)).item() > 0:
                print("zzvar:nan")
                pass
            uniview_mu_list.append(z_mu_v)
            uniview_sca_list.append(z_sca_v)

        return uniview_mu_list, uniview_sca_list

    def poe_aggregate(self, mu, var, mask=None, eps=1e-5):
        if mask is None:
            mask_matrix = torch.ones_like(mu).to(mu.device)
        else:
            mask_matrix = mask.transpose(0, 1).unsqueeze(-1)
        mask_matrix_new = torch.cat([torch.ones([1, mask_matrix.shape[1], mask_matrix.shape[2]]).cuda(), mask_matrix],dim=0)
        p_z_mu = torch.zeros([1, mu.shape[1], mu.shape[2]]).cuda()
        p_z_var = torch.ones([1, mu.shape[1], mu.shape[2]]).cuda()
        mu_new = torch.cat([p_z_mu, mu], dim=0)
        var_new = torch.cat([p_z_var, var], dim=0)
        exist_mu = mu_new * mask_matrix_new

        T = 1. / (var_new + eps)
        if torch.sum(torch.isnan(exist_mu)).item() > 0:
            print('.')
        if torch.sum(torch.isinf(T)).item() > 0:
            print('.')
        exist_T = T * mask_matrix_new
        aggregate_T = torch.sum(exist_T, dim=0)
        aggregate_var = 1. / (aggregate_T + eps)
        aggregate_mu = torch.sum(exist_mu * exist_T, dim=0) / (aggregate_T + eps)
        if torch.sum(torch.isnan(aggregate_mu)).item() > 0:
            print(',')
        return aggregate_mu, aggregate_var

    def forward(self, x_list, mask=None):
        uniview_mu_list, uniview_sca_list = self.inference_z(x_list)
        z_mu = torch.stack(uniview_mu_list, dim=0)
        z_sca = torch.stack(uniview_sca_list, dim=0)

        if torch.sum(torch.isnan(z_mu)).item() > 0:
            print("z:nan")
            pass

        fusion_mu, fusion_sca = self.poe_aggregate(z_mu, z_sca, mask)

        if torch.sum(torch.isnan(fusion_mu)).item() > 0:
            pass
        assert torch.sum(fusion_sca < 0).item() == 0
        z_sample = gaussian_reparameterization_var(fusion_mu, fusion_sca, self.training, times=10)
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            print("z:nan")
            pass

        view_sample_list = []
        for v in range(self.num_views):
            view_sample_list.append(gaussian_reparameterization_var(uniview_mu_list[v], uniview_sca_list[v], self.training, times=10))

        return z_sample, uniview_mu_list, uniview_sca_list, fusion_mu, fusion_sca, view_sample_list


class VAE_specific(nn.Module):
    def __init__(self, d_list, z_dim, class_num):
        super(VAE_specific, self).__init__()
        self.x_dim_list = d_list
        self.k = class_num
        self.z_dim = z_dim
        self.num_views = len(d_list)

        self.z_inference = []
        for v in range(self.num_views):
            self.z_inference.append(inference_mlp(self.x_dim_list[v], self.z_dim))
        self.qz_inference = nn.ModuleList(self.z_inference)
        self.z_inference_header = []
        for v in range(self.num_views):
            self.z_inference_header.append(specificQz_inference_mlp(self.z_dim, self.z_dim))
        self.qz_inference_header = nn.ModuleList(self.z_inference_header)

        self.x_generation = []
        for v in range(self.num_views):
            self.x_generation.append(Px_generation_mlp(self.z_dim*2, self.x_dim_list[v]))
        self.px_generation = nn.ModuleList(self.x_generation)

    def inference_z(self, x_list):
        uniview_mu_list = []
        uniview_sca_list = []
        for v in range(self.num_views):
            if torch.sum(torch.isnan(x_list[v])).item() > 0:
                print("zzz:nan")
                pass
            fea = self.qz_inference[v](x_list[v])
            if torch.sum(torch.isnan(fea)).item() > 0:
                print("zz:nan")
                pass
            z_mu_v, z_sca_v = self.qz_inference_header[v](fea)
            if torch.sum(torch.isnan(z_mu_v)).item() > 0:
                print("zzmu:nan")
                pass
            if torch.sum(torch.isnan(z_sca_v)).item() > 0:
                print("zzvar:nan")
                pass
            uniview_mu_list.append(z_mu_v)
            uniview_sca_list.append(z_sca_v)

        return uniview_mu_list, uniview_sca_list

    def generation_x(self, z_specific):
        xr_dist = []
        for v in range(self.num_views):
            xrs_loc = self.px_generation[v](z_specific[v])
            xr_dist.append(xrs_loc)
        return xr_dist

    def forward(self, x_list, z ,view_sample_list, mask=None):
        uniview_mu_list, uniview_sca_list = self.inference_z(x_list)

        mask = mask.float()
        summvz = 0
        z_specific = []
        z_specific_cat = []
        for v in range(self.num_views):
            z_specific.append(gaussian_reparameterization_var(uniview_mu_list[v], uniview_sca_list[v],self.training, times=10))
            z_specific_cat.append(torch.cat((z_specific[v], view_sample_list[v]), dim=1))
            summvz += torch.diag(mask[:, v]).mm(z_specific[v])
        wei = 1 / torch.sum(mask, 1)
        z_specific_fusion = torch.diag(wei).mm(summvz)


        xr_list = self.generation_x(z_specific_cat)
        return z_specific,z_specific_cat,z_specific_fusion, uniview_mu_list, uniview_sca_list, xr_list


class VGAE_label_embedding(nn.Module):
    def __init__(self, in_dim, out_dim,num_layers,num_classes,adj,hidden_dim=[1024]):
        super(VGAE_label_embedding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.Adj = adj.float().to('cuda:0')

        self.GIN_encoder = nn.ModuleList()
        if in_dim != out_dim:
            first_layer_res = False
        else:
            first_layer_res = True
        batchNorm = True
        nonlinearity = 'leaky_relu'
        negative_slope = 0.01
        eps = 0.0
        train_eps = True
        residual = True
        self.GIN_encoder.append(GINLayer(GIN_MLP(in_dim, out_dim, hidden_dim, batchNorm,
                                                 nonlinearity, negative_slope), eps, train_eps, first_layer_res))
        for i in range(num_layers - 1):
            self.GIN_encoder.append(GINLayer(GIN_MLP(out_dim, out_dim, hidden_dim, batchNorm,
                                                     nonlinearity, negative_slope), eps, train_eps, residual))

        self.gcn_mean = GINLayer(GIN_MLP(out_dim, out_dim, [], batchNorm, nonlinearity, negative_slope,
                                         with_output_nonlineartity=False), eps, train_eps, False)
        self.gcn_var = GINLayer(GIN_MLP(out_dim, out_dim, [], batchNorm, 'Softplus', negative_slope
                                              ), eps, train_eps, False)

    def encode(self, X,adj):
        hidden = X
        for l in self.GIN_encoder:
            hidden = l(hidden, adj)

        mean = self.gcn_mean(hidden,adj)
        var = self.gcn_var(hidden,adj)
        if self.training:
            gaussian_noise = torch.randn(X.size(0), self.out_dim).to('cuda:0')
            std = torch.sqrt(var)
            sampled_z = gaussian_noise * std + mean
        else:
            sampled_z = mean
        return sampled_z,mean,var

    def forward(self, X):
        Z,mean,var = self.encode(X,self.Adj)
        A_pred = dot_product_decode(Z)
        return A_pred, Z , mean ,var

def dot_product_decode(Z):
    A_pred = torch.sigmoid(torch.matmul(Z,Z.t())) #计算潜在表示Z的点积，并通过sigmoid函数将结果映射到(0,1)区间，表示边的存在概率
    return A_pred

class GIN_MLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=[], batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.1,
                 with_output_nonlineartity=True):
        super(GIN_MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        self.fcs = nn.ModuleList()
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]
            for i in range(len(in_dims)):
                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))
                if with_output_nonlineartity or i < len(hidden_features):
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    elif nonlinearity == 'Softplus':
                        self.fcs.append(nn.Softplus( ))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
        else:
            self.fcs.append(nn.Linear(in_features, out_features))
            if with_output_nonlineartity:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_features, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                elif nonlinearity == 'Softplus':
                    self.fcs.append(nn.Softplus())
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


    def reset_parameters(self):
        for l in self.fcs:
            if l.__class__.__name__ == 'Linear':
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope,
                                         nonlinearity=self.nonlinearity)
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            elif l.__class__.__name__ == 'BatchNorm1d':
                l.reset_parameters()

    def forward(self, input):
        for l in self.fcs:
            input = l(input)
        return input

class GINLayer(nn.Module):
    def __init__(self, mlp, eps=0.0, train_eps=True, residual=True):
        super(GINLayer, self).__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.residual = residual

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, input, adj):
        res = input

        neighs = torch.matmul(adj, res)

        res = (1 + self.eps) * res + neighs

        res = self.mlp(res)

        if self.residual:
            output = res + input
        else:
            output = res

        return output
