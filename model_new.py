import torch
import torch.nn as nn 
import random
import numpy as np
from model_VAE_new import VAE_specific,VGAE_label_embedding

def Init_random_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Net(nn.Module):
    def __init__(self, d_list,num_classes,z_dim,adj,inp,rand_seed=0):
        super(Net, self).__init__()
        # self.configs = configs
        self.rand_seed = rand_seed
        # self.best_view = best_view
        self.num_views = len(d_list)

        self.inp = inp
        self.adj = adj
        self.z_dim = z_dim

        self.label_VGAE = VGAE_label_embedding(num_classes, z_dim,2,num_classes,self.adj,hidden_dim=[1024])
        self.mix_prior = None
        self.mix_mu = None
        self.mix_sca = None
        self.k = num_classes

        self.specificVAE = VAE_specific(d_list=d_list,z_dim=z_dim,class_num=num_classes)

        self.cls_conv = nn.Conv1d(num_classes, num_classes,
                                  z_dim, groups=num_classes)

        self.label_important = nn.Linear(z_dim,z_dim)
        self.set_prior()
        self.cuda()
    def set_prior(self):
        self.mix_prior = nn.Parameter(torch.full((self.k,), 1 / self.k), requires_grad=True)

        self.mix_mu = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)

        self.mix_sca = nn.Parameter(torch.rand((self.k,self.z_dim)),requires_grad=True)

    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.label_adj)
        self.FD_model.reset_parameters()
        self.cls_conv.reset_parameters()

    def get_VGAE_para(self):
        return self.label_VGAE.parameters()

    def get_other_para(self):
        all_params = set(self.parameters())
        vage_params = set(self.get_VGAE_para())
        return all_params - vage_params

    def forward(self, x_list, z_sample, view_sample_list , consisit_uniview_mu_list, consisit_uniview_sca_list, fusion_z_mu, fusion_z_sca,mask):
        A_pred, label_embedding, label_embedding_mean, label_embedding_var =  self.label_VGAE(self.inp)
        if torch.sum(torch.isnan(label_embedding)).item() > 0:
            assert torch.sum(torch.isnan(label_embedding)).item() == 0
        if torch.sum(torch.isnan(z_sample)).item() > 0:
            pass
        z_specific, z_specific_cat,z_specific_fusion,specific_uniview_mu_list,specific_uniview_sca_list,specific_xr_list= self.specificVAE(x_list,z_sample,view_sample_list,mask)

        final_feature = z_sample.mul(z_specific_fusion.sigmoid_())
        feature_importance = torch.sigmoid(self.label_important(label_embedding))
        qc_z =(final_feature.unsqueeze(1) * feature_importance.unsqueeze(0)).permute(0, 1, 2)
        p = self.cls_conv(qc_z).squeeze(-1)
        p = torch.sigmoid(p)
        return  z_sample, consisit_uniview_mu_list, consisit_uniview_sca_list, fusion_z_mu, fusion_z_sca,specific_uniview_mu_list,specific_uniview_sca_list,specific_xr_list,p,A_pred, label_embedding, label_embedding_mean, label_embedding_var

def get_model_second(d_list, num_classes, z_dim, adj,inp,rand_seed=0):
    model = Net(d_list, num_classes=num_classes, z_dim=z_dim, adj=adj,inp=inp, rand_seed=rand_seed)
    model = model.to(torch.device('cuda' if torch.cuda.is_available()
                                  else 'cpu'))
    return model
    
if __name__=="__main__":
    pass