import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def kl_div_var_specific(q_mu, q_var, eps=1e-12):
      return -0.5 * (1 + torch.log(q_var) - torch.pow(q_mu, 2) - q_var)

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def corherent_loss_specific(self, uniview_dist_mu, uniview_dist_sca, mask=None):
        if mask is None:
            mask = torch.ones_like((uniview_dist_mu[0].shape[0], len(uniview_dist_mu))).to(uniview_dist_mu[0].device)

        z_tc_loss = []
        norm = torch.sum(mask, dim=1)
        weight = F.softmax(torch.stack(uniview_dist_sca, dim=1), dim=1)
        for v in range(len(uniview_dist_mu)):
            zv_tc_loss = torch.mean(kl_div_var_specific(uniview_dist_mu[v], uniview_dist_sca[v]),dim=1)
            exist_loss = zv_tc_loss * mask[:, v]
            z_tc_loss.append(exist_loss)
        z_tc_loss = torch.stack(z_tc_loss, dim=1)

        sample_ave_tc_term_loss = torch.sum(z_tc_loss) / mask.sum()

        return sample_ave_tc_term_loss

    def weighted_BCE_loss(self,pred,label,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(pred))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - pred + 1e-5))).item() == 0
        res=torch.abs((label.mul(torch.log(pred + 1e-5)) \
                                                + (1-label).mul(torch.log(1 - pred + 1e-5))).mul(inc_L_ind))
        assert torch.sum(torch.isnan(res)).item() == 0
        assert torch.sum(torch.isinf(res)).item() == 0

        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res

    def weighted_wmse_loss_sum(self,input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        if torch.sum(torch.isnan(ret)).item()>0:
            print(ret)
        if reduction == 'mean':
            # return torch.mean(ret)
            return torch.mean(ret, dim=1)
        elif reduction=='sum':
            return torch.sum(ret)
        elif reduction=='none':
            return ret








    
    