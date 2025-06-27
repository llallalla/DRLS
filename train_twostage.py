import os
import os.path as osp
import utils
from utils import AverageMeter,binary_adj,get_config
import MLdataset
import argparse
import time
from model_new import get_model_second,Init_random_seed
from model_VAE_new import VAE_consist,VAE_consist_encoder
import evaluation
import torch
import numpy as np
import copy
from myloss import Loss
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

def train_first(loader, model, loss_model, opt, sche, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data = [v_data.to('cuda:0') for v_data in data]

        inc_V_ind = inc_V_ind.float().to('cuda:0')

        _, consisit_uniview_mu_list, consisit_uniview_sca_list, _, _, consist_xr_list = model(data, mask=inc_V_ind)

        if epoch < args.pre_epochs:
            loss_CL_views = 0

            loss = loss_CL_views
            assert torch.sum(torch.isnan(loss)).item() == 0
            assert torch.sum(torch.isinf(loss)).item() == 0
        else:
            cohr_loss_consist = loss_model.corherent_loss_specific(consisit_uniview_mu_list, consisit_uniview_sca_list, mask=inc_V_ind)

            loss_mse_consist = 0
            for v in range(len(data)):
                for j in range(len(data)):
                    loss_mse_consist += loss_model.weighted_wmse_loss_sum(data[j],consist_xr_list[v][j],(inc_V_ind[:,v].int() & inc_V_ind[:,j].int()).float(),reduction='mean')
            loss_mse_consist = torch.mean(loss_mse_consist/torch.sum(inc_V_ind,dim=1))

            assert torch.sum(torch.isnan(loss_mse_consist)).item() == 0
            loss =  loss_mse_consist + cohr_loss_consist * args.alpha

        opt.zero_grad()
        loss.backward()
        if isinstance(sche, CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))

        opt.step()
        losses.update(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()
    if isinstance(sche, StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                'Time {batch_time.avg:.3f}\t'
                'Data {data_time.avg:.3f}\t'
                'Loss {losses.avg:.3f}'.format(
        epoch, batch_time=batch_time,
        data_time=data_time, losses=losses))
    return losses, model

def train_second(loader, model_first,model_second, loss_model ,opt, sche, epoch,logger,adj):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model_first.train()
    model_second.train()
    end = time.time()

    All_preds = torch.tensor([]).cuda()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data_time.update(time.time() - end)
        data=[v_data.to('cuda:0') for v_data in data]

        label = label.to('cuda:0')
        inc_V_ind = inc_V_ind.float().to('cuda:0')
        inc_L_ind = inc_L_ind.float().to('cuda:0')

        z_sample, consisit_uniview_mu_list, consisit_uniview_sca_list, fusion_z_mu, fusion_z_sca , view_sample_list= model_first(data, mask=inc_V_ind)

        z_sample, consisit_uniview_mu_list, consisit_uniview_sca_list, fusion_z_mu, fusion_z_sca, specific_uniview_mu_list, specific_uniview_sca_list, specific_xr_list, pred, A_pred, label_embedding, label_embedding_mean, label_embedding_var = model_second(
            data,z_sample,view_sample_list, consisit_uniview_mu_list, consisit_uniview_sca_list, fusion_z_mu, fusion_z_sca,mask=inc_V_ind )

        All_preds = torch.cat([All_preds,pred],dim=0)

        if epoch<args.pre_epochs:
            loss_CL_views = 0

            loss = loss_CL_views
            assert torch.sum(torch.isnan(loss)).item() == 0
            assert torch.sum(torch.isinf(loss)).item() == 0
        else:

            loss_CL = loss_model.weighted_BCE_loss(pred,label,inc_L_ind)

            loss_VGAE_loglik = nn.functional.binary_cross_entropy(A_pred.view(-1), adj.float().view(-1).to('cuda:0'))

            kl_divergence = 0.5 / A_pred.size(0) * (1 + torch.log(label_embedding_var) - label_embedding_mean ** 2 - label_embedding_var ).sum(1).mean()

            cohr_loss_consist = loss_model.corherent_loss_specific(consisit_uniview_mu_list, consisit_uniview_sca_list, mask=inc_V_ind)

            cohr_loss_specific = loss_model.corherent_loss_specific(specific_uniview_mu_list,specific_uniview_sca_list ,mask=inc_V_ind)

            loss_mse_specific = 0
            for v in range(len(data)):
                loss_mse_specific += loss_model.weighted_wmse_loss_sum(data[v],specific_xr_list[v],inc_V_ind[:,v],reduction='mean')
            loss_mse_specific = torch.mean(loss_mse_specific/torch.sum(inc_V_ind,dim=1))

            loss = loss_CL + loss_mse_specific*args.beta  + cohr_loss_specific*args.beta  +cohr_loss_consist*args.beta + loss_VGAE_loglik*args.gamma - kl_divergence*args.gamma

        opt.zero_grad()
        loss.backward()
        if isinstance(sche,CosineAnnealingWarmRestarts):
            sche.step(epoch + i / len(loader))
        
        opt.step()
        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    if isinstance(sche,StepLR):
        sche.step()
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}'.format(
                        epoch,   batch_time=batch_time,
                        data_time=data_time, losses=losses))
    return losses,model_second,model_first,All_preds,label_embedding

def test_second(loader,model_first, model_second, loss_model, epoch,logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    total_labels = []
    total_preds = []
    model_first.eval()
    model_second.eval()
    end = time.time()
    for i, (data, label, inc_V_ind, inc_L_ind) in enumerate(loader):
        data=[v_data.to('cuda:0') for v_data in data]
        inc_V_ind = inc_V_ind.float().to('cuda:0')

        z_sample, consisit_uniview_mu_list, consisit_uniview_sca_list, fusion_z_mu, fusion_z_sca, view_sample_list= model_first(data, mask=inc_V_ind)
        _, _, _, _, _, _, _, _, pred, _, _, _, _ = model_second(data, z_sample, view_sample_list, consisit_uniview_mu_list, consisit_uniview_sca_list, fusion_z_mu, fusion_z_sca, mask=inc_V_ind)

        pred = pred.cpu()
        total_labels = np.concatenate((total_labels,label.numpy()),axis=0) if len(total_labels)>0 else label.numpy()
        total_preds = np.concatenate((total_preds,pred.detach().numpy()),axis=0) if len(total_preds)>0 else pred.detach().numpy()
        
        loss=loss_model.weighted_BCE_loss(pred,label,inc_L_ind)

        losses.update(loss.item())
        batch_time.update(time.time()- end)
        end = time.time()
    total_labels=np.array(total_labels)
    total_preds=np.array(total_preds)

    evaluation_results=evaluation.do_metric(total_preds,total_labels)
    logger.info('Epoch:[{0}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Loss {losses.avg:.3f}\t'
                  'AP {ap:.3f}\t'
                  'HL {hl:.3f}\t'
                  'RL {rl:.3f}\t'
                  'AUC {auc:.3f}\t'.format(
                        epoch,   batch_time=batch_time,
                        losses = losses,
                        ap=evaluation_results[0], 
                        hl=evaluation_results[1],
                        rl=evaluation_results[2],
                        auc=evaluation_results[3]
                        ))
    return evaluation_results

def main(args,file_path):
    data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view.mat')
    fold_data_path = osp.join(args.root_dir, args.dataset, args.dataset+'_six_view_MaskRatios_' + str(
                                args.mask_view_ratio) + '_LabelMaskRatio_' +
                                str(args.mask_label_ratio) + '_TraindataRatio_' + 
                                str(args.training_sample_ratio) + '.mat')
    
    folds_num = args.folds_num
    folds_results = [AverageMeter() for i in range(9)]
    if args.logs:
        logfile = osp.join(args.logs_dir,args.name+args.dataset+'_V_' + str(
                                    args.mask_view_ratio) + '_L_' + str(args.mask_label_ratio) + '_T_' + str(args.training_sample_ratio) + '_'+
                           str(args.alpha)+'_'+str(args.beta)+'_'+str(args.gamma)+'.txt')
    else:
        logfile=None
    logger = utils.setLogger(logfile)
    label_Inp_list = []
    for fold_idx in range(folds_num):
        fold_idx=fold_idx
        train_dataloder,train_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='train',batch_size=args.batch_size,shuffle = True,num_workers=4)
        test_dataloder,test_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,val_ratio=0.15,fold_idx=fold_idx,mode='test',batch_size=args.batch_size,num_workers=4)
        val_dataloder,val_dataset = MLdataset.getIncDataloader(data_path, fold_data_path,training_ratio=args.training_sample_ratio,fold_idx=fold_idx,mode='val',batch_size=args.batch_size,num_workers=4)
        d_list = train_dataset.d_list
        classes_num = train_dataset.classes_num
        labels = torch.tensor(train_dataset.cur_labels).float().to('cuda:0')

        dep_graph = torch.matmul(labels.T,labels)
        dep_graph = dep_graph/(torch.diag(dep_graph).unsqueeze(1)+1e-10)
        dep_graph.fill_diagonal_(fill_value=0.)
        pri_c = train_dataset.cur_labels.sum(axis=0)/train_dataset.cur_labels.shape[0]
        pri_c = torch.tensor(pri_c).cuda()


        adj = dep_graph
        adj_target = binary_adj(adj)
        inp = torch.eye(classes_num, dtype=torch.float).to('cuda:0')

        model_first = VAE_consist(d_list,args.z_dim,classes_num)
        model_first = model_first.to(torch.device('cuda' if torch.cuda.is_available()
                                      else 'cpu'))
        model_second = get_model_second(d_list,num_classes=classes_num,z_dim=args.z_dim,adj=adj,inp=inp,rand_seed=0) #得到模型

        loss_model = Loss()
        optimizer_first =  Adam(model_first.parameters(), lr=args.lr)
        scheduler = None

        logger.info('train_data_num:'+str(len(train_dataset))+'  test_data_num:'+str(len(test_dataset))+'   fold_idx:'+str(fold_idx))
        print(args)
        best_loss = 1000000000
        total_losses = AverageMeter()
        best_model_first_dict = {'model':model_first.state_dict(),'epoch':0}
        best_epoch = 0
        static_res = 0
        #first_stage_train
        for epoch in range(args.epochs_first):
            if epoch==0:
                All_preds = None

            train_losses,model_first = train_first(train_dataloder,model_first,loss_model,optimizer_first,scheduler,epoch,logger)

            if epoch>=args.pre_epochs:

                if train_losses.avg <= best_loss:
                    best_loss = train_losses.avg
                    best_model_first_dict['model'] = copy.deepcopy(model_first.state_dict())
                    best_model_first_dict['epoch'] = epoch
                    best_epoch = epoch
                train_losses_last = train_losses
                total_losses.update(train_losses.sum)
        model_first.load_state_dict(best_model_first_dict['model'])
        print("epoch", best_model_first_dict['epoch'])
        logger.info('final_first: fold_idx:{} best_epoch:{}\n'.format(fold_idx,best_epoch))
        pertrained_Encoder = VAE_consist_encoder(model_first)

        logger.info('train_data_num:' + str(len(train_dataset)) + '  test_data_num:' + str(
            len(test_dataset)) + '   fold_idx:' + str(fold_idx))
        print(args)
        static_res = 0
        epoch_results = [AverageMeter() for i in range(9)]
        total_losses = AverageMeter()
        train_losses_last = AverageMeter()
        best_epoch = 0
        best_model_second_dict = {'model': model_second.state_dict(), 'epoch': 0}

        param_groups = [
            {'params': pertrained_Encoder.parameters(), 'lr': args.lr / 10},
            {'params': list(model_second.get_VGAE_para()), 'lr': args.lr * 10},
            {'params': list(model_second.get_other_para()), 'lr': args.lr}]
        optimizer_second = Adam(param_groups)
        #second_stage_train
        for epoch in range(args.epochs_second):
            if epoch == 0:
                All_preds = None

            train_losses, model_second,pertrained_Encoder, All_preds, label_emb_sample = train_second(train_dataloder, pertrained_Encoder , model_second , loss_model,
                                                                     optimizer_second, scheduler, epoch, logger,adj_target)
            label_InP = label_emb_sample.mm(label_emb_sample.t())

            if epoch >= args.pre_epochs:
                val_results = test_second(val_dataloder,pertrained_Encoder, model_second, loss_model, epoch, logger)

                if val_results[0] * 0.25 + val_results[2] * 0.25 + val_results[3] * 0.25 >= static_res:
                    static_res = val_results[0] * 0.25 + val_results[2] * 0.25 + val_results[3] * 0.25
                    best_model_second_dict['model'] = copy.deepcopy(model_second.state_dict())
                    best_model_second_dict['model_first'] = copy.deepcopy(pertrained_Encoder.state_dict())
                    best_model_second_dict['epoch'] = epoch
                    best_epoch = epoch
                train_losses_last = train_losses
                total_losses.update(train_losses.sum)
        model_second.load_state_dict(best_model_second_dict['model'])
        pertrained_Encoder.load_state_dict(best_model_second_dict['model_first'])
        print("epoch", best_model_second_dict['epoch'])
        test_results = test_second(test_dataloder,pertrained_Encoder, model_second, loss_model, epoch, logger)
        label_Inp_list.append(label_InP.cpu().detach().numpy())
        logger.info(
            'final: fold_idx:{} best_epoch:{}\t best:ap:{:.4}\t HL:{:.4}\t RL:{:.4}\t AUC_me:{:.4}\n'.format(fold_idx,best_epoch,
                                                                        test_results[0],test_results[1],test_results[2],test_results[3]))

        for i in range(9):
            folds_results[i].update(test_results[i])
        if args.save_curve:
            np.save(osp.join(args.curve_dir, args.dataset + '_V_' + str(args.mask_view_ratio) + '_L_' + str(
                args.mask_label_ratio)) + '_' + str(fold_idx) + '.npy',
                    np.array(list(zip(epoch_results[0].vals, train_losses.vals))))

    folder_path = "mid_res"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(f"mid_res/label_InP_{args.dataset}.npy", np.array(label_Inp_list))

    file_handle = open(file_path, mode='a')
    if os.path.getsize(file_path) == 0:
        file_handle.write(
            'AP 1-HL 1-RL AUCme 1-oneE 1-Cov macAUC macro_f1 micro_f1 lr alpha beta gamma\n')
    res_list = [str(round(res.avg, 3)) + '+' + str(round(res.std, 3)) for res in folds_results]
    res_list.extend([str(args.lr), str(args.alpha), str(args.beta), str(args.gamma)])
    res_str = ' '.join(res_list)
    file_handle.write(res_str)
    file_handle.write('\n')
    file_handle.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--logs', default=False, type=bool)
    parser.add_argument('--records-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'final_records'))
    parser.add_argument('--file-path', type=str, metavar='PATH',  default='')
    parser.add_argument('--config-path', type=str, metavar='PATH',  default=osp.join(working_dir, 'config'))
    parser.add_argument('--root-dir', type=str, metavar='PATH', default='data/')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--datasets', type=list, default=['corel5k'])
    parser.add_argument('--mask-view-ratio', type=float, default=0.5)
    parser.add_argument('--mask-label-ratio', type=float, default=0.5)
    parser.add_argument('--training-sample-ratio', type=float, default=0.7)
    parser.add_argument('--folds-num', default=10, type=int)
    parser.add_argument('--weights-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'weights'))
    parser.add_argument('--curve-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'curves'))
    parser.add_argument('--save-curve', default=False, type=bool)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--workers', default=8, type=int)
    
    parser.add_argument('--name', type=str, default='final_')
    # Optimization args
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--pre_epochs', type=int, default=0)
    parser.add_argument('--epochs_first', type=int, default=100)
    parser.add_argument('--epochs_second', type=int, default=50)
    # Training args
    parser.add_argument('--z_dim', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--alpha', type=float, default=1e-1)
    parser.add_argument('--beta', type=float, default=1e-2)
    parser.add_argument('--gamma', type=float, default=1e-3)

    args = parser.parse_args()
    args = get_config(args, osp.join(args.config_path, args.dataset + '.yaml'))
    Init_random_seed(args.seed)

    if args.logs:
        if not os.path.exists(args.logs_dir):
            os.makedirs(args.logs_dir)
    if args.save_curve:
        if not os.path.exists(args.curve_dir):
            os.makedirs(args.curve_dir)
    if True:
        if not os.path.exists(args.records_dir):
            os.makedirs(args.records_dir)
    if args.lr >= 0.01:
        args.momentumkl = 0.90

    file_path = osp.join(args.records_dir,args.name+args.dataset+'_VM_' + str(
                    args.mask_view_ratio) + '_LM_' +
                    str(args.mask_label_ratio) + '_T_' +
                    str(args.training_sample_ratio) + '.txt')
    args.file_path = file_path
    main(args,file_path)