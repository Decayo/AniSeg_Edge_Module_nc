from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from config import config
from dataloader import get_train_loader
from network import Network
from dataloader import AniSeg
from utils.init_func import init_weight, group_weight
from engine.lr_policy import WarmUpPolyLR
from engine.engine import Engine
from seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from c2pe_loss.c2pe_criterion import CriterionAll
from c2pe_loss.target_generation import generate_edge_tensor
# from seg_opr.sync_bn import DataParallelModel, Reduce, BatchNorm2d
from tensorboardX import SummaryWriter

scaler = torch.cuda.amp.GradScaler()
# try:
#     from apex.parallel import DistributedDataParallel, SyncBatchNorm
# except ImportError:
#     raise ImportError(
#         "Please install apex from https://www.github.com/nvidia/apex .")

try:
    from azureml.core import Run
    azure = True
    run = Run.get_context()
except:
    azure = False

parser = argparse.ArgumentParser()

os.environ['MASTER_PORT'] = '169711'

if os.getenv('debug') is not None:
    is_debug = os.environ['debug']
else:
    is_debug = False


'''
For CutMix.
'''
import mask_gen
from custom_collate import SegCollate
mask_generator = mask_gen.BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                           random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                           prop_by_area=not config.cutmix_boxmask_by_size, within_bounds=not config.cutmix_boxmask_outside_bounds,
                                           invert=not config.cutmix_boxmask_no_invert)

add_mask_params_to_batch = mask_gen.AddMaskParamsToBatch(
    mask_generator
)
collate_fn = SegCollate()
mask_collate_fn = SegCollate(batch_aug_fn=add_mask_params_to_batch)
if __name__ == '__main__':
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        cudnn.benchmark = True

        seed = config.seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # data loader + unsupervised data loader

        train_loader, train_sampler = get_train_loader(engine, AniSeg, train_source=config.train_source, \
                                                    unsupervised=False, collate_fn=collate_fn)
        unsupervised_train_loader, unsupervised_train_sampler_0 = get_train_loader(engine, AniSeg, \
                    train_source=config.unsup_source, unsupervised=True, collate_fn=mask_collate_fn)

        if engine.distributed and (engine.local_rank == 0):
            tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + '/tb'
            logger = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)

        # config network and criterion
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        criterion_csst = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        c2pe_criterion = CriterionAll(lambda_1=config.lambda_s, lambda_2=config.lambda_e, lambda_3=config.lambda_c,
                        num_classes=config.num_classes)
        
        
        if engine.distributed:
            BatchNorm2d = SyncBatchNorm
        else :
            print("----not distributed")
            BatchNorm2d = nn.BatchNorm2d
        model = Network(config.num_classes, criterion=criterion,
                        pretrained_model=config.pretrained_model,
                        norm_layer=BatchNorm2d)
        # init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
        #             BatchNorm2d, config.bn_eps, config.bn_momentum,
        #             mode='fan_in', nonlinearity='relu')
        # init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
        #             BatchNorm2d, config.bn_eps, config.bn_momentum,
        #             mode='fan_in', nonlinearity='relu')

        # set the lr
        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr * 1

        # define two optimizers
        params_list_l = []
        params_list_l = group_weight(params_list_l, model.branch1.backbone,
                                BatchNorm2d, base_lr)
        # for module in model.branch1.business_layer:
        #     params_list_l = group_weight(params_list_l, module, BatchNorm2d,
        #                             base_lr)        # head lr * 10

        optimizer_l = torch.optim.SGD(params_list_l,
                                    lr=base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)

        params_list_r = []
        params_list_r = group_weight(params_list_r, model.branch2.backbone,
                                BatchNorm2d, base_lr)
        # for module in model.branch2.business_layer:
        #     params_list_r = group_weight(params_list_r, module, BatchNorm2d,
        #                             base_lr)        # head lr * 10

        optimizer_r = torch.optim.SGD(params_list_r,
                                    lr=base_lr,
                                    momentum=config.momentum,
                                    weight_decay=config.weight_decay)



        # config lr policy
        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

        if engine.distributed:
            print('distributed !!')
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(model)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = nn.DataParallel(model, device_ids=engine.devices)
            model.to(device)

        engine.register_state(dataloader=train_loader, model=model,
                            optimizer_l=optimizer_l, optimizer_r=optimizer_r)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        model.train()
        print('begin train')

        for epoch in range(engine.state.epoch, config.nepochs):
            if engine.distributed:
                train_sampler.set_epoch(epoch)
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

            if is_debug:
                pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
            else:
                pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)

            dataloader = iter(train_loader)
            unsupervised_dataloader= iter(unsupervised_train_loader)

            sum_loss_sup = 0
            sum_loss_sup_r = 0
            sum_cps = 0

            ''' supervised part '''
            for idx in pbar:
                optimizer_l.zero_grad()
                optimizer_r.zero_grad()
                engine.update_iteration(epoch, idx)
                start_time = time.time()

                minibatch = dataloader.next()
                unsup_minibatch = unsupervised_dataloader.next()
                imgs = minibatch['data']
                gts = minibatch['label']
                unsup_imgs = unsup_minibatch['data']

                imgs = imgs.cuda(non_blocking=True)
                gts = gts.cuda(non_blocking=True)
                unsup_imgs = unsup_imgs.cuda(non_blocking=True)
                h, w = imgs.size(2), imgs.size(3)
                # unsupervised loss on model/branch#1
                with torch.cuda.amp.autocast():
                    # unsupervised loss on both models
                    pred_unsup_l = model(unsup_imgs, step=1)
                    pred_unsup_r = model(unsup_imgs, step=2)
                    # supervised loss on both models
                    l_c2pe_output = model(imgs, step=1)
                    r_c2pe_output = model(imgs, step=2)
                    
                    #loss_sup = criterion(sup_pred_l, gts)
                    edges = generate_edge_tensor(gts)
                    pred_unsup_l = F.interpolate(input=pred_unsup_l[0][-1], size=(h, w),
                                        mode='bilinear', align_corners=True)
                    pred_unsup_r = F.interpolate(input=pred_unsup_r[0][-1], size=(h, w),
                                        mode='bilinear', align_corners=True)                    
                    pred_sup_l =  F.interpolate(input=l_c2pe_output[0][-1], size=(h, w),
                                        mode='bilinear', align_corners=True)
                    pred_sup_r = F.interpolate(input=r_c2pe_output[0][-1], size=(h, w),
                                        mode='bilinear', align_corners=True)
                    
                    pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
                    pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
                    max_probs_l, max_l = torch.max(pred_l, dim=1)
                    max_probs_r, max_r = torch.max(pred_r, dim=1)
                    max_l = max_l.long()
                    max_r = max_r.long()
                    mask_l = max_probs_l.ge(config.prob_threshold).float()
                    mask_r = max_probs_r.ge(config.prob_threshold).float()
                    #use FixMatch probs mask 
                    cps_loss = (F.cross_entropy(pred_l, max_r,
                                  reduction='none',ignore_index=255) * mask_r).mean() + (F.cross_entropy(pred_r, max_l,
                                  reduction='none',ignore_index=255) * mask_l).mean()
                    cps_loss = cps_loss * config.cps_weight
                    loss_sup_l_c2pe = c2pe_criterion(l_c2pe_output,[gts.type(torch.cuda.LongTensor),
                                                                    edges.type(torch.cuda.LongTensor),
                                                                    None,None])
                    #dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
                    #loss_sup = loss_sup 

                    #loss_sup_r = criterion(sup_pred_r, gts)
                    #dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
                    loss_sup_r_c2pe = c2pe_criterion(r_c2pe_output,[gts.type(torch.cuda.LongTensor),
                                                                    edges.type(torch.cuda.LongTensor),
                                                                    None,None])
                    
                    current_idx = epoch * config.niters_per_epoch + idx
                    lr = lr_policy.get_lr(current_idx)

                    # print(len(optimizer.param_groups))
                    optimizer_l.param_groups[0]['lr'] = lr
                    optimizer_l.param_groups[1]['lr'] = lr
                    for i in range(2, len(optimizer_l.param_groups)):
                        optimizer_l.param_groups[i]['lr'] = lr
                    optimizer_r.param_groups[0]['lr'] = lr
                    optimizer_r.param_groups[1]['lr'] = lr
                    for i in range(2, len(optimizer_r.param_groups)):
                        optimizer_r.param_groups[i]['lr'] = lr

                    loss = cps_loss + loss_sup_l_c2pe + loss_sup_r_c2pe
                scaler.scale(loss).backward()
                #loss.backward()
                scaler.step(optimizer_l)
                scaler.step(optimizer_r)
                # optimizer_l.step()
                # optimizer_r.step()
                scaler.update()

                print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
                            + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
                            + ' lr=%.2e' % lr \
                            + ' loss_sup=%.2f' % loss_sup_l_c2pe.item() \
                            + ' loss_sup_r=%.2f' % loss_sup_r_c2pe.item() \
                            + ' loss_cps=%.4f' % cps_loss.item()

                sum_loss_sup += loss_sup_l_c2pe.item()
                sum_loss_sup_r += loss_sup_r_c2pe.item()
                sum_cps += cps_loss.item()
                pbar.set_description(print_str, refresh=False)

                end_time = time.time()
            f=open(r'/content/drive/MyDrive/_Anime_paper_/0429_nc/logloss.txt', "a+")
            f.write('epoch %d\r\n'%epoch)
            f.write(str(sum_loss_sup / len(pbar))+'\r\n')
            f.write(str(sum_loss_sup_r / len(pbar))+'\r\n')
            f.write(str(sum_cps / len(pbar))+'\r\n')
            f.write('_______________ \r\n')
            f.close()
            print('loss:')
            print(sum_loss_sup / len(pbar))
            print(sum_loss_sup_r / len(pbar))
            print(sum_cps / len(pbar))
            if engine.distributed and (engine.local_rank == 0):
                logger.add_scalar('train_loss_sup', sum_loss_sup / len(pbar), epoch)
                logger.add_scalar('train_loss_sup_r', sum_loss_sup_r / len(pbar), epoch)
                logger.add_scalar('train_loss_cps', sum_cps / len(pbar), epoch)

            if azure and engine.local_rank == 0:
                run.log(name='Supervised Training Loss', value=sum_loss_sup / len(pbar))
                run.log(name='Supervised Training Loss right', value=sum_loss_sup_r / len(pbar))
                run.log(name='Supervised Training Loss CPS', value=sum_cps / len(pbar))
            if (epoch % config.snapshot_iter == 0) or (epoch == config.nepochs - 1):
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(config.snapshot_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.snapshot_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)