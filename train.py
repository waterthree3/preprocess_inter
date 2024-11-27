import os
import math
import time

import torchvision
import yaml
import argparse
import torch
import torch.optim as optim
import os.path as op
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import utils  # my tool box
import dataset
from net_stdf import MFVQE
from net_pwc import Network
from net_pwc import backwarp
from mv_search import *

def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option_R3_mfqev2_4G.yml', 
        help='Path to option YAML file.'
        )
    parser.add_argument(
        '--local_rank', type=int, default=0, 
        help='Distributed launcher requires.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path
    opts_dict['train']['rank'] = args.local_rank

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    
    opts_dict['train']['num_gpu'] = torch.cuda.device_count()
    if opts_dict['train']['num_gpu'] > 1:
        opts_dict['train']['is_dist'] = True
    else:
        opts_dict['train']['is_dist'] = False
    
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )

    return opts_dict

netPWC = None
def prediction(tenOne, tenTwo):
    global netPWC
    netPWC = Network().cuda().eval()
    for param in netPWC.parameters():
        param.requires_grad = False
    # end
    tenOne = tenOne.expand(-1, 3, -1, -1)  # 使用 expand
    tenTwo = tenTwo.expand(-1, 3, -1, -1)  # 使用 expand
    assert(tenOne.shape[2] == tenTwo.shape[2])
    assert(tenOne.shape[3] == tenTwo.shape[3])

    intWidth = tenOne.shape[3]
    intHeight = tenOne.shape[2]
    batchsize = tenOne.shape[0]
    channel = tenOne.shape[1]

    # assert(intWidth == 1024) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
    # assert(intHeight == 436) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

    tenPreprocessedOne = tenOne.cuda().view(batchsize, channel, intHeight, intWidth)
    tenPreprocessedTwo = tenTwo.cuda().view(batchsize, channel, intHeight, intWidth)

    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedOne = torch.nn.functional.interpolate(input=tenPreprocessedOne, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
    tenPreprocessedTwo = torch.nn.functional.interpolate(input=tenPreprocessedTwo, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

    tenFlow = torch.nn.functional.interpolate(input=netPWC(tenPreprocessedOne, tenPreprocessedTwo), size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    warp_img = backwarp(tenTwo, tenFlow)
    return warp_img


class MCTF_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction='none')
        self.l1_none = nn.L1Loss(reduction='none')
        self.l1 = nn.L1Loss()

    def forward(self, lq_data_first, lq_data_second, warp_img_first, warp_img_second, gt_data, enhanced_data):
        out = {}

        out["warp_first_loss"] = self.l1(lq_data_first, warp_img_first)
        # out["warp_first_loss"] = self.mse_none(lq_data_first, warp_img_first)
        out["warp_second_loss"] = self.l1(lq_data_second, warp_img_second)
        # out["warp_second_loss"] = self.mse_none(lq_data_second, warp_img_second)

        # ============根据每一个batch平均值卡对应batch的阈值===========
        # batch_means_first = out["warp_first_loss"].mean(dim=(1, 2, 3), keepdim=True)
        # mask_first = (out["warp_first_loss"] > 1.5 * batch_means_first).float()
        # batch_means_second = out["warp_second_loss"].mean(dim=(1, 2, 3), keepdim=True)
        # mask_second = (out["warp_second_loss"] > 1.5 * batch_means_second).float()
        # =========================================================

        # # #=========验证测试=============
        # img_file_path = rf"D:\MachineLearning_Project\stdf-pytorch\exp\MFQEv2_pwc_opt_orign_weight_0.5_mask_opt_v2\pic"
        # torchvision.utils.save_image(mask_first, img_file_path + "/mask_first.png", nrow=1)
        # torchvision.utils.save_image(out["warp_first_loss"], img_file_path + "/warp_first_loss.png", nrow=1)
        # torchvision.utils.save_image(warp_img_first, img_file_path + "/warp_first_test.png", nrow=1)
        # # #=============================

        # out["ground_loss"] = self.mse(gt_data, enhanced_data)
        out["ground_loss"] = self.l1(gt_data, enhanced_data)

        # out["loss"] = (out["warp_first_loss"] * (1 - mask_first)).mean() + (out["warp_second_loss"] * (1 - mask_second)).mean() + 0.25 *
        # out["loss"] = (out["warp_first_loss"] * (1 - mask_first)).mean() + (out["warp_second_loss"] * (1 - mask_second)).mean()
        out["loss"] = (out["warp_first_loss"]) + (out["warp_second_loss"]) + 0.5 * out["ground_loss"]

        return out["loss"]


class MCTF_Loss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.mse_none = nn.MSELoss(reduction='none')
        self.l1_none = nn.L1Loss(reduction='none')
        self.l1 = nn.L1Loss()

    def forward(self, first_loss, second_loss, gt_data, enhanced_data):
        out = {}

        out["ground_loss"] = self.l1(gt_data, enhanced_data)
        N, C, H, W = gt_data.shape
        num_pixels = N * H * W
        out["first_sum_loss"] = torch.sum(first_loss) / num_pixels
        out["second_sum_loss"] = torch.sum(second_loss) / num_pixels

        out["loss"] = 0.5 * out["ground_loss"] + out["first_sum_loss"] + out["second_sum_loss"]
        return out["loss"]


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    rank = opts_dict['train']['rank']
    unit = opts_dict['train']['criterion']['unit']
    num_iter = int(opts_dict['train']['num_iter'])
    interval_print = int(opts_dict['train']['interval_print'])
    interval_val = int(opts_dict['train']['interval_val'])
    
    # ==========
    # init distributed training
    # ==========

    if opts_dict['train']['is_dist']:
        utils.init_dist(
            local_rank=rank, 
            backend='nccl'
            )

    # TO-DO: load resume states if exists
    pass

    # ==========
    # create logger
    # ==========

    if rank == 0:
        log_dir = op.join("exp", opts_dict['train']['exp_name'])
        if not op.exists(log_dir):
            utils.mkdir(log_dir)
        log_fp = open(opts_dict['train']['log_path'], 'w')

        # log all parameters
        msg = (
            f"{'<' * 10} Hello {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]\n"
            f"\n{'<' * 10} Options {'>' * 10}\n"
            f"{utils.dict2str(opts_dict)}"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # TO-DO: init tensorboard
    # ==========

    pass
    
    # ==========
    # fix random seed
    # ==========

    seed = opts_dict['train']['random_seed']
    # >I don't know why should rs + rank
    utils.set_random_seed(seed + rank)

    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create train and val data prefetchers
    # ==========
    
    # create datasets
    train_ds_type = opts_dict['dataset']['train']['type']
    val_ds_type = opts_dict['dataset']['val']['type']
    radius = opts_dict['network']['radius']
    assert train_ds_type in dataset.__all__, \
        "Not implemented!"
    assert val_ds_type in dataset.__all__, \
        "Not implemented!"
    train_ds_cls = getattr(dataset, train_ds_type)
    val_ds_cls = getattr(dataset, val_ds_type)
    train_ds = train_ds_cls(
        opts_dict=opts_dict['dataset']['train'], 
        radius=radius
        )
    val_ds = val_ds_cls(
        opts_dict=opts_dict['dataset']['val'], 
        radius=radius
        )

    # create datasamplers
    train_sampler = utils.DistSampler(
        dataset=train_ds, 
        num_replicas=opts_dict['train']['num_gpu'], 
        rank=rank, 
        ratio=opts_dict['dataset']['train']['enlarge_ratio']
        )
    val_sampler = None  # no need to sample val data

    # create dataloaders
    train_loader = utils.create_dataloader(
        dataset=train_ds, 
        opts_dict=opts_dict, 
        sampler=train_sampler, 
        phase='train',
        seed=opts_dict['train']['random_seed']
        )
    val_loader = utils.create_dataloader(
        dataset=val_ds, 
        opts_dict=opts_dict, 
        sampler=val_sampler, 
        phase='val'
        )
    assert train_loader is not None

    batch_size = opts_dict['dataset']['train']['batch_size_per_gpu'] * \
        opts_dict['train']['num_gpu']  # divided by all GPUs
    num_iter_per_epoch = math.ceil(len(train_ds) * \
        opts_dict['dataset']['train']['enlarge_ratio'] / batch_size)
    num_epoch = math.ceil(num_iter / num_iter_per_epoch)
    val_num = len(val_ds)
    
    # create dataloader prefetchers
    tra_prefetcher = utils.CPUPrefetcher(train_loader)
    val_prefetcher = utils.CPUPrefetcher(val_loader)

    # ==========
    # create model
    # ==========

    model = MFVQE(opts_dict=opts_dict['network'])

    model = model.to(rank)
    if opts_dict['train']['is_dist']:
        model = DDP(model, device_ids=[rank])

    # # load pre_trained generator
    # ckp_path = opts_dict['train']['pre_train']['load_path']
    # checkpoint = torch.load(ckp_path)
    # state_dict = checkpoint['state_dict']
    # model.load_state_dict(state_dict)

    """
    # load pre-trained generator
    ckp_path = opts_dict['network']['stdf']['load_path']
    checkpoint = torch.load(ckp_path)
    state_dict = checkpoint['state_dict']
    if ('module.' in list(state_dict.keys())[0]) and (not opts_dict['train']['is_dist']):  # multi-gpu pre-trained -> single-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f'loaded from {ckp_path}')
    elif ('module.' not in list(state_dict.keys())[0]) and (opts_dict['train']['is_dist']):  # single-gpu pre-trained -> multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k  # add module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        print(f'loaded from {ckp_path}')
    else:  # the same way of training
        model.load_state_dict(state_dict)
        print(f'loaded from {ckp_path}')
    """

    # ==========
    # define loss func & optimizer & scheduler & scheduler & criterion
    # ==========

    # define loss func
    # assert opts_dict['train']['loss'].pop('type') == 'CharbonnierLoss', \
    #     "Not implemented."
    # loss_func = utils.CharbonnierLoss(**opts_dict['train']['loss'])
    loss_func = MCTF_Loss2()

    # define optimizer
    assert opts_dict['train']['optim'].pop('type') == 'Adam', \
        "Not implemented."
    optimizer = optim.Adam(
        model.parameters(), 
        **opts_dict['train']['optim']
        )

    # define scheduler
    if opts_dict['train']['scheduler']['is_on']:
        assert opts_dict['train']['scheduler'].pop('type') == \
            'CosineAnnealingRestartLR', "Not implemented."
        del opts_dict['train']['scheduler']['is_on']
        scheduler = utils.CosineAnnealingRestartLR(
            optimizer, 
            **opts_dict['train']['scheduler']
            )
        opts_dict['train']['scheduler']['is_on'] = True

    # define criterion
    assert opts_dict['train']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()

    #

    start_iter = 0  # should be restored
    start_epoch = start_iter // num_iter_per_epoch

    # display and log
    if rank == 0:
        msg = (
            f"\n{'<' * 10} Dataloader {'>' * 10}\n"
            f"total iters: [{num_iter}]\n"
            f"total epochs: [{num_epoch}]\n"
            f"iter per epoch: [{num_iter_per_epoch}]\n"
            f"val sequence: [{val_num}]\n"
            f"start from iter: [{start_iter}]\n"
            f"start from epoch: [{start_epoch}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    # ==========
    # evaluate original performance, e.g., PSNR before enhancement
    # ==========

    vid_num = val_ds.get_vid_num()
    if opts_dict['train']['pre-val'] and rank == 0:
        msg = f"\n{'<' * 10} Pre-evaluation {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        per_aver_dict = {}
        for i in range(vid_num):
            per_aver_dict[i] = utils.Counter()
        pbar = tqdm(
                total=val_num, 
                ncols=opts_dict['train']['pbar_len']
                )

        # fetch the first batch
        val_prefetcher.reset()
        val_data = val_prefetcher.next()

        while val_data is not None:
            # get data
            gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = val_data['lq'].to(rank)  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!
            b, _, _, _, _  = lq_data.shape
            
            # eval
            batch_perf = np.mean(
                [criterion(lq_data[i,radius,...], gt_data[i]) for i in range(b)]
                )  # bs must be 1!
            
            # log
            per_aver_dict[index_vid].accum(volume=batch_perf)

            # display
            pbar.set_description(
                "{:s}: [{:.3f}] {:s}".format(name_vid, batch_perf, unit)
                )
            pbar.update()

            # fetch next batch
            val_data = val_prefetcher.next()

        pbar.close()

        # log
        ave_performance = np.mean([
            per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
            ])
        msg = "> ori performance: [{:.3f}] {:s}".format(ave_performance, unit)
        print(msg)
        log_fp.write(msg + '\n')
        log_fp.flush()

    if opts_dict['train']['is_dist']:
        torch.distributed.barrier()  # all processes wait for ending

    if rank == 0:
        msg = f"\n{'<' * 10} Training {'>' * 10}"
        print(msg)
        log_fp.write(msg + '\n')

        # create timer
        total_timer = utils.Timer()  # total tra + val time of each epoch

    # ==========
    # start training + validation (test)
    # ==========

    model.train()
    num_iter_accum = start_iter
    for current_epoch in range(start_epoch, num_epoch + 1):
        # shuffle distributed subsamplers before each epoch
        if opts_dict['train']['is_dist']:
            train_sampler.set_epoch(current_epoch)

        # fetch the first batch
        tra_prefetcher.reset()
        train_data = tra_prefetcher.next()

        # train this epoch
        while train_data is not None:

            # over sign
            num_iter_accum += 1
            if num_iter_accum > num_iter:
                break

            # get data
            gt_data = train_data['gt'].to(rank)  # (B [RGB] H W)
            lq_data = train_data['lq'].to(rank)  # (B T [RGB] H W)
            b, _, c, _, _  = lq_data.shape
            input_data = torch.cat(
                [lq_data[:,:,i,...] for i in range(c)], 
                dim=1
                )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
            enhanced_data = model(input_data)
            # 提取第一个通道并重新组成新的张量
            lq_data_second = lq_data[:, 2, :, :, :]
            lq_data_first = lq_data[:, 0, :, :, :]  # 保持形状为 (16, 1, 128, 128)
            #gt_data_first = gt_data
            # warp_img_second = prediction(lq_data_second, enhanced_data)
            # warp_img_first = prediction(lq_data_first, enhanced_data)
            second_error = mctf_mv_search(enhanced_data, lq_data_second)
            first_error = mctf_mv_search(enhanced_data, lq_data_first)
            # warp_img_second = warp_img_second[:, :1, :, :]
            # warp_img_first = warp_img_first[:, :1, :, :]
            # #=========验证测试=============
            # img_file_path = rf"D:\MachineLearning_Project\stdf-pytorch\exp\MFQEv2_pwc_opt_orign_weight_0.5_mask_opt\pic"
            # torchvision.utils.save_image(warp_img_second, img_file_path + "/warp_img_second.png", nrow=1)
            # torchvision.utils.save_image(lq_data_second, img_file_path + "/lq_data_second.png", nrow=1)
            # torchvision.utils.save_image(enhanced_data, img_file_path + "/enhanced_data.png", nrow=1)
            # torchvision.utils.save_image(gt_data, img_file_path + "/gt_data.png", nrow=1)
            # #=============================
            # get loss
            optimizer.zero_grad()  # zero grad
            # loss = torch.mean(torch.stack(
            #     [loss_func(enhanced_data[i], gt_data[i]) for i in range(b)]
            #     ))  # cal loss
            # loss = loss_func(lq_data_first, lq_data_second, warp_img_first, warp_img_second, gt_data, enhanced_data)
            loss = loss_func(first_error, second_error, gt_data, enhanced_data)
            loss.backward()  # cal grad
            optimizer.step()  # update parameters

            # update learning rate
            if opts_dict['train']['scheduler']['is_on']:
                scheduler.step()  # should after optimizer.step()

            if (num_iter_accum % interval_print == 0) and (rank == 0):
                # display & log
                lr = optimizer.param_groups[0]['lr']
                loss_item = loss.item()
                msg = (
                    f"iter: [{num_iter_accum}]/{num_iter}, "
                    f"epoch: [{current_epoch}]/{num_epoch - 1}, "
                    "lr: [{:.3f}]x1e-4, loss: [{:.4f}]".format(
                        lr*1e4, loss_item
                        )
                    )
                print(msg)
                log_fp.write(msg + '\n')
            if ((num_iter_accum % interval_val == 0) or \
                (num_iter_accum == num_iter)) and (rank == 0):
                # save model
                checkpoint_save_path = (
                    f"{opts_dict['train']['checkpoint_save_path_pre']}"
                    f"{num_iter_accum}"
                    ".pt"
                    )
                state = {
                    'num_iter_accum': num_iter_accum, 
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 
                    }
                if opts_dict['train']['scheduler']['is_on']:
                    state['scheduler'] = scheduler.state_dict()
                torch.save(state, checkpoint_save_path)
                
                # validation
                with torch.no_grad():
                    per_aver_dict = {}
                    for index_vid in range(vid_num):
                        per_aver_dict[index_vid] = utils.Counter()
                    pbar = tqdm(
                            total=val_num, 
                            ncols=opts_dict['train']['pbar_len']
                            )
                
                    # train -> eval
                    model.eval()

                    # fetch the first batch
                    val_prefetcher.reset()
                    val_data = val_prefetcher.next()
                    
                    while val_data is not None:
                        # get data
                        gt_data = val_data['gt'].to(rank)  # (B [RGB] H W)
                        lq_data = val_data['lq'].to(rank)  # (B T [RGB] H W)
                        index_vid = val_data['index_vid'].item()
                        name_vid = val_data['name_vid'][0]  # bs must be 1!
                        b, _, c, _, _  = lq_data.shape
                        input_data = torch.cat(
                            [lq_data[:,:,i,...] for i in range(c)], 
                            dim=1
                            )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W
                        enhanced_data = model(input_data)  # (B [RGB] H W)

                        #=================opt flow================
                        # 提取第一个通道并重新组成新的张量
                        lq_data_first = lq_data[:, 0, :, :, :]  # 保持形状为 (16, 1, 128, 128)
                        lq_data_second = lq_data[:, 2, :, :, :]
                        # gt_data_first = gt_data
                        # warp_img_first = prediction(lq_data_first, enhanced_data)
                        # warp_img_second = prediction(lq_data_second, enhanced_data)
                        # warp_img_first = warp_img_first[:, :1, :, :]
                        # warp_img_second = warp_img_second[:, :1, :, :]
                        # batch_perf = loss_func(lq_data_first, lq_data_second, warp_img_first, warp_img_second, gt_data,
                        #                  enhanced_data).cpu()
                        batch_perf = 0
                        # eval
                        # batch_perf = np.mean(
                        #     [criterion(enhanced_data[i], gt_data[i]) for i in range(b)]
                        #     ) # bs must be 1!

                        # display
                        pbar.set_description(
                            "{:s}: [{:.6f}] {:s}"
                            .format(name_vid, batch_perf, unit)
                            )
                        pbar.update()

                        # log
                        per_aver_dict[index_vid].accum(volume=batch_perf)

                        # fetch next batch
                        val_data = val_prefetcher.next()
                    
                    # end of val
                    pbar.close()

                    # eval -> train
                    model.train()

                # log
                ave_per = np.mean([
                    per_aver_dict[index_vid].get_ave() for index_vid in range(vid_num)
                    ])
                msg = (
                    "> model saved at {:s}\n"
                    "> ave val per: [{:.6f}] {:s}"
                    ).format(
                        checkpoint_save_path, ave_per, unit
                        )
                print(msg)
                log_fp.write(msg + '\n')
                log_fp.flush()

            if opts_dict['train']['is_dist']:
                torch.distributed.barrier()  # all processes wait for ending

            # fetch next batch
            train_data = tra_prefetcher.next()

        # end of this epoch (training dataloader exhausted)

    # end of all epochs

    # ==========
    # final log & close logger
    # ==========

    if rank == 0:
        total_time = total_timer.get_interval() / 3600
        msg = "TOTAL TIME: [{:.1f}] h".format(total_time)
        print(msg)
        log_fp.write(msg + '\n')
        
        msg = (
            f"\n{'<' * 10} Goodbye {'>' * 10}\n"
            f"Timestamp: [{utils.get_timestr()}]"
            )
        print(msg)
        log_fp.write(msg + '\n')
        
        log_fp.close()


if __name__ == '__main__':
    main()
    
