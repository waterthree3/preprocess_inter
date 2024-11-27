import os
import time

import torchvision
import yaml
import argparse
import torch
import os.path as op
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import utils  # my tool box
import dataset
from net_stdf import MFVQE


def receive_arg():
    """Process all hyper-parameters and experiment settings.
    
    Record in opts_dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--opt_path', type=str, default='option.yml', 
        help='Path to option YAML file.'
        )
    args = parser.parse_args()
    
    with open(args.opt_path, 'r') as fp:
        opts_dict = yaml.load(fp, Loader=yaml.FullLoader)

    opts_dict['opt_path'] = args.opt_path

    if opts_dict['train']['exp_name'] == None:
        opts_dict['train']['exp_name'] = utils.get_timestr()

    opts_dict['train']['log_path'] = op.join(
        "exp", opts_dict['train']['exp_name'], "log_test.log"
        )
    opts_dict['train']['checkpoint_save_path_pre'] = op.join(
        "exp", opts_dict['train']['exp_name'], "ckp_"
        )
    opts_dict['test']['restore_iter'] = int(
        opts_dict['test']['restore_iter']
        )
    opts_dict['test']['checkpoint_save_path'] = (
        f"{opts_dict['train']['checkpoint_save_path_pre']}"
        f"{opts_dict['test']['restore_iter']}"
        '.pt'
        )

    return opts_dict


def save_to_yuv(enhanced_data, yuv_path):
    # 将enhanced_data转换为numpy数组
    enhanced_data_np = enhanced_data.cpu().numpy()
    y_data = enhanced_data_np[0, 0, :, :]  # 只提取亮度分量Y

    y_data = (y_data * 255).astype(np.uint8)
    # 创建U和V分量，所有值设为128
    u_data = np.full_like(y_data, 128, dtype=np.uint8)
    v_data = np.full_like(y_data, 128, dtype=np.uint8)
    yuv_data = np.concatenate([y_data, u_data, v_data], axis=0)
    # 将Y分量写入YUV文件
    with open(yuv_path, 'ab') as yuv_file:
        # 将Y分量写入YUV文件
        yuv_file.write(yuv_data.tobytes())


def main():
    # ==========
    # parameters
    # ==========

    opts_dict = receive_arg()
    unit = opts_dict['test']['criterion']['unit']

    # ==========
    # open logger
    # ==========

    log_fp = open(opts_dict['train']['log_path'], 'w')
    msg = (
        f"{'<' * 10} Test {'>' * 10}\n"
        f"Timestamp: [{utils.get_timestr()}]\n"
        f"\n{'<' * 10} Options {'>' * 10}\n"
        f"{utils.dict2str(opts_dict['test'])}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ========== 
    # Ensure reproducibility or Speed up
    # ==========

    #torch.backends.cudnn.benchmark = False  # if reproduce
    #torch.backends.cudnn.deterministic = True  # if reproduce
    torch.backends.cudnn.benchmark = True  # speed up

    # ==========
    # create test data prefetchers
    # ==========
    
    # create datasets
    test_ds_type = opts_dict['dataset']['test']['type']
    radius = opts_dict['network']['radius']
    assert test_ds_type in dataset.__all__, \
        "Not implemented!"
    test_ds_cls = getattr(dataset, test_ds_type)
    test_ds = test_ds_cls(
        opts_dict=opts_dict['dataset']['test'], 
        radius=radius
        )

    test_num = len(test_ds)
    test_vid_num = test_ds.get_vid_num()

    # create datasamplers
    test_sampler = None  # no need to sample test data

    # create dataloaders
    test_loader = utils.create_dataloader(
        dataset=test_ds, 
        opts_dict=opts_dict, 
        sampler=test_sampler, 
        phase='val'
        )
    assert test_loader is not None

    # create dataloader prefetchers
    test_prefetcher = utils.CPUPrefetcher(test_loader)

    # ==========
    # create & load model
    # ==========

    model = MFVQE(opts_dict=opts_dict['network'])

    checkpoint_save_path = opts_dict['test']['checkpoint_save_path']
    msg = f'loading model {checkpoint_save_path}...'
    print(msg)
    log_fp.write(msg + '\n')

    checkpoint = torch.load(checkpoint_save_path)
    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])
    
    msg = f'> model {checkpoint_save_path} loaded.'
    print(msg)
    log_fp.write(msg + '\n')

    model = model.cuda()
    model.eval()

    # ==========
    # define criterion
    # ==========

    # define criterion
    assert opts_dict['test']['criterion'].pop('type') == \
        'PSNR', "Not implemented."
    criterion = utils.PSNR()

    # ==========
    # validation
    # ==========
                
    # create timer
    total_timer = utils.Timer()

    # create counters
    per_aver_dict = dict()
    ori_aver_dict = dict()
    name_vid_dict = dict()
    for index_vid in range(test_vid_num):
        per_aver_dict[index_vid] = utils.Counter()
        ori_aver_dict[index_vid] = utils.Counter()
        name_vid_dict[index_vid] = ""

    pbar = tqdm(
        total=test_num, 
        ncols=opts_dict['test']['pbar_len']
        )

    # ======save to yuv file========
    yuv_dir = opts_dict['test']['yuv_pro_store']['save_path']
    if yuv_dir and os.path.exists(yuv_dir):
        yuv_name = opts_dict['test']['yuv_pro_store']['file_name']
        yuv_path = os.path.join(yuv_dir, yuv_name)
        if os.path.exists(yuv_path):
            os.remove(yuv_path)
    # =====period enhance==========
    enhance_period = opts_dict['test']['yuv_pro_store']['enhance_order']


    # fetch the first batch
    test_prefetcher.reset()
    val_data = test_prefetcher.next()

    index = 0
    epsilon = 1e-8  # 设置一个很小的偏移量
    with torch.no_grad():
        while val_data is not None:
            # get data
            gt_data = val_data['gt'].cuda()  # (B [RGB] H W)
            lq_data = val_data['lq'].cuda()  # (B T [RGB] H W)
            index_vid = val_data['index_vid'].item()
            name_vid = val_data['name_vid'][0]  # bs must be 1!

            b, _, c, _, _  = lq_data.shape
            assert b == 1, "Not supported!"

            input_data = torch.cat(
                [lq_data[:,:,i,...] for i in range(c)],
                dim=1
                )  # B [R1 ... R7 G1 ... G7 B1 ... B7] H W

            if index % enhance_period == 0:
                enhanced_data = model(input_data)  # (B [RGB] H W)
            else:
                enhanced_data = gt_data
            # # #=========验证测试=============
            img_file_path = rf"D:\MachineLearning_Project\stdf-pytorch\exp\MFQEv2_pwc_mctf_opt2\pic"
            lq_data_fir = input_data[:, :1, :, :]
            lq_data_second = input_data[:, 1:2, :, :]
            lq_data_third = input_data[:, 2:3, :, :]
            torchvision.utils.save_image(enhanced_data, img_file_path + "/enhanced_data.png", nrow=1)
            torchvision.utils.save_image(gt_data, img_file_path + "/gt_data.png", nrow=1)
            torchvision.utils.save_image(lq_data_fir, img_file_path + "/lq_data_fir.png", nrow=1)
            torchvision.utils.save_image(lq_data_second, img_file_path + "/lq_data_second.png", nrow=1)
            torchvision.utils.save_image(lq_data_third, img_file_path + "/lq_data_third.png", nrow=1)
            # # #=============================

            # 计数更新
            index += 1

            # eval
            # batch_ori = criterion(lq_data[0, radius, ...], gt_data[0])
            batch_ori = criterion(enhanced_data[0]+epsilon, gt_data[0])
            batch_perf = criterion(enhanced_data[0]+epsilon, gt_data[0])

            # display
            pbar.set_description(
                "{:s}: [{:.3f}] {:s} -> [{:.3f}] {:s}"
                .format(name_vid, batch_ori, unit, batch_perf, unit)
                )
            pbar.update()

            # log
            per_aver_dict[index_vid].accum(volume=batch_perf)
            ori_aver_dict[index_vid].accum(volume=batch_ori)
            if name_vid_dict[index_vid] == "":
                name_vid_dict[index_vid] = name_vid
            else:
                assert name_vid_dict[index_vid] == name_vid, "Something wrong."

            # ======save to yuv file========
            if yuv_dir and os.path.exists(yuv_dir):
                save_to_yuv(enhanced_data, yuv_path)
            # fetch next batch
            val_data = test_prefetcher.next()
        
    # end of val
    pbar.close()

    # log
    msg = '\n' + '<' * 10 + ' Results ' + '>' * 10
    print(msg)
    log_fp.write(msg + '\n')
    for index_vid in range(test_vid_num):
        per = per_aver_dict[index_vid].get_ave()
        ori = ori_aver_dict[index_vid].get_ave()
        name_vid = name_vid_dict[index_vid]
        msg = "{:s}: [{:.3f}] {:s} -> [{:.3f}] {:s}".format(
            name_vid, ori, unit, per, unit
            )
        print(msg)
        log_fp.write(msg + '\n')
    ave_per = np.mean([
        per_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    ave_ori = np.mean([
        ori_aver_dict[index_vid].get_ave() for index_vid in range(test_vid_num)
        ])
    msg = (
        f"{'> ori: [{:.3f}] {:s}'.format(ave_ori, unit)}\n"
        f"{'> ave: [{:.3f}] {:s}'.format(ave_per, unit)}\n"
        f"{'> delta: [{:.3f}] {:s}'.format(ave_per - ave_ori, unit)}"
        )
    print(msg)
    log_fp.write(msg + '\n')
    log_fp.flush()

    # ==========
    # final log & close logger
    # ==========

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
    