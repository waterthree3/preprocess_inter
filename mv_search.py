import imageio
from scipy.signal import convolve2d
from scipy.ndimage import zoom
from torch.cuda.amp import autocast

from io_yuv import *
import numpy as np
from scipy.ndimage import convolve
import cv2
import torch
import torch.nn.functional as F
from scipy.signal import correlate

def mv_search(y, block_size=8):
    device = torch.device('cuda')
    # block_size = 16
    search_range = 5
    motion_vector_factor = 16   # mv 16精度
    batch, num_frames, h, w = y.shape

    cur_y = y[:, 1]     # 当前图像 第二张
    # 合并 batch 和 i, j 坐标
    y_unf = cur_y.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
    blocks = (h // block_size) * (w // block_size)
    y_unf = y_unf.contiguous().view(batch, blocks, block_size, block_size)
    current_block_flattened = y_unf.view(batch, blocks, -1)
    # current_block_flattened = current_block_flattened.unsqueeze(1)

    ref_frame = y[:, :1]    # 当前图像 第一张
    # mv_vector = torch.zeros((batch, blocks), dtype=torch.int32, device=device)
    min_error = torch.full((batch, blocks), float('inf'), dtype=torch.float32, device=device)
    ref_block = F.unfold(ref_frame, kernel_size=(block_size, block_size), stride=1, padding= block_size // 2)
    ref_block = ref_block.permute(0, 2, 1)  # [N, H*W, C*k*k]
    N = current_block_flattened.shape[1]

    # ref_block_expanded = ref_block.unsqueeze(1) # [batch_size, 1, H*W, block_flattened_size]
    # 计算当前块与整个 ref_block 的绝对差值，并按元素相加
    # current_block_flattened = current_block_flattened.unsqueeze(2)
    # abs_diff_sum = torch.sum(torch.abs(current_block_flattened - ref_block_expanded), dim=-1)
    # min_abs_diff, min_index = torch.min(abs_diff_sum, dim=-1)
    # mv_vector[:, :] = min_index

    # stride = 64
    # for i in range(0, N, stride):
    #     current_batch = current_block_flattened[:, i:i+stride].unsqueeze(2)   # 取出当前的 (1, 256)
    #     ref_block_expanded = ref_block.unsqueeze(1) # [batch_size, 1, H*W, block_flattened_size]
    #     # 计算当前块与整个 ref_block 的绝对差值，并按元素相加
    #     abs_diff_sum = torch.sum(torch.abs(current_batch - ref_block_expanded), dim=-1)
    #     min_abs_diff, min_index = torch.min(abs_diff_sum, dim=-1)
    #     mv_vector[:, i:i+stride] = min_index
    #     del abs_diff_sum, min_abs_diff, min_index, current_batch, ref_block_expanded
    #     torch.cuda.empty_cache()

    # 假设你的数据和变量已初始化
    quarter = N // 4
    ref_block_expanded = ref_block.unsqueeze(1)

    current_block_flattened = current_block_flattened.unsqueeze(2)

    # 使用混合精度（如果需要减少显存）
    with autocast():
        # 处理第一部分数据
        abs_diff_sum1 = torch.sum(torch.abs(current_block_flattened[:, :quarter] - ref_block_expanded), dim=-1)
        min_error[:, :quarter] = torch.min(abs_diff_sum1, dim=-1)[0]

        # 释放第一部分计算中的临时变量
        # del current_batch1, abs_diff_sum1, min_abs_diff1, min_index1
        # torch.cuda.empty_cache()  # 手动释放显存

        # 处理第二部分数据
        abs_diff_sum2 = torch.sum(torch.abs(current_block_flattened[:, quarter:2 * quarter] - ref_block_expanded), dim=-1)
        min_error[:, quarter:2 * quarter] = torch.min(abs_diff_sum2, dim=-1)[0]

        # 释放第二部分计算中的临时变量
        # del current_batch2, abs_diff_sum2, min_abs_diff2, min_index2
        # torch.cuda.empty_cache()  # 手动释放显存

        # 处理第三部分数据
        abs_diff_sum3 = torch.sum(torch.abs(current_block_flattened[:, 2 * quarter:3 * quarter] - ref_block_expanded), dim=-1)
        min_error[:, 2 * quarter:3 * quarter] = torch.min(abs_diff_sum3, dim=-1)[0]

        # 释放第三部分计算中的临时变量
        # del current_batch3, abs_diff_sum3, min_abs_diff3, min_index3
        # torch.cuda.empty_cache()  # 手动释放显存

        # 处理第四部分数据
        abs_diff_sum4 = torch.sum(torch.abs(current_block_flattened[:, 3 * quarter:] - ref_block_expanded), dim=-1)
        min_error[:, 3 * quarter:] = torch.min(abs_diff_sum4, dim=-1)[0]

        # 释放第四部分计算中的临时变量
        # del current_batch4, abs_diff_sum4, min_abs_diff4, min_index4
        # torch.cuda.empty_cache()  # 手动释放显存

    return min_error


def apply_motion_vectors(y, mv, block_size):
    batch, num_frames, height, width = y.shape
    predicted_frame = torch.zeros_like(y)

    ref_frame = y[:, :1]
    ref_block = F.unfold(ref_frame, kernel_size=(block_size, block_size), stride=1, padding=block_size // 2)
    ref_block = ref_block.permute(0, 2, 1)  # [N, H*W, C*k*k]
    mv = mv.long()
    width_block = width //block_size
    height_block = height // block_size

    for ba in range(batch):
        for j in range(height_block):
            for i in range(width_block):
                # 当前块的索引
                block_idx = j * width_block + i
                # 从参考块中提取相应的块
                idx = mv[ba, block_idx]
                block = ref_block[ba, idx].view(block_size, block_size)
                # 放置块到目标图像中
                predicted_frame[ba, :, j * block_size: (j + 1) * block_size, i * block_size: (i + 1) * block_size] = block

    return predicted_frame


def print_grad(grad):
    print("Gradient:", grad)

def mctf_mv_search(tensor_one, tensor_two):
    y = torch.cat((tensor_one, tensor_two), dim=1)

    min_error = mv_search(y, 8)

    # predicted_frame = apply_motion_vectors(y, mv_final, 8)

    # return predicted_frame[:, :1]
    return min_error


if __name__ == '__main__':
    # sys.exit(main())
    mctf_mv_search()