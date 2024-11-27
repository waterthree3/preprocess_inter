import numpy as np


# ==========
# YUV420P
# ==========


def import_yuv(seq_path, h, w, tot_frm, yuv_type='420p', start_frm=0, only_y=True):
    """Load Y, U, and V channels separately from a 8bit yuv420p video.

    Args:
        seq_path (str): .yuv (imgs) path.
        h (int): Height.
        w (int): Width.
        tot_frm (int): Total frames to be imported.
        yuv_type: 420p or 444p
        start_frm (int): The first frame to be imported. Default 0.
        only_y (bool): Only import Y channels.

    Return:
        y_seq, u_seq, v_seq (3 channels in 3 ndarrays): Y channels, U channels,
        V channels.

    Note:
        YUV传统上是模拟信号格式, 而YCbCr才是数字信号格式.YUV格式通常实指YCbCr文件.
        参见: https://en.wikipedia.org/wiki/YUV
    """
    # setup params
    if yuv_type == '420p':
        hh, ww = h // 2, w // 2
    elif yuv_type == '444p':
        hh, ww = h, w
    else:
        raise Exception('yuv_type not supported.')

    y_size, u_size, v_size = h * w, hh * ww, hh * ww
    blk_size = y_size + u_size + v_size

    # init
    y_seq = np.zeros((tot_frm, h, w), dtype=np.uint8)
    if not only_y:
        u_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)
        v_seq = np.zeros((tot_frm, hh, ww), dtype=np.uint8)

    # read data
    with open(seq_path, 'rb') as fp:
        for i in range(tot_frm):
            fp.seek(int(blk_size * (start_frm + i)), 0)  # skip frames
            y_frm = np.fromfile(fp, dtype=np.uint8, count=y_size).reshape(h, w)
            if only_y:
                y_seq[i, ...] = y_frm
            else:
                u_frm = np.fromfile(fp, dtype=np.uint8, \
                                    count=u_size).reshape(hh, ww)
                v_frm = np.fromfile(fp, dtype=np.uint8, \
                                    count=v_size).reshape(hh, ww)
                y_seq[i, ...], u_seq[i, ...], v_seq[i, ...] = y_frm, u_frm, v_frm

    if only_y:
        return y_seq
    else:
        return y_seq, u_seq, v_seq


def write_ycbcr(y, cb, cr, vid_path):
    with open(vid_path, 'wb') as fp:
        for ite_frm in range(len(y)):
            fp.write(y[ite_frm].reshape(((y[0].shape[0]) * (y[0].shape[1]),)))
            fp.write(cb[ite_frm].reshape(((cb[0].shape[0]) * (cb[0].shape[1]),)))
            fp.write(cr[ite_frm].reshape(((cr[0].shape[0]) * (cr[0].shape[1]),)))


def write_y(y, vid_path):
    with open(vid_path, 'wb') as fp:
        for ite_frm in range(len(y)):
            frame = y[ite_frm].reshape(((y[0].shape[0]) * (y[0].shape[1]),))
            fp.write(frame)


def write_y_tenosr(y, vid_path):
    y = (y * 255).clamp(0, 255).byte()
    y = y.cpu().numpy()
    with open(vid_path, 'wb') as fp:
        for ite_frm in range(len(y)):
            frame = y[ite_frm].reshape(((y[0].shape[0]) * (y[0].shape[1]),))
            fp.write(frame)