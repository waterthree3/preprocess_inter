import os
import re
import sys

def yuv_to_png(yuv_path):
    match = re.search(r'_(\d+)x(\d+)_', yuv_path)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        print(f'Width: {width}, Height: {height}')
    else:
        print('Resolution not found in the filename')
        sys.exit()

    filename = os.path.basename(yuv_path).split('.')[0]
    parent_dir = os.path.dirname(yuv_path)
    grandparent_dir = os.path.dirname(parent_dir)
    # rec_path = os.path.join(grandparent_dir, "enhance", "frame_%d.png")
    rec_dir = os.path.join(grandparent_dir, "enhance", "img")
    if not os.path.exists(rec_dir):
        os.makedirs(rec_dir)
    filename = filename + "_%d.png"
    rec_path = os.path.join(rec_dir, filename)

    os.system(
        r'ffmpeg -loglevel error -f rawvideo -s ' + str(width) + 'x' + str(
            height) + ' -pix_fmt yuv420p -i ' + yuv_path + ' -vf "extractplanes=y" ' + rec_path
    )
    return rec_dir


def png_to_yuv(rec_dir):
    png_dir = rec_dir
    png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]

    png_filename = png_files[0].rsplit('_', 1)[0] + '_%d.png'
    yuv_filename = png_files[0].rsplit('_', 1)[0] + '_enhance.yuv'
    match = re.search(r'_(\d+)x(\d+)_', png_filename)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
        print(f'Width: {width}, Height: {height}')
    else:
        print('Resolution not found in the filename')
        sys.exit()
    png_path = os.path.join(png_dir, png_filename)

    parent_dir = os.path.dirname(png_dir)
    yuv_path = os.path.join(parent_dir, yuv_filename)
    os.system(
        'ffmpeg -loglevel error -f image2 -framerate 30 -i ' + png_path + ' -pix_fmt yuv444p -s ' + str(width) + 'x' + str(height)
    + ' ' + yuv_path)


def main():
    yuv_path = rf"D:\MachineLearning_Project\stdf_dataset\MFQEv2_dataset\test_01\raw\BasketballDrill_832x480_500.yuv"
    rec_dir = yuv_to_png(yuv_path)
    png_to_yuv(rec_dir)


if __name__ == '__main__':
    main()