import os
import argparse
import shutil


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--image_src_dir', type=str, default="original_mra",
                        help='Source folder containing subfolders for each image')
    parser.add_argument('--mask_src_dir', type=str, default="manual_segmentation",
                        help='Folder containing masks')
    parser.add_argument('--image_tgt_dir', type=str, default="images",
                        help='Target directory for consolidated images')
    parser.add_argument('--mask_tgt_dir', type=str, default="labels",
                        help='Target directory for consolidated masks')

    args = parser.parse_args()
    image_srcdir = args.image_src_dir
    mask_srcdir = args.mask_src_dir
    image_tgtdir = args.image_tgt_dir
    mask_tgtdir = args.mask_tgt_dir      

    try:
        for subdir in os.scandir(image_srcdir):
            image_list = sorted(os.listdir(subdir))
            mask_list = sorted(os.listdir(os.path.join(mask_srcdir, os.path.basename(subdir))))
            for i in range(len(image_list)):
                if image_list[i].endswith('.jpg'):
                    shutil.copy(os.path.join(subdir.path, image_list[i]), os.path.join(image_tgtdir, image_list[i])) 
                    #shutil.copy(, os.path.join(image_tgtdir, image_list[i]))
                    shutil.copy(os.path.join(os.path.join(mask_srcdir, os.path.basename(subdir)), mask_list[i]), os.path.join(mask_tgtdir, image_list[i]))

    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()                        