import glob
import os


def delete_superpixels_from_data():
    dir_path = os.path.join(os.curdir, 'data', 'ISIC_*_superpixels.png')
    file_list = glob.glob(dir_path)
    for path in file_list:
        os.remove(path)

if __name__ == '__main__':
    delete_superpixels_from_data()
