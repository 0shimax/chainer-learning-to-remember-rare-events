import sys
import os
import random
import PIL.Image
import PIL.ImageStat
import matplotlib.pyplot as plt


def show_categ(c, imgs_for_categ, max_num, image_dir_path, n_column=5, figsize=(18,5)):
    num_row = max_num//(n_column+1)+1

    n = 0
    for f in random.sample(imgs_for_categ[c], min(max_num, len(imgs_for_categ[c]))):
        if n==0:
            plt.figure(figsize=figsize)
        img = PIL.Image.open(image_dir_path+'/'+f)
        plt.subplot( 1, n_column, n+1)
        plt.axis('off')
        plt.imshow(img)
        plt.title(os.path.basename(f))
        if n==n_column-1:
            plt.show(f)
        n = (n+1)%n_column
    if n!=0:
        plt.show()

def show_all_categ(imgs_for_categ, max_num, image_dir_path):
    for c in sorted(imgs_for_categ):
        print(c, ':', len(imgs_for_categ[c]))
        show_categ(c, imgs_for_categ, max_num, image_dir_path)


if __name__=='__main__':
    import subprocess

    p1_label2clsval ={
        'SNE':0,
        'LY':1,
        'MO':2,
        'BNE':3,
        'EO':4,
        'BA':5,
        'MY':6,
        'MMY':7,
        'ERB':8,
        'BL':9,
        'PMY':10,
        'GT':11,
        'ART':12,
        'SMU':13,
        # 'TAG':14,
        'VLY':15,
        'PC':16,
        'NC':17,
        'OTH':17,
        # 'UI':17,
        'MEK':18,
        'ERC':14,  # TAGがERCと間違えてラベル付けされていたため、ERCも14としている
    }

    DATA_POINTER_PATH = './data/phase1_image_pointer'
    DATA_DIR_PATH = './data/phase1_stage1'

    imgs_for_categ = {}
    for label in p1_label2clsval.keys():
        proc = subprocess.Popen( \
                "cat {}|grep ''^{}_'".format(DATA_POINTER_PATH, label), \
                stdout=subprocess.PIPE,
                shell=True)
        out, err = proc.communicate()
        out = out.decode('utf-8')
        fname_list = out.rstrip().split('\n')

        imgs_for_categ[label] = fname_list

    show_all_categ(imgs_for_categ, 10, DATA_DIR_PATH)
