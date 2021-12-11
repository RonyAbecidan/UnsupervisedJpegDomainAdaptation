import numpy as np
import os
import h5py
import cv2
from tqdm import tqdm
from sklearn.model_selection import KFold

#Path to the folder containing the database
imgs_path = r'./All_splicing'

if not os.path.isdir(f"Domains"):
    os.mkdir(f"Domains")
    os.mkdir(f"Domains/Sources")
    os.mkdir(f"Domains/Targets")

# https://www.kite.com/python/answers/how-to-write-a-list-to-a-file-in-python
def list_to_txt(my_list,path):

    textfile = open(path, "w")

    for element in my_list:

        textfile.write(element + "\n")

    textfile.close()

#https://www.kite.com/python/answers/how-to-convert-each-line-in-a-text-file-into-a-list-in-python
def txt_to_list(txt_path):
    file=open(txt_path, "r")
    return [(line.strip()).split() for line in file]


def return_filenames(folder_name):

    if not os.path.isfile(f'filenames_{folder_name}.txt'):
        print('Warning : the order of the paths is not available, you may create different databases from the ones',
              f'used for the experiments of the article. If it matters to you, add the file filenames_{folder_name}.txt)',
              f'from the repo in the working space. Otherwise a new file will be created')
        filenames=[]
        for filename in tqdm((os.listdir(f'./All_splicing/{folder_name}'))):

            filenames.append(filename)

        list_to_txt(filenames,f'filenames_{folder_name}.txt')
    else:
        filenames=txt_to_list(f'filenames_{folder_name}.txt')

    filenames=np.array(filenames).reshape(-1)

    return filenames

img_folder='img'
addrs=return_filenames(img_folder)

def is_genuine(mask):
    m = np.sum(mask)
    return (m==0)

def is_forged_and_not_too_extreme(mask):   
    mask=(mask>0).astype(int)
    m = np.mean(mask)
    return (m>=0.2 and m<=0.8)

def sharpen_img(img, alpha=1):
    kernel = -np.ones((3, 3), dtype=np.float32) * (1 / alpha)
    kernel[0, 0] = 0
    kernel[-1, 0] = 0
    kernel[0, -1] = 0
    kernel[-1, -1] = 0
    kernel[1, 1] = 1 - np.sum(kernel) + kernel[1, 1]

    return cv2.filter2D(img, -1, kernel)

def sharpen(alpha):

    def sub_function(im):
        return  sharpen_img(im,alpha)

    return sub_function

def blurry(img):
    return cv2.blur(img,(10,10))

def compress_img(img,fq=100):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), fq]
    result, encimg = cv2.imencode('.jpg',img,encode_param)
    img = cv2.imdecode(encimg, 1)
    return img

def compress(fq):
    def sub_function(im):
        return compress_img(im,fq)

    return sub_function


def extract_patches_for_classification(fake_path,mask_path,real,fake,position,windows=(128,128,3),stride=None,transformation=None,seed=2021,nb_patch_max=2):

    img=cv2.imread(fake_path)
    mask=cv2.imread(mask_path)

    if not(img is None):
        w, h, c = img.shape
        
        if stride is None:
            stride=windows[0]
        
        if not(transformation is None):
            img=transformation(img)

        W=w//stride
        H=h//stride

        cpt_real=0
        cpt_fake=0
        outer_break=False
        np.random.seed(seed)

        for i in range(0, w-windows[0]+1, stride):
            for j in range(0, h-windows[1]+1, stride):
                
                patch_img = img[i:i+windows[0],j:j+windows[1]]
                patch_mask = mask[i:i+windows[0],j:j+windows[1]]


                if is_genuine(patch_mask) and (np.random.random()<(nb_patch_max/(H*W))):
                        real[nb_patch_max*position+cpt_real]=patch_img
                        cpt_real+=1
                            
                if is_forged_and_not_too_extreme(patch_mask):
                        fake[nb_patch_max*position+cpt_fake]=patch_img
                        cpt_fake+=1

                if (cpt_fake>=nb_patch_max) or (cpt_real>=nb_patch_max):
                    outer_break=True
                    break

            if outer_break:
                break


                    
def save_base(domain_type,name,transformation=None,patch_size=128,nb_split=3,seed=2021,nb_patch_max=2,size=len(addrs)):

    #shuffle
    np.random.seed(seed)
    final_addrs=np.random.choice(a=addrs, size=size, replace=False)

    assert (domain_type=='Source' or domain_type=='Target'), "you should name the domain type as 'Source' or 'Target' "
    
    if domain_type=='Source':
        dataset=final_addrs[0:size//2]
    else:
        dataset=final_addrs[size//2:]

    hdf5_path = f"./Domains/{domain_type}s/{name}.hdf5"
    
    kf = KFold(n_splits=nb_split,shuffle=False)

    with h5py.File(hdf5_path, mode='w') as f:
        for i,(train_index, test_index) in enumerate(kf.split(dataset)):

            real=np.zeros(shape=(len(train_index)*nb_patch_max,patch_size,patch_size,3),dtype=np.uint8)
            fake=np.zeros(shape=(len(train_index)*nb_patch_max,patch_size,patch_size,3),dtype=np.uint8)

            print('len real', len(real))


            train_addrs=dataset[train_index]
            test_addrs=dataset[test_index]


            for k,addr in tqdm(enumerate(train_addrs)):

                        addr=f'{imgs_path}/{img_folder}/{addr}'
                        mask_addr=addr[:-4].replace(img_folder,mask_folder)+'.jpg'
                        extract_patches_for_classification(real=real,fake=fake,position=k,fake_path=addr, mask_path=mask_addr,
                                                           windows=(patch_size,patch_size,3),transformation=transformation,seed=seed,nb_patch_max=nb_patch_max)


            real_sum=real.sum(axis=(1,2,3))
            real=real[real_sum>0]

            fake_sum=fake.sum(axis=(1,2,3))
            fake=fake[fake_sum>0]

            I=np.random.choice(a=np.arange(len(real)),size=len(fake),replace=False)
            real=real[I]
            
            train=np.concatenate([real,fake],axis=0)
            l_train=np.concatenate([np.zeros(len(real)),np.ones(len(fake))],axis=0)
            
            f[f'train_{i}']=train
            f[f'l_train_{i}']=l_train

    
            real=np.zeros(shape=(len(test_index)*nb_patch_max,patch_size,patch_size,3),dtype=np.uint8)
            fake=np.zeros(shape=(len(test_index)*nb_patch_max,patch_size,patch_size,3),dtype=np.uint8)

            # loop over test paths
            for k,addr in tqdm(enumerate(test_addrs)):

                        addr=f'{imgs_path}/{img_folder}/{addr}'
                        mask_addr=addr[:-4].replace(img_folder,mask_folder)+'.jpg'
                        extract_patches_for_classification(real=real,fake=fake,position=k,fake_path=addr, mask_path=mask_addr,
                                                           windows=(patch_size,patch_size,3),transformation=transformation,seed=seed,nb_patch_max=nb_patch_max)


            real_sum=real.sum(axis=(1,2,3))
            real=real[real_sum>0]

            fake_sum=fake.sum(axis=(1,2,3))
            fake=fake[fake_sum>0]

            I=np.random.choice(a=np.arange(len(real)),size=len(fake),replace=False)
            real=real[I]
            
            test=np.concatenate([real,fake],axis=0)
            l_test=np.concatenate([np.zeros(len(real)),np.ones(len(fake))],axis=0)

            f[f'test_{i}']=test
            f[f'l_test_{i}']=l_test



if __name__ == "__main__":

    save_base('Source',transformation=None,patch_size=128,nb_split=3,name='source-none')
    save_base('Source',transformation=compress(100),patch_size=128,nb_split=3,name='source-qf(100)')
    save_base('Source',transformation=compress(5),patch_size=128,nb_split=3,name='source-qf(5)')

    save_base('Target',transformation=None,patch_size=128,nb_split=3,name='target-none')
    save_base('Target',transformation=compress(5),patch_size=128,nb_split=3,name='target-qf(5)')
    save_base('Target',transformation=compress(10),patch_size=128,nb_split=3,name='target-qf(10)')
    save_base('Target',transformation=compress(20),patch_size=128,nb_split=3,name='target-qf(20)')
    save_base('Target',transformation=compress(50),patch_size=128,nb_split=3,name='target-qf(50)')
    save_base('Target',transformation=compress(100),patch_size=128,nb_split=3,name='target-qf(100)')
