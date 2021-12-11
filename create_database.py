import os
import shutil
from tqdm import tqdm

defacto_path='./DEFACTO/Splicing'

if not os.path.isdir(f"All_splicing"):
    os.mkdir('All_splicing')
    os.mkdir('All_splicing/img')
    os.mkdir('All_splicing/probe_mask')

for folder in tqdm(os.listdir(defacto_path)):
    for im in tqdm(os.listdir(f'{defacto_path}/{folder}/img')):
        shutil.copyfile(f'{defacto_path}/{folder}/img/{im}', f'All_splicing/img/{im}')

    for im in tqdm(os.listdir(f'{defacto_path}/{folder}/probe_mask')):
        shutil.copyfile(f'{defacto_path}/{folder}/probe_mask/{im}', f'All_splicing/probe_mask/{im}')


