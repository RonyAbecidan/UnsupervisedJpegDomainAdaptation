import os
import torch
import numpy as np
import yaml
from pathlib import Path
import gc

from torch import nn
from torch.utils.data import Subset
from torch.utils.data import DataLoader, ConcatDataset
#Tools for creating a Dataloader
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import *

if not os.path.isdir(f"Results"):
        os.mkdir(f"Results")

sources=[f'source-{domain}.hdf5' for domain in ['qf(5)','qf(100)','none']]
targets=[f'target-{domain}.hdf5' for domain in ['qf(5)','qf(10)','qf(20)','qf(50)','qf(100)','none']]

print(device)

class MyEarlyStopping(EarlyStopping):

    def on_train_epoch_end(self, trainer, pl_module):
         trainer.model.validation()
         self._run_early_stopping_check(trainer)

class ForgeryDetector(pl.LightningModule):

    # Model Initialization/Creation
    def __init__(self,data,hyperparameters):
        super(ForgeryDetector, self).__init__()
        self.hyperparameters=hyperparameters

        seed_everything(self.hyperparameters['seed'])

        self.source_train, self.source_test, self.target_train, self.target_test = data

        self.detector=BayarClassifier(in_channel=3,im_size=hyperparameters['im_size'])

        self.epoch_count = 0
        self.cpt = 0
        self.train_count = 0
        self.lamb = self.hyperparameters['training'].setdefault('lamb', 0)
        self.lr = self.hyperparameters['training'].setdefault('lr', 10 ** (-3))

        self.folder_path=f"{hyperparameters['training']['setup']}-{hyperparameters['precisions']}"


        if self.hyperparameters['training']['setup']=='Update':
            assert ('sigmas' in self.hyperparameters['training']), "You should precise the bandwiths parameters in the yaml file hyperparameters.yaml " \
                                                         " if you want to test the adaptation using the MMD with gaussian kernels (Liu, 2015) " \
                                                         " \n hyperparameters->training->sigmas"

            self.sigmas = self.hyperparameters['training']['sigmas']

            self.alpha=self.hyperparameters['training'].setdefault('alpha',0.5)


    # Forward Pass of Model
    def forward(self, x):
        return self.detector(x)

    def compute_accuracy_during_training(self, preds, target, source=True):
        with torch.no_grad():
            to_prob = nn.Sigmoid()
            pred_to_prob = to_prob(preds)
            final = (pred_to_prob >= 0.5).type_as(torch.IntTensor(0)).to(device)
            s = (final == target).type_as(torch.IntTensor(0)).sum()

            if source:
                self.source_train_accuracy += s.item()
                self.already_passed_source += len(target)

            else:
                self.target_train_accuracy += s.item()
                self.already_passed_target += len(target)

    def compute_test_accuracy(self, test_loader):
        '''
        Computation of the accuracy of our detector on a test set
        '''
        self.eval()
        with torch.no_grad():
            s = 0
            tot = 0
            to_prob = nn.Sigmoid()
            for images, target in test_loader:
                images_device, target_device = images.to(device), target.to(device)
                preds = self.forward(images_device).view(-1)
                pred_to_prob = to_prob(preds)
                final = (pred_to_prob > 0.5).type_as(torch.IntTensor(0)).to(device)
                s += (final == target_device).type_as(torch.IntTensor(0)).sum().item()
                tot += len(target)

            return s / tot

    # Loss Function
    def loss(self, y_hat, y):
        return nn.BCEWithLogitsLoss()(y_hat,y)

    # Optimizers
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.detector.parameters(),lr=self.lr,weight_decay=self.lamb)
#
        return [optimizer],[]

    # Calls after prepare_data for DataLoader
    def train_dataloader(self):
        return self.source_train

    def on_train_epoch_start(self):
        self.train()
        self.epoch_count += 1

        self.source_train_accuracy=0
        self.target_train_accuracy=0
        self.already_passed_source = 0
        self.already_passed_target = 0

        self.it = iter(self.target_train)


    # Training Loop
    def training_step(self, batch, batch_idx):
        images_source, labels_source = batch

        try:
                images_target, labels_target = next(self.it)
                images_target,labels_target=images_target.to(device),labels_target.to(device)
        except:
                self.it = iter(self.target_train)
                images_target, labels_target = next(self.it)
                images_target,labels_target=images_target.to(device),labels_target.to(device)

        if (self.hyperparameters['training']['setup']=='SrcOnly') or (self.hyperparameters['training']['setup']=='Mix'):
            total_loss=SrcOnly(self, data=[images_source,labels_source,images_target,labels_target])

        if self.hyperparameters['training']['setup']=='Update':
            total_loss=Adaptation_with_MMD(self, data=[images_source,labels_source,images_target,labels_target])

        output ={'loss': total_loss}

        return output


    def validation(self):
            if not(hasattr(self,'s_test')):
                self.s_test=0

            self.source_test_accuracy = self.compute_test_accuracy(self.source_test)
            self.target_test_accuracy = self.compute_test_accuracy(self.target_test)
            self.log('s_test',self.source_test_accuracy,on_epoch=True,prog_bar=True)
            self.log('t_test',self.target_test_accuracy,on_epoch=True,prog_bar=True)

            if  self.source_test_accuracy>self.s_test:
                self.s_test=self.source_test_accuracy
                # self.best_val_loss=self.val_loss
                print(f'Current best : {self.target_test_accuracy}')
                torch.save(self.state_dict(), f"./Results/{self.folder_path}/{self.hyperparameters['training']['setup']}-best_model.pt")

            if self.hyperparameters['training']['save_at_each_epoch']:
                torch.save(self.state_dict(), f"./Results/{self.folder_path}/{self.hyperparameters['training']['setup']}-{self.epoch_count}.pt")

def save_results(acc_test,hyperparameters):

    filename=f"{hyperparameters['detector_name']}-{hyperparameters['training']['setup']}-{hyperparameters['precisions']}"
    folder_path = f"{hyperparameters['training']['setup']}-{hyperparameters['precisions']}"

    description= f"""
=== FORGERY DETECTION TASK ===
source : {hyperparameters['source']['name']}
target : {hyperparameters['target']['name']}
    
--- DATA ---
source : random half images sampled from the "Splicing" category of the database DEFACTO
repartition : 1-1/{hyperparameters['N_fold']} 1/{hyperparameters['N_fold']}

A {hyperparameters['N_fold']}-fold cutting is applied to the source to split it into {hyperparameters['N_fold']} train and test sets
Then, the images in each cuts are transformed into batches of {hyperparameters['im_size']}x{hyperparameters['im_size']} patches.

**In each set, there is a perfect balance between forged and non-forged patches : **
    
-A patch associated to a forged region is kept in the sets only if the forged region occupy a space between 20% and 80% 
of the total space (128x128). 
-The real patches are chosen randomly so that there is an equal amount of forged and non-forged
patches.
-We kept only 2 patches for each class at maximum by image

target : 30000 images different of the source and potentially presenting a different preprocessing compared to the source 
(for instance a change in the quality factor for the compression).
repartition : 1-1/{hyperparameters['N_fold']} 1/{hyperparameters['N_fold']}

The preprocessing of the target images is the same as the one presented above

--- TRAINING ---
trainings_epochs on each fold : {hyperparameters['training']['max_epochs']}
hyperparameters_file : hyperparameters-{hyperparameters['training']['setup']}-{hyperparameters['precisions']}.yaml

--- RESULTS --- 

"""

    results="\n"
    for j in range(acc_test.shape[1]):
        results+=f"{hyperparameters['eval']['domain_names'][j]} : {np.round(np.mean(acc_test[:,j]),3)*100}% +/- {np.round(np.std(acc_test[:,j]),2)*100}%"
        results+="\n"

    with open(f"Results/{folder_path}/{filename}.txt", "w") as file:
        file.write(description+results)


def save_hyperparameters(hyperparameters):

    folder_path = f"./Results/{hyperparameters['training']['setup']}-{hyperparameters['precisions']}"

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)

    file_path=f"{folder_path}/hyperparameters-{hyperparameters['training']['setup']}-{hyperparameters['precisions']}.yaml"

 
    with open(file_path, 'w') as file:
       yaml.dump(hyperparameters, file,default_flow_style=False,sort_keys=False)


def train_detector(hyperparameters,acc_targets_global,i,mix=None):

    batch_size=hyperparameters['training']['batch_size']
    source_path=hyperparameters['source']['filename']
    target_path=hyperparameters['target']['filename']

    early_stop_callback = MyEarlyStopping(
        monitor='s_test',
        min_delta=0.00,
        patience=hyperparameters['training']['earlystop_patience'],
        verbose=True,
        mode='max'
    )


    train_set_source = MyDataset(f'../Domains/Sources/{source_path}', key1=f'train_{i}', key2=f'l_train_{i}')
    test_set_source = MyDataset(f'../Domains/Sources/{source_path}', key1=f'test_{i}', key2=f'l_test_{i}')

    train_set_target = MyDataset(f'../Domains/Targets/{target_path}', key1=f'train_{i}', key2=f'l_train_{i}')
    test_set_target = MyDataset(f'../Domains/Targets/{target_path}', key1=f'test_{i}', key2=f'l_test_{i}')

    seed_everything(hyperparameters['seed'])
  
    if hyperparameters['training']['setup']=='Mix':
        mix_set=ConcatDataset([train_set_source,train_set_target])
        N_mix=len(mix_set)
        indices_mix = torch.randperm(len(mix_set))[0:N_mix//2]

        source_train, source_test = DataLoader(Subset(mix_set, indices_mix), batch_size=batch_size,shuffle=True), \
                                    DataLoader(test_set_source, batch_size=batch_size, shuffle=True)

        target_train, target_test = DataLoader(train_set_target, batch_size=batch_size,shuffle=True),  \
                                    DataLoader(test_set_target, batch_size=batch_size, shuffle=True)
    else:

        N_target_train = len(train_set_target)
        to_keep = min(N_target_train, hyperparameters['target']['max_size'])
        indices_target = torch.randperm(len(train_set_target))[0:to_keep]

        N_source_train = len(train_set_source)
        to_keep = min(N_source_train, hyperparameters['source']['max_size'])
        indices_source = torch.randperm(len(train_set_source))[0:to_keep]

        source_train=  DataLoader(Subset(train_set_source, indices_source), batch_size=batch_size,shuffle=True)
        source_test = DataLoader(test_set_source, batch_size=batch_size, shuffle=True)

        target_train = DataLoader(Subset(train_set_target, indices_target), batch_size=batch_size,shuffle=True)
        target_test = DataLoader(test_set_target, batch_size=batch_size, shuffle=True)

    FD = ForgeryDetector(data=[source_train, source_test, target_train, target_test], hyperparameters=hyperparameters)
    if not(i):
        save_hyperparameters(FD.hyperparameters)

    if device=='cpu':
        trainer = Trainer(max_epochs=hyperparameters['training']['max_epochs'], progress_bar_refresh_rate=1,
                          callbacks=[early_stop_callback],
                          enable_checkpointing=False, logger=False, deterministic=True)
    else:
        trainer = Trainer(gpus=torch.cuda.device_count(),max_epochs=hyperparameters['training']['max_epochs'], progress_bar_refresh_rate=1,
                          callbacks=[early_stop_callback],
                          enable_checkpointing=False, logger=False, deterministic=True)

    trainer.fit(FD)
    FD.load_state_dict(torch.load(f"./Results/{FD.folder_path}/{hyperparameters['training']['setup']}-best_model.pt"))
    torch.save(FD.state_dict(), f"./Results/{FD.folder_path}/{hyperparameters['training']['setup']}-best_model_{i}.pt")

    acc_targets_local = []

    with torch.no_grad():
        FD.to(device)

        for path in hyperparameters['eval']['domain_filenames']:
            eval_set = MyDataset(f'../Domains/Targets/{path}', key1=f'test_{i}', key2=f'l_test_{i}')

            eval_loader = DataLoader(eval_set, batch_size=hyperparameters['eval']['batch_size'], shuffle=True)

            acc_targets_local.append(FD.compute_test_accuracy(eval_loader))

        acc_targets_global.append(acc_targets_local)

    del trainer
    del FD
    del eval_loader
    del source_train
    del source_test
    del target_train
    del target_test
    del early_stop_callback
    gc.collect()
    torch.cuda.empty_cache()

def simulate(hyperparameters_config_path):

    hyperparameters=initialize_hyperparameters(hyperparameters_config_path)

    acc_targets_global = []

    for i in range(hyperparameters['N_fold']):
        print('********************')
        print(f"Fold {i + 1}/{hyperparameters['N_fold']}")
        print('********************  \n')

        hyperparameters['training']['save_at_each_epoch']=False if i>0 else hyperparameters['training']['save_at_each_epoch']
        train_detector(hyperparameters,acc_targets_global,i)

    acc_targets_global=np.array(acc_targets_global)
    save_results(acc_targets_global,hyperparameters)


def retrieve_results(hyperparameters):

    FD = ForgeryDetector(data=[0,0,0,0], hyperparameters=hyperparameters)
    FD.to(device)

    acc_targets_global=[]
    folder_path=FD.folder_path

    for i in range(hyperparameters['N_fold']):
        acc_targets_local=[]
        print('********************')
        print(f"Fold {i + 1}/{hyperparameters['N_fold']}")
        print('********************  \n')
        FD.load_state_dict(torch.load(f"./Results/{folder_path}/{hyperparameters['training']['setup']}-best_model_{i}.pt"))

        with torch.no_grad():

            for path in hyperparameters['eval']['domain_filenames']:
                eval_set = MyDataset(f'../Domains/Targets/{path}', key1=f'test_{i}', key2=f'l_test_{i}')

                eval_loader = DataLoader(eval_set, batch_size=hyperparameters['eval']['batch_size'], shuffle=True)

                acc_targets_local.append(FD.compute_test_accuracy(eval_loader))

        acc_targets_global.append(acc_targets_local)


    acc_targets_global=np.array(acc_targets_global)


    save_results(acc_targets_global, hyperparameters)

code_to_experiment=yaml.safe_load(Path('code_to_experiment.yaml').read_text())

def reproduce(code):
    hyperparameters_config_path=f'./Results/{code_to_experiment[code]}/hyperparameters-{code_to_experiment[code]}.yaml'
    simulate(hyperparameters_config_path)

if __name__ == "__main__":
    reproduce(4)
    #for i in range(0,14):
    #     reproduce(i)

#hyperparameters=initialize_hyperparameters(hyperparameters_config_path)
