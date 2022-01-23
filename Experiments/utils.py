import os
import numpy as np
import pytorch_lightning as pl
# ##Pytorch
import torch
from torch import nn
import  torchvision.transforms as transforms
import yaml
from pathlib import Path

import h5py as h5
from torch.utils import data


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformation=transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class MyDataset(data.Dataset):
    
    def __init__(self,archive,key1='img',key2='labels',transform=transformation):
        self.archive = h5.File(archive, 'r')
        self.data = self.archive[key1]
        self.labels = self.archive[key2]
        self.transform=transform
     
    def __getitem__(self, index):
        datum = self.data[index]
        datum = torch.tensor(datum).transpose(1,2).transpose(0,1)
        datum=datum/255
        datum=self.transform(datum)
        


        return datum, torch.tensor(self.labels[index]).type_as(torch.FloatTensor(0))

    def __len__(self):
        return len(self.labels)

    def close(self):
        self.archive.close()

default_config_path='default_config.yaml'

def initialize_hyperparameters(my_config_path=default_config_path):

    hyperparameters=yaml.safe_load(Path(my_config_path).read_text())
    hyperparameters['source']['name']=hyperparameters['source']['filename'][:-5].replace('source-','')
    hyperparameters['target']['name']=hyperparameters['target']['filename'][:-5].replace('target-','')
    hyperparameters['eval']['domain_names']=[domain[:-5].replace('target-','') for domain in hyperparameters['eval']['domain_filenames']]
    
    if hyperparameters['precisions'] is None:
        hyperparameters['precisions']=f"s={hyperparameters['source']['name']}_t={hyperparameters['target']['name']}"
    else:
        if not ('s=' in hyperparameters['precisions']):
            hyperparameters['precisions']=f"s={hyperparameters['source']['name']}_t={hyperparameters['target']['name']}_{hyperparameters['precisions']}"

    return hyperparameters
       

def MMD(x, y, sigma=0.5):
    # Gaussian MMD
    # https: // discuss.pytorch.org / t / maximum - mean - discrepancy - mmd - and -radial - basis - function - rbf / 1875
    #  x and y have size [B, 1, W, H]

    B = max(2, x.size(0))

    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)

    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    K = torch.exp(- (rx.t() + rx - 2 * xx) / (2 * sigma ** 2))
    L = torch.exp(- (ry.t() + ry - 2 * yy) / (2 * sigma ** 2))
    P = torch.exp(- (rx.t() + ry - 2 * zz) / (2 * sigma ** 2))

    beta = (1. / (B * (B - 1)))
    gamma = (2. / (B * B))

    return beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)

class BayarClassifier(pl.LightningModule):
    def __init__(self, im_size=128, in_channel=3):
        super(BayarClassifier, self).__init__()
        
        self.im_size=im_size
        
        self.in_channel=in_channel
        
        self.BayarConv2D=nn.Conv2d(in_channel,12, 5, 1, padding=0)  # (im_size-4)x(im_size-4)
        
        f_size=np.ceil((self.im_size-4)/8).astype(int)
        
        self.conv = nn.Sequential(
                                nn.Conv2d(12,64,7,2,padding=3), #(im_size-4)//2 x (im_size-4)//2
                                nn.MaxPool2d(kernel_size=3,stride=2,padding=1),  # (im_size-4)//4 x (im_size-4)//4,
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64,48,5, 1, padding=2),
                                nn.MaxPool2d(kernel_size=3,stride=2,padding=1),  # (im_size-4)//8 x (im_size-4)//8,
                                nn.BatchNorm2d(48),
                                nn.ReLU(),
                                nn.Flatten()
                                )

        self.linear1=nn.Sequential(nn.Linear(48*f_size*f_size,256),
                                   nn.ReLU(),
                                   nn.Dropout(0.5))

        self.linear2=nn.Sequential(nn.Linear(256,256),
                                   nn.ReLU(),
                                   nn.Dropout(0.5))

        self.linear3 = nn.Linear(256,1)
        
        self.bayar_mask = torch.ones((5, 5),device=device)
        self.bayar_final = torch.zeros((5, 5),device=device)
        
        self.bayar_mask[2, 2] = 0
        self.bayar_final[2, 2] = -1

        
    def forward(self, x):

        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(12,self.in_channel,1, 1), -1)
        self.BayarConv2D.weight.data += self.bayar_final
        
        output = self.BayarConv2D(x)
        output = self.conv(output)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        
        return output


def SrcOnly(forgery_detector,data):

    images_source,labels_source,images_target,labels_target=data

    out_source_final = forgery_detector.forward(images_source)
    out_target_final = forgery_detector.forward(images_target)

    total_loss = forgery_detector.loss(out_source_final.view(-1), labels_source)

    forgery_detector.compute_accuracy_during_training(out_source_final.view(-1),labels_source,source=True)
    forgery_detector.compute_accuracy_during_training(out_target_final.view(-1),labels_target,source=False)

    # forgery_detector.log('BCE_loss',loss_classification,on_step=True,on_epoch=True,prog_bar=True)
    forgery_detector.log('s_train', forgery_detector.source_train_accuracy / forgery_detector.already_passed_source, on_step=True, on_epoch=True, prog_bar=True)
    forgery_detector.log('t_train', forgery_detector.target_train_accuracy / forgery_detector.already_passed_target, on_step=True, on_epoch=True, prog_bar=True)

    forgery_detector.cpt += 1

    return total_loss


def Adaptation_with_MMD(forgery_detector,data):
    images_source,labels_source,images_target,labels_target=data

    size_1 = images_source.size(0)
    size_2 = images_target.size(0)
    min_size = min(size_1, size_2)

    images_source = images_source[0:min_size, :]
    images_target = images_target[0:min_size, :].to(device)
    labels_target=labels_target[0:min_size].to(device)
    labels_source=labels_source[0:min_size]

    out_source1 = forgery_detector.detector.BayarConv2D(images_source)
    out_source1 = forgery_detector.detector.conv(out_source1)
    out_source2 = forgery_detector.detector.linear1(out_source1)
    out_source3 = forgery_detector.detector.linear2(out_source2)
    out_source4 = forgery_detector.detector.linear3(out_source3)

    out_target1 = forgery_detector.detector.BayarConv2D(images_target)
    out_target1 = forgery_detector.detector.conv(out_target1)
    out_target2 = forgery_detector.detector.linear1(out_target1)
    out_target3 = forgery_detector.detector.linear2(out_target2)
    out_target4 = forgery_detector.detector.linear3(out_target3)

    adaptation_loss = MMD(out_source2.to(device), out_target2.to(device), sigma=forgery_detector.sigmas[0]) + \
                      MMD(out_source3.to(device), out_target3.to(device), sigma=forgery_detector.sigmas[1]) + \
                      MMD(out_source4.to(device), out_target4.to(device), sigma=forgery_detector.sigmas[2])

    if forgery_detector.cpt == 0:
        forgery_detector.adaptation_ref = (adaptation_loss.item())  # computation of the normalization term

    # Normalization
    adaptation_loss = (adaptation_loss / forgery_detector.adaptation_ref)
    forgery_detector.log('MMD_loss', adaptation_loss, on_step=True, on_epoch=True, prog_bar=True)

    classification_loss = forgery_detector.loss(out_source4.view(-1), labels_source)

    total_loss = (1 - forgery_detector.alpha) * classification_loss + forgery_detector.alpha * adaptation_loss

    forgery_detector.compute_accuracy_during_training(out_source4.view(-1),labels_source,source=True)
    forgery_detector.compute_accuracy_during_training(out_target4.view(-1),labels_target,source=False)

    forgery_detector.log('s_train', forgery_detector.source_train_accuracy / forgery_detector.already_passed_source, on_step=True, on_epoch=True, prog_bar=True)
    forgery_detector.log('t_train', forgery_detector.target_train_accuracy / forgery_detector.already_passed_target, on_step=True, on_epoch=True, prog_bar=True)

    forgery_detector.cpt += 1

    return total_loss
