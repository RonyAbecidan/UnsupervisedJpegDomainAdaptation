## Installation

### First steps

- Of course, start by cloning this repo
- Make sure you have at least **Python 3.8**
- Then we strongly advise to the readers to create a new environment for this project for instance with the command `conda create -n the_name_you_want`
- You will need after to install pip with `conda install pip` 
- If you have a gpu do `conda install pytorch torchvision cudatoolkit=11.3 -c pytorch` otherwise if you prefer to use your CPU do `conda install pytorch torchvision cpuonly -c pytorch`
- At last launch `pip install -r requirements.txt` 

### Construction of the domains and running of our experiments
After preparing the required environment, you need to construct the bases which will be used for the training and evaluation phases.
- The first step for that is to ask access to the DEFACTO database [here](https://defactodataset.github.io/) (Download it at the level of the repository root)
  Once this is done rename the folder containing all the images DEFACTO. 

- Then, you can launch the script `python data_pipeline.py`, it will take a while. 
  This script will first gather all the images and associated masks of forged regions from the Splicing category of [DEFACTO](https://defactodataset.github.io/) into two folders. 
  Then, it builds the source and target domains, a domain being a set of 128x128 forged or not forged patches obtained from spliced images of DEFACTO (eventually compressed before the cutting into patches).
  Please note that by default, each domain is made of 3 fold in order to realize a 3 fold cross validation. If you don't want to do a cross validation or if you want to test more fold, you can change it straightforwardly in the file `construct_domains.py`.
  
  *To be sure that you construct the same domains as ours, we saved a .txt file containing the order of the image paths we extracted from the Splicing category. By default, the code available here takes into account that list and not the one you could obtain from your side that can lead to a different order.*


- When this is finished, you can reproduce our experiments lauching the script `simulations.py` moving to the folder `Experiments`. By default it will reproduce all the experiments of the paper given that you created previously all the necessary domains.
  Each experiment of our paper is associated to a code. Giving the code to the function `reproduce` from `simulations.py` enables to reproduce our experiments. However, this does not ensure that you will obtain the same results since it can change according to your GPU/CPU. **All our experiments have been launched with the GPU NVIDIA GeForce GTX 1060 6GB** .
You can be at least sure to start the trainings with the same weights.

Please find below a table linking a code to an experiment presented in the paper : 

| Name of the experiment                                                                                   | Code |
|----------------------------------------------------------------------------------------------------------|------|
| SrcOnly with Source=None and Target=QF(5)                                                                | 0    |
| TgtOnly with Source=QF(5) and Target=QF(5)                                                               | 1    |
| TgtOnly with Source=QF(5) and Target=QF(5) using only 1000 patches for the source during training        | 2    |
| SrcOnly with Source=QF(100) and Target=QF(5)                                                             | 3    |
| Update($`\sigma=8`$) with Source=None and Target=QF(5)                                                     | 4    |
| Update($`\sigma=8`$) with Source=None and Target=QF(5) using only 10 patches for the target training set   | 5    |
| Update($`\sigma=8`$) with Source=None and Target=QF(5) using only 100 patches for the target training set  | 6    |
| Update($`\sigma=8`$) with Source=None and Target=QF(5) using only 1000 patches for the target training set | 7    |
| Update($`\sigma=0.01`$) with Source=None and Target=QF(5)  											   | 8    |
| Update($`\sigma=1000`$) with Source=None and Target=QF(5)                                                 | 9    |
| Update($`\sigma=8`$) with Source=QF(100) and Target=QF(5)                                                  | 10   |
| Update($`\sigma=8`$) with Source=QF(5) and Target=QF(100)                                                  | 11   |
| Mix with Source=None and Target=QF(5)                                                                    | 12   |
| Mix with Source=QF(100) and Target=QF(5)                                                                 | 13   |

- If you want to make other experiments feel free to precise your hyperparameters using a yaml config file like the one in `.\Experiments\example.yaml` before calling the function `simulate(hyperparameters_config_path)`.

- Each experiment leads to the creation of a folder in `Experiments\Results` where the hyperparameters you gave, the weights of the best models found during the trainings and the final results are stored progressively. This folder has a name deduced 
from the filenames of the source and the target. For instance SrcOnly(None --> QF(5)) leads to the construction of the folder `SrcOnly_s=none_t=qf(5)`. You can add to this name some details about your experiment using the key `'precisions'` in the config file.

- The name *hyperparameters* is in a broad sense since this dictionary should contain information such as learning rate, batch sizes, etc... but also the filenames (with extensions) of the bases you want to use as your source and your target. 

- On top of that, you have the possibility to do the evaluation phase on several domains but you need to precise their paths in the config file. 
  If you want to test the Update setup, please note that you need to precise the bandwiths used for each kernel in a list with the key `sigmas` in the config file.
  
 In order to better realize how things work here you can have a look on the notebook [Demo.ipynb](./Experiments/Demo.ipynb)
 
 At the end, the files should be organized like that : 
 
 
 ```
 unsupervisedjpegdomainadaptation/
└── DEFACTO/ 
	├──	Splicing/ 
		├── 1 
			├── img
			├── probe_mask
		├── .
		├── .
		├── .
		├── 7
			├── img
			├── probe_mask
	├──	.
	├──	.
	├──	.
	├──	Swapping
	
└── All_splicing/
	├── img
	├── probe_mask
	
└── Domains/

	├── Sources/
		├── source-none.hdf5
		├── source-qf(5).hdf5
		├── source-qf(100).hdf5
		
	├── Targets/
		├── target-none.hdf5
		├── target-qf(5).hdf5
		├── target-qf(10).hdf5
		├── target-qf(20).hdf5
		├── target-qf(50).hdf5
		├── target-qf(100).hdf5
		
└── Experiments/

	├── Results/
		├── SrcOnly-s=none_t=qf(5)
			├── Bayar-SrcOnly-s=none_t=qf(5).txt
			├── hyperparameters-SrcOnly-s=none_t=qf(5).yaml
		├── .
		├── .
		├── .
	├── Demo.ipynb 
	├── utils.py
	├── simulations.py
	├── code_to_experiment.yaml
	├── example.yaml
	
└── Presentation/
	├── Wifs2021_Presentation.pdf
	
└── create_database.py
└── create_domains.py 
└── data_pipeline.py
└── filenames_img.txt 
└── requirements.txt 	
```

		

	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
 
