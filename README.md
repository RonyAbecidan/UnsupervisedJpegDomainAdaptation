![](https://img.shields.io/badge/Official%20-Yes-1E8449.svg) ![](https://img.shields.io/badge/Library%20-Pytorch_Lightning-6C3483.svg) ![](https://img.shields.io/badge/Topic%20-Forensics_&_Domain_Adaptation-2E86C1.svg)



## Unsupervised JPEG Domain Adaptation for Practical Digital Image Forensics
### @[WIFS2021](https://wifs2021.lirmm.fr/) (Montpellier, France)
##### Rony Abecidan, Vincent Itier, Jeremie Boulanger, Patrick Bas
[![](https://img.shields.io/badge/Bibtex-0C0C0C?style=for-the-badge)](#CitingUJDA)  [![](https://img.shields.io/badge/Article-2E86C1?style=for-the-badge)](https://hal.archives-ouvertes.fr/hal-03374780/)  [![](https://img.shields.io/badge/Website-239B56?style=for-the-badge)](https://ronyabecidan.github.io/UJDA/) [![](https://img.shields.io/badge/Presentation-F7DC6F?style=for-the-badge)](https://u.pcloud.link/publink/show?code=kZAtfNXZaKn5FuwW7Dz80kPTFgcEJFWFDJT7)

![](https://svgshare.com/i/cX6.svg)


## Installation

To be able to reproduce our experiments and do your own ones, please follow our [Installation Instructions](INSTALL.md)


# Architecture used

![](https://svgshare.com/i/cWR.svg)


## Domain Adaptation in action

- **Source** : Half of images from the Splicing category of DEFACTO 
- **Target** : Other half of the images from the Splicing category of DEFACTO, compressed to JPEG with a quality factor of 5%

To have a quick idea of the adaptation impact on the training phase, we selected a batch of size 512 from the target and, we represented the evolution of the final embeddings distributions from this batch during the training according to the setups **SrcOnly** and **Update($`\sigma=8`$)**
described in the paper. The training relative to the SrcOnly setup is on the left meanwhile the one relative to **Update($`\sigma=8`$)** is on the right.

**Don't hesitate to click on the gif below to see it better !**


![](./adaptation.gif)

- As you can observe, in the **SrcOnly** setup, the forgery detector is more and more prone to false alarms, certainly because compressing images to QF5 results in creating artifacts in the high frequencies that can be misinterpreted by the model. However, it has no real difficulty to identify correctly the forged images.

- In parallel, in the **Update** setup, the forgery detector is more informed and make less false alarms during the training.

## Discrepancies with the first version of our article 

Several modifications have been carried out since the writing of this paper in order to :

- **Generate databases as most clean as possible**
- **Make our results as most reproducible as possible**
- **Reduce effectively computation time and memory space**

Considering that remark, you will not exactly retrieve the results we shared in the first version of the paper with the implementation proposed here. Nevertheless, the results we got from this new implementation are comparable with the previous ones and you should obtain similar results as the ones shared in this page.

For more information about the modifications we performed and the reasons behind, click [here](./UPDATES.md)

## Main references

```BibTeX
@inproceedings{mandelli2020training,
  title={Training {CNNs} in Presence of {JPEG} Compression: Multimedia Forensics vs Computer Vision},
  author={Mandelli, Sara and Bonettini, Nicol{\`o} and Bestagini, Paolo and Tubaro, Stefano},
  booktitle={2020 IEEE International Workshop on Information Forensics and Security (WIFS)},
  pages={1--6},
  year={2020},
  organization={IEEE}
}

@inproceedings{bayar2016,
  title={A deep learning approach to universal image manipulation detection using a new convolutional layer},
  author={Bayar, Belhassen and Stamm, Matthew C},
  booktitle={Proceedings of the 4th ACM workshop on information hiding and multimedia security (IH\&MMSec)},
  pages={5--10},
  year={2016}
}

@inproceedings{long2015learning,
  title={Learning transferable features with deep adaptation networks},
  author={Long, M. and Cao, Y. and Wang, J. and Jordan, M.},
  booktitle={International Conference on Machine Learning},
  pages={97--105},
  year={2015},
  organization={PMLR}
}


@inproceedings{DEFACTODataset, 
	author = {Ga{\"e}l Mahfoudi and Badr Tajini and Florent Retraint and Fr{\'e}d{\'e}ric Morain-Nicolier and Jean Luc Dugelay and Marc Pic},
	title={{DEFACTO:} Image and Face Manipulation Dataset},
	booktitle={27th European Signal Processing Conference (EUSIPCO 2019)},
	year={2019}
}
```

---
## <a name="CitingUJDA"></a>Citing our paper
### If you wish to refer to our paper,  please use the following BibTeX entry
```BibTeX
@inproceedings{abecidan:hal-03374780,
  TITLE = {{Unsupervised JPEG Domain Adaptation for Practical Digital Image Forensics}},
  AUTHOR = {Abecidan, Rony and Itier, Vincent and Boulanger, J{\'e}r{\'e}mie and Bas, Patrick},
  URL = {https://hal.archives-ouvertes.fr/hal-03374780},
  BOOKTITLE = {{WIFS 2021 : IEEE International Workshop on Information Forensics and Security}},
  ADDRESS = {Montpellier, France},
  PUBLISHER = {{IEEE}},
  YEAR = {2021},
  MONTH = Dec,
  PDF = {https://hal.archives-ouvertes.fr/hal-03374780/file/2021_wifs.pdf},
  HAL_ID = {hal-03374780}
}
```
