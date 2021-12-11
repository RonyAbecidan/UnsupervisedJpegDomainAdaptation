As said in the main README file, several modifications have been carried out since the writing of this paper in order to :

- **Generate databases as most clean as possible**
- **Make our results as reproducible as possible**
- **Reduce effectively computation time and memory space**

We details here exactly what has changed and why it has changed

- **Generate databases as most clean as possible** :

Originally when we generated the databases from the sources and the targets, we didn't pay attention to the content of the patches used for the train and test phases.

More precisely, between two variations from the source or the target (e.g. QF(5%) and QF(10%)), for each fold, there were not exactly the same contents between the patches at the same locations. For instance, if we considered the first fold of the database Source(None), the content of the patch located at position 42 was not the same content as the patch from the first fold of the database Source(QF(5%)) at the same position.

Ideally, to be the most fair as possible, it's important to train and test our detector on sets of patches with the same contents in each fold for each kind of source or target. If it's not the case, it's possible that one set of patches is more advantageous for the training compared to an other and the same remark applies for the testing phase. **This is now corrected** in our implementation and you can check it with our notebook.

- **Make our results as reproducible as possible**

There was a problem with the reproducibility of the experiments we made for the paper. Notably, even if the seed was correctly fixed for the initialization of the detector each time, the gradient descent was, however, not deterministic. **This is now corrected** in our implementation and you may retrieve the interesting results we are sharing in this repository (but it depends also of your GPU).

- **Reduce effectively computation time and memory space**

1 - Previously, we stored unintentionally the images with an int32 format. This is not a good practice since it takes way more memory space for absolutely no change compared to **int8**. **This is now corrected**

2 - Instead of doing a 5-fold validation, we noticed that a **3-fold validation** was already sufficient to show that our results are not resulting from randomness meanwhile ensuring a lighter storage of our databases.

3 - The Bayar architecture was a bit heavy for the task we wanted to solve. We finally decided to reduce the dimensions used for the hidden dense layers from 1024 to 256. This enabled to drastically reduce the training time and the memory taken for the storage of the weights meanwhile keeping an excellent performance on our datasets.

4 - The batch size was previously fixed at 64 because of memory space issues with the GPU. Finally, with a bit of memory management, it was possible to use a greater batch size of 128 which is a more reliable choice for estimating the MMD at each training step. Moreover, a larger batch size enables to speed the training time.
