# Adaptive Calibration of Electrode Array Shifts Enables Robust Myoelectric Control (TBME 2019)
By [Xu Zhang](https://est.ustc.edu.cn/2015/0729/c4618a42767/page.psp), [Le Wu](https://github.com/wule1994), Bin Yu, [Xiang Chen](https://scholar.google.com/citations?hl=en&user=JURnq4QAAAAJ), [Xun Chen](http://staff.ustc.edu.cn/~xunchen/index.htm)

This document contains tutorial for running the codes associated with the method reported in the paper entitled "adaptive calibration of electrode array shifts enables robust myoelectric control", whcih has been accepted by IEEE Transactions on Biomedical Engineering. You can refer to "https://ieeexplore.ieee.org/document/8895996" for the published paper, and if you build upon our work, please cite our work with:  
X. Zhang, L. Wu, B. Yu, X. Chen, and X. Chen, “Adaptive Calibration of Electrode Array Shifts Enables Robust Myoelectric Control,” IEEE Transactions on Biomedical Engineering, pp. 1-1, 2019.


## Principle
![](./image/flowchart.png "flowchart of the proposed method")
Given the current electrode array as an example, there was an 8 × 8 subarray in the center of the entire 10 × 10 array. The muscle region, covered by this subarray at the baseline position, was termed core recording region (CRR). Only the data from the CRR were used to train the classification model and any data from other shift locations and the peripheral electrodes around the central subarray were not used. We assumed that any electrode shift was less than 7 mm in the proximal/distal and left/right directions, without loss of generality (any larger shift can be supported by relatively increasing the area of the entire array and decreasing the area of the CRR). During the testing phase, given the unknown but reasonably small shift of the electrode array, the CRR was still covered by the array (Fig. 2b). Therefore, the myoelectric pattern recognition can be achieved by searching and detecting the learned CRR object within the entire array image of a testing sample. Both its location (representing direction and distance of the shift) and pattern label were reported simultaneously. 

### License

The repository is released under the Apache-2.0 License (refer to the LICENSE file for details).

## Installing dependencies
The code is developed based on the Keras framwork.
* **Keras (version: 2.2.4)**: we use tensorflow backend. The Keras installation and introduction are available at `https://keras.io/`.
* **tensorflow (version: 1.13.1)**: tensorflow installation instructions are available at `https://www.tensorflow.org/`.

## Demonstration with an exemplary dataset
This is a demo when running our codes on an exempleary dataset, which can also be publicly downloaded via the same link as the source codes. Here are step-by-step instructions from downloading both the source codes and the data to result display:

### preparation
* download data folder (from [google driver](https://drive.google.com/file/d/1LsSEDZS2wbthcNZeqBXdfE-hNCIc6Cif/view?usp=sharing) or [baidu net disk](https://pan.baidu.com/s/1Xz9yrlO6h7HltbchAAJHSw)), this folder contains `training data` from baseline position and `testing data` from shift position.
* download model folder(from [google driver](https://drive.google.com/file/d/1aC1t7AHnsG10E6x76A6kFEUHyfSIpcrJ/view?usp=sharing) or [baidu net disk](https://pan.baidu.com/s/1tO8TskdZ-rZrdAsngANX8Q)), this folder includes `VGG16 weight` and a `pretrained model`. The pretrained model trained only with the data from the baseline position.

For more details, you can referring to the correspond code files or leave a message in the issue.

### Training
Then, you just input the following sentence to train it.
```bash
python train.py
```
***Or you can skip this step using our pretrained model.***

### Testing & Result display
To obtain the results by running the program for testing, you need to type and operate the following instruction:
```bash
python demo.py
```
***Note: if the classification model needs to be customized and re-trained, please uncomment line 181 and comment line 182 in the code file “demo.py.”***
The final classification accuracy is 0.9762. After the program is implemented, a confusion matrix and an alignment matrix are also displayed in the window. According to the alignment matrix, our method is able to predict that the array was shifted in the left-proximal direction. It is truly the case.

![](./image/result.png "results")
