# Adaptive Calibration of Electrode Array Shifts toward Robust Myoelectric Control
By [Xu Zhang](https://est.ustc.edu.cn/2015/0729/c4618a42767/page.psp), [Le Wu](https://github.com/wule1994), Bin Yu, [Xiang Chen](https://scholar.google.com/citations?hl=en&user=JURnq4QAAAAJ), [Xun Chen](http://staff.ustc.edu.cn/~xunchen/index.htm)

This code is to to automatic and adaptive calibration of electrode array and assist to enhance robustness of myoelectric control systems. The code is developed based on the Keras framwork.

## Introduction
![](./image/flowchart.png "flowchart of the proposed method")
Overview of the proposed method. Given the current electrode array as an example, we assumed that any electrode shift was no greater than 7mm along both the proximal/distal and left/right directions (any larger deviation can be easily sensed). Therefore, a central region, covered by the central portion of electrode array in a form of 8 × 8, is supposed to be always covered by various shift conditions. This region is termed as core recording region (CRR). The classification model only trained with the data from CRR at the baseline position. During the testing phase, given the unknown but reasonably small shift of the electrode array, it is assumed that the muscle area corresponding to the CRR is still covered by the array. Therefore, the myoelectric pattern recognition can be achieved by considering the learnt HD-sEMG image corresponding to the CRR as an object to be detected within the entire array image.

### License

The reposity is released under the Apache-2.0 License (refer to the LICENSE file for details).