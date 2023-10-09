# See_SIFT_in_a_Rain
W. Wu, H. Chang and Z. Li, "See SIFT in a Rain," in IEEE Transactions on Circuits and Systems for Video Technology, doi: 10.1109/TCSVT.2023.3318304.

## Abstract 
Rain streaks bring complicated pixel intensity changes and additional gradients, greatly obstructing the extraction of image features from background. This causes serious performance degradation in feature-based applications. Thus, it is critical to remove rain streaks from a single rainy image to recover image features. Recently, many excellent image deraining methods have made remarkable progress. However, these human visual system-driven approaches mainly focus on improving image quality with pixel recovery as loss function, and neglect how to enhance image feature recovery ability. To address this issue, we propose a task-driven image deraining algorithm to strengthen image feature supply for subsequent feature-based applications. Due to the extensive use and strong practicability of Scale-Invariant Feature Transform (SIFT), we first propose two separate networks using distinct losses and modules to achieve two goals, respectively. One is difference of Gaussian (DoG) pyramid recovery network (DPRNet) for SIFT detection, and the other gradients of Gaussian images recovery network (GGIRNet) for SIFT description. Second, in the DPRNet we propose an alternative interest point loss that directly penalizes scale response extrema to recover the DoG pyramid. Third, we advance a gradient attention module in the GGIRNet to recover those gradients of Gaussian images. Finally, with the recovered DoG pyramid and gradients, we can regain SIFT key points. This divide-and-conquer scheme to set different objectives for SIFT detection and description leads to good robustness. Compared with state-of-the-art methods, experimental results demonstrate that our proposed algorithm achieves better performance in both the number of recovered SIFT key points and their accuracy.

## Link
IEEE Xplore:[paper](https://ieeexplore.ieee.org/document/10261252)

百度网盘：[pretrained models and derain results](https://pan.baidu.com/s/1jw9PtvwaBSyPsvoLexpwdA) 
提取码：1252
