# Conv1d_Modules_tf
This repository contains 1-dimmensional cnn modules for regression task. I used these modules for time-series prediction such as blood pressure(BP) prediction using electrocardiogram(ECG) and ballistocardiogram(BCG).
I tried to follow the original paper's implementation [1][2][3] as closely as possible, but I made some improvement in certain parts. [4] 

## Supported Modules
1. VGG_Net_1D [1]
2. ResNet_1D [2]
3. RepVGG_Net_1D [3]

## How To Use :
```bash
git clone https://github.com/NoahYn/Conv1d_Modules.git
cd Conv1d_Modules
python [file name to run]
```
you can choose sub_module in each file.

## Specification :
1. VGG_Net
![image](https://github.com/NoahYn/Conv1d_Modules/assets/101003842/4736ba46-fe58-45cc-ad51-be86cdd12136)

2. ResNet
   
![image](https://github.com/NoahYn/Conv1d_Modules/assets/101003842/c91bcdbd-637d-4705-b562-bfe9353d354a)

3. RepVGG
![image](https://github.com/NoahYn/Conv1d_Modules/assets/101003842/3c752142-1245-46fa-86d1-8df0d9502ad3)

## TODO :
---
  Implement DenseNet_1D, ResNeXt_1D, EfficientNet_1D ...

  Add explanation 
  
  Add example to test
  
  Add pytorch version

## References
---
**[1]** Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for Large-Scale Image Recognition. https://arxiv.org/abs/1409.1556.  

**[2]** He, K., & Zhang, X., & Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition.
https://arxiv.org/abs/1512.03385

**[3]** Ding, X., & Zhang, X., & Ma, N., & Han, J., & Ding, G., & Sun, J. (2021) RepVGG: Making VGG-style ConvNets Great Again https://arxiv.org/pdf/2101.03697

**[4]** Ioffe, S., & Szegedy, C. (2021). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. arXiv.org. Retrieved 30 August 2021, from https://arxiv.org/abs/1502.03167.

## Related Works
https://github.com/Sakib1263/VGG-1D-2D-Tensorflow-Keras
https://github.com/DingXiaoH/RepVGG/tree/main
https://github.com/hoangthang1607/RepVGG-Tensorflow-2
