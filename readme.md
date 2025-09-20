# st-DTPM

**paper**:  [st-DTPM: Spatial-Temporal Guided Diffusion Transformer Probabilistic Model for Delayed Scan PET Image Prediction](https://arxiv.org/abs/2410.22732)

**Authors**: Ran Hong, Yuxia Huang, Lei Liu, Mengxiao Geng, Zhonghui Wu, Bingxuan Li, Xuemei Wang, Qiegen Liu*

https://ieeexplore.ieee.org/abstract/document/10980366      

**Date**: Apr. 28, 2025

The code and the algorithm are for non-commercial use only.

Copyright 2025, School of Information Engineering, Nanchang University.

----

### Intro :cherry_blossom:

The target of delayed scan PET image prediction is to predict delayed scan PET image from first scan PET image.

![target](./assert/intro.bmp)

----

### Motivation :tulip:

The time interval between first and delayed PET image is a crucial factor affecting delayed imaging. And in clinical practice, the time interval for each patient to perform delayed imaging is uncertain.

![](./assert/NoP.jpg)

----

### Proposed :sunflower:

A **Diffusion** model with **Transformer** under **Spatial-Temporal** guidance is proposed. Spatial condition is first scan PET image; Temporal condition is delay time interval.

![model](./assert/st-DTPM.png)

----

### Results :maple_leaf:

![result](./assert/results.png)

----

### Training & Testing :evergreen_tree:

**Training for first and delayed PET images. **

--embDTMode and --transEmbDTMode can choose the method of embedding temporal condition into ConvBlock and TransformerBlock, respectively.

| Option value | Method               |
| ------------ | -------------------- |
| 1            | each block embedding |
| 2            | linear cat embedding |
| 3            | add embedding        |
| 4            | linear add embedding |

--condition can choose if use spatial guidance.

--embDT can choose if use temporal guidance.

```python
python runner/train.py --embDTMode=1 --transEmbDTMode=1 --condition=True --embDT=True --runType="train"
```

**Testing for specific delay time interval.**

--delayed_time is the delay time interval you given.

```python
python runner/train.py --embDTMode=1 --transEmbDTMode=1 --condition=True --embDT=True --runType="train" --delayed_time=120
```

----

### Other Related Projects
<div align="center"><img src="https://github.com/yqx7150/PET_AC_sCT/blob/main/samples/algorithm-overview.png" width = "800" height = "500"> </div>
 Some examples of invertible and variable augmented network: IVNAC, VAN-ICC, iVAN and DTS-INN.   

  * Variable Augmented Network for Invertible Modality Synthesis and Fusion  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10070774)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/iVAN)    
  
 * Variable augmentation network for invertible MR coil compression  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X24000225)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VAN-ICC)         

 * Virtual coil augmentation for MR coil extrapoltion via deep learning  [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S0730725X22001722)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/VCA)    

  * Synthetic CT Generation via Invertible Network for All-digital Brain PET Attenuation Correction  [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2310.01885)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PET_AC_sCT)        

  * Temporal Image Sequence Separation in Dual-tracer Dynamic PET with an Invertible Network  [<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/10542421)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DTS-INN)        

  * Invertible and Variable Augmented Network for Pretreatment Patient-Specific Quality Assurance Dose Prediction  [<font size=5>**[Paper]**</font>](https://link.springer.com/article/10.1007/s10278-023-00930-w)                 
  * A Prior-Guided Joint Diffusion Model in Projection Domain for PET Tracer Conversion [<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2506.16733) [<font size=5>**[Code]**</font>](https://github.com/yqx7150/PJDM)    
        
  * Variable augmented neural network for decolorization and multi-exposure fusion [<font size=5>**[Paper]**</font>](https://www.sciencedirect.com/science/article/abs/pii/S1566253517305298)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/DecolorNet_FusionNet_code)   [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/EDAEPRec/tree/master/Slide)   
   

### Related work :four_leaf_clover:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)



