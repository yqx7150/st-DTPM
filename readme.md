# st-DTPM

**paper**:  [st-DTPM: Spatial-Temporal Guided Diffusion Transformer Probabilistic Model for Delayed Scan PET Image Prediction](https://arxiv.org/abs/2410.22732)

**Authors**: Ran Hong, Yuxia Huang, Lei Liu, Mengxiao Geng, Zhonghui Wu, Bingxuan Li, Xuemei Wang, Qiegen Liu*

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

### Related work :four_leaf_clover:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672)

- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/abs/2104.07636)



