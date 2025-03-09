## Contents

This folder contains pretrained weights for some of the sample networks used here (to be expanded).

The following table provides an overview:
  
  

| File                                   | Description |
|:--------------------------------------|-------------|
| resnet18_caltech101_compressed.pt     | ResNet-18 trained on CalTech-101 dataset, ternary weights which are entropy-coded. |
| resnet18_caltech101_fp32_ternary.pt   | ResNet-18 trained on CalTech-101 dataset, weights are still 32-bit FP but are trained with enforcement of ternary values, the non-compressed version of the file above. |
| resnet18_caltech101_fp32_full.pt      | ResNet-18 trained on CalTech-101 dataset, weights from training without any quantization constraints |

