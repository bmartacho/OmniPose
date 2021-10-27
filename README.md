# OmniPose

  <a href="https://arxiv.org/abs/2103.10180">**OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation**</a>.
</p><br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/omnipose.png" title="OmniPose Architecture">
  Figure 1: OmniPose framework for multi-person pose estimation. The input color image of dimensions (HxW) is fed through the improvedHRNet backbone and WASPv2 module to generate one heatmap per joint, or class.
</p><br />

<p align="justify">
We propose OmniPose, a multi-scale framework for multi-person pose estimation. The OmniPose architecture leverages multi-scale feature representations to increase the effectiveness of  backbone feature extractors, with no significant increase in network size and no postprocessing. 
The OmniPose framework incorporates contextual information across scales and joint localization with Gaussian heatmap modulation at the multi-scale feature extractor to estimate human pose with state-of-the-art accuracy.
The multi-scale representations allowed by the improved waterfall module in the OmniPose framework leverage the efficiency of progressive filtering in the cascade architecture, while maintaining multi-scale fields-of-view comparable to spatial pyramid configurations.
Our results on multiple datasets demonstrate that OmniPose, with an improved HRNet backbone and waterfall module, is a robust and efficient architecture for multi-person pose estimation with state-of-the-art results. 

We propose the upgraded “Waterfall Atrous Spatial Pyramid” module, shown in Figure 2. WASPv2 is a novel architecture with Atrous Convolutions that is able to leverage both the larger Field-of-View of the Atrous Spatial Pyramid Pooling configuration and the reduced size of the cascade approach.<br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/WASPv2.png" width=1000 title="WASPv2 module"><br />
  Figure 2: WASPv2 Module.
</p><br />

Examples of the OmniPose architecture for Multi-Person Pose Estimation are shown in Figures 3.<br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/samples.png" width=1000 title="WASP module"><br />
  Figure 3: Pose estimation samples for OmniPose.
  <br /><br />
  
Link to the published article at <a href="http://www.brunoartacho.com">ArXiv</a>.
</p><br />

**Datasets:**
<p align="justify">
Datasets used in this paper and required for training, validation, and testing can be downloaded directly from the dataset websites below:<br />
  COCO Dataset: https://cocodataset.org/<br />
  MPII Dataset: http://human-pose.mpi-inf.mpg.de/<br />
</p><br />

**Pre-trained Models:**
<p align="justify">
The pre-trained weights for OmniPose can be downloaded at
  <a href="https://drive.google.com/drive/folders/1NoDE3plZoqF_O00xei0woH5cPy67arXq?usp=sharing">here</a>.

The pre-trained weights for HRNet can be downloaded at
  <a href="https://mailustceducn-my.sharepoint.com/personal/aa397601_mail_ustc_edu_cn/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Faa397601%5Fmail%5Fustc%5Fedu%5Fcn%2FDocuments%2FHRNet%2DBottom%2Dup%2DPose%2DEstimation%2Fmodel&originalPath=aHR0cHM6Ly9tYWlsdXN0Y2VkdWNuLW15LnNoYXJlcG9pbnQuY29tLzpmOi9nL3BlcnNvbmFsL2FhMzk3NjAxX21haWxfdXN0Y19lZHVfY24vRWdONEpjT0VfS05IcUc3Y29OT1RfYkFCWnZNV3BhSnhweTFKLTl5MWdkdUdjUT9ydGltZT1COWNHbWdQZjJFZw">here</a>.
</p><br />

**Data preparation:**

**COCO**
Download the dataset and extract it at {OMNIPOSE_ROOT}/data, as follows:

    ${OMNIPOSE_ROOT}
    |-- data
    `-- |-- coco
        `-- |-- annotations
            |   |-- person_keypoints_train2017.json
            |   `-- person_keypoints_val2017.json
            `-- images
                |-- train2017.zip
                `-- val2017.zip
                
**MPII**
Download the dataset and extract it at {OMNIPOSE_ROOT}/data, as follows:

    ${OMNIPOSE_ROOT}
    |-- data
    `-- |-- mpii
        `-- |-- annot
            |   |-- train.json
            |   `-- valid.json
            `-- images

#### Training
In order to train OmniPose uncomment the appropriate config file on 'run_train.sh' and run the following command:
```
bash run_train.sh
```

#### Testing
In order to test OmniPose uncomment the appropriate config file on 'run_test.sh' and run the following command:
```
bash run_demo.sh
```

#### Demo
In order to run demo of OmniPose on a sample image, uncomment the appropriate config file on 'run_demo.sh', and python command to the appropriate pre-trained dataset, and run the following command:
```
bash run_demo.sh
```
  
  
**Contact:**

<p align="justify">
Bruno Artacho:<br />
  E-mail: bmartacho@mail.rit.edu<br />
  Website: https://www.brunoartacho.com<br />
  
Andreas Savakis:<br />
  E-mail: andreas.savakis@rit.edu<br />
  Website: https://www.rit.edu/directory/axseec-andreas-savakis<br /><br />
</p>


**Citation:**

<p align="justify"> Artacho, B.; Savakis, A. OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation. in ArXiv, 2021. <br />

```
@InProceedings{Artacho_2021_ArXiv,
  title = {OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation},
  author = {Artacho, Bruno and Savakis, Andreas},
  eprint={2103.10180},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  year = {2021},
}
```

<p align="justify"> Artacho, B.; Savakis, A. UniPose+: A unified framework for 2D and 3D human pose estimation in images and videos. on IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021. <br />

```
@article{Artacho_2021_PAMI,
  title = {UniPose+: A unified framework for 2D and 3D human pose estimation in images and videos},
  author = {Artacho, Bruno and Savakis, Andreas},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year = {2021},
}
```

<p align="justify"> Artacho, B.; Savakis, A. UniPose: Unified Human Pose Estimation in Single Images and Videos. in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. <br />

```
@inproceedings{Artacho_2020_CVPR,
title = {UniPose: Unified Human Pose Estimation in Single Images and Videos},
author = {Artacho, Bruno and Savakis, Andreas},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2020}
}
```
