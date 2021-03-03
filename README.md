# OmniPose

  <a href="http://www.brunoartacho.com">**OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation**</a>.
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
  <img src="https://people.rit.edu/bm3768/images/waspv2.png" width=500 title="WASPv2 module"><br />
  Figure 2: WASPv2 Module.
</p><br />

Examples of the OmniPose architecture for Multi-Person Pose Estimation are shown in Figures 3.<br />

<p align="center">
  <img src="https://people.rit.edu/bm3768/images/COCO_sample.png" width=500 title="WASP module"><br />
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
The pre-trained weights can be downloaded at
  <a href="">here</a>.
</p><br />


**Contact:**

<p align="justify">
Bruno Artacho:<br />
  E-mail: bmartacho@mail.rit.edu<br />
  Website: https://people.rit.edu/bm3768<br />
  
Andreas Savakis:<br />
  E-mail: andreas.savakis@rit.edu<br />
  Website: https://www.rit.edu/directory/axseec-andreas-savakis<br /><br />
</p>

**Citation:**

<p align="justify">
Artacho, B.; Savakis, A. UniPose: Unified Human Pose Estimation in Single Images and Videos. in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020. <br />

Latex:<br />
@InProceedings{Artacho_2020_CVPR,<br />
  title = {OmniPose: A Multi-Scale Framework for Multi-Person Pose Estimation},<br />
  author = {Artacho, Bruno and Savakis, Andreas},<br />
  booktitle = {Arxiv},<br />
  year = {2021},<br />
}<br />
</p>
