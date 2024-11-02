# EventHDR
 This is the implementation and dataset for [Learning To Reconstruct High Speed and High Dynamic Range Videos From Events](https://openaccess.thecvf.com/content/CVPR2021/papers/Zou_Learning_To_Reconstruct_High_Speed_and_High_Dynamic_Range_Videos_CVPR_2021_paper.pdf), CVPR 2021, and its journal extension [EventHDR: From Event to High-Speed HDR Videos and Beyond](https://ieeexplore.ieee.org/abstract/document/10713104), TPAMI 2024

## News
* 2024/10/09: The [**journal extension**](https://ieeexplore.ieee.org/abstract/document/10713104) of EventHDR has been accepted by TPAMI 2024
* 2023/09/19: Both the training and testing datasets are avalable at [**OneDrive**](https://1drv.ms/f/s!AuA3qjJbfh9FjQa4GvHC_9Fn9UQm?e=jODI9N)
* 2023/09/27: Release the evaluation code and pre-trained model
## Introduction
In this work, we present
a convolutional recurrent neural network which takes a
sequence of neighboring event frames to reconstruct high speed HDR videos. To facilitate the process of network learning, we design a novel optical system and collect a real-world dataset with paired high speed HDR videos and event streams

## Highlights
* We propose a convolutional recurrent neural network for the reconstruction of high speed HDR videos from events. Our architecture carefully considers the alignment and temporal correlation for events

<img src="figs/overview.png" width="500px"/>

* To bridge the gap between simulated and real HDR videos, we design an elaborate system to synchronously capture paired high speed HDR video and the corresponding event stream

<img src="figs/system.png" width="500px"/>

* We collect a real-world dataset with paired high speed HDR videos (high-bit) and event streams. Each frame of our HDR videos are merged from two precisely synchronized LDR frames

<img src="figs/dataset.png" width="500px"/>

## Dataset
* Download our EventHDR dataset at [**OneDrive**](https://1drv.ms/f/s!AuA3qjJbfh9FjQa4GvHC_9Fn9UQm?e=jODI9N)
* Both the training and testing data are stored in ```hdf5``` format, you can use ```h5py``` tool to read the data. To create a pytorch dataset, please refer to [**DynamicH5Dataset**](https://github.com/TimoStoff/event_cnn_minimal/blob/master/data_loader/dataset.py)
* We build our dataset based on the data structure of [**events_contrast_maximization**](https://github.com/TimoStoff/events_contrast_maximization/tree/master), and the structure tree is shown below
```
EventHDR Dataset Structure Tree
+-- '/'
|   +-- attribute "duration"
|   +-- attribute "num_events"
|   +-- attribute "num_flow"
|   +-- attribute "num_neg"
|   +-- attribute "num_pos"
|   +-- attribute "sensor_resolution"
|   +-- attribute "t0"
|   +-- group "events"
|   |   +-- dataset "ps"
|   |   +-- dataset "ts"
|   |   +-- dataset "xs"
|   |   +-- dataset "ys"
|   |   
|   +--	group "images"
|   |   +-- attribute "num_images"
|   |   +-- dataset "image000000000"
|   |   |   +-- attribute "event_idx"
|   |   |   +-- attribute "size"
|   |   |   +-- attribute "timestamp"
|   |   |   +-- attribute "type"
|   |   |   
|   ...
|   +--	group "images"
|   |   +-- attribute "num_images"
|   |   +-- dataset "image000002827"
|   |   |   +-- attribute "event_idx"
|   |   |   +-- attribute "size"
|   |   |   +-- attribute "timestamp"
|   |   |   +-- attribute "type"
|   |   |   
|   |   
|   
```
## Usage
* Create ```conda``` environment and download our repository
```
conda create -n eventhdr python=3.6
conda activate eventhdr
git clone https://github.com/jackzou233/EventHDR
cd EventHDR
pip install -r requirements.txt
```
* Compile dependencies for deformable convolution with
```
BASICSR_EXT=True python setup.py develop
```
* Download our ```hdf5``` foramt evaluation data from [**OneDrive**](https://1drv.ms/f/s!AuA3qjJbfh9FjQa4GvHC_9Fn9UQm?e=jODI9N), and put it in folder ```./eval_data```
* Download the pre-trained model at [**pretrained**](https://1drv.ms/u/s!AuA3qjJbfh9FjVhdCwPFPSiy-r94?e=VpdPU4), and move it to ```./pretrained```
* Run the following script to reconstruct HDR videos from event streams!
```
bash test.sh
```
* To make a comparing video of the reconstruction, please run
```
python mk_video.py
```
Then, you will obtain results like
<img src="figs/preview.gif"/>
## Results
The quantitative results are shown below
| Method  | PSNR | SSIM | LPIPS | TC       |
| ------- | ---- | ---- | ----- | -------- |
| HF | 10.99 | 0.2708 | 0.4434 | 1.2051 |
| E2VID | 12.78 | 0.5753 | 0.3541 | 1.0305 |
| EF | 13.23 | 0.5914 | 0.3030 | 0.9729 |
| Ours | 15.31 | 0.7084 | 0.2424 | 0.5198 |
## Citation
If you find this work useful for your research, please cite: 
```
@inproceedings{zou2021learning,
  title={Learning To Reconstruct High Speed and High Dynamic Range Videos From Events},
  author={Zou, Yunhao and Zheng, Yinqiang and Takatani, Tsuyoshi and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2024--2033},
  year={2021}
}
```
```
@article{zou2024eventhdr,
  title={EventHDR: From Event to High-Speed HDR Videos and Beyond},
  author={Zou, Yunhao and Fu, Ying and Takatani, Tsuyoshi and Zheng, Yinqiang},
  journal={IEEE Trans. Pattern Anal. Mach. Intell.},
  note={{E}arly Access},
  year={2024},
  pages={1--18},
  publisher={IEEE}
}
```
## contact
If you have any problems, please feel free to contact me at zouyunhao@bit.edu.cn
##
## Acknowlegment
The code borrows from [**event_cnn_minimal**](https://github.com/TimoStoff/event_cnn_minimal), [**EDVR**](https://github.com/xinntao/EDVR) and [**E2VID**](https://github.com/uzh-rpg/rpg_e2vid), please also cite their great work