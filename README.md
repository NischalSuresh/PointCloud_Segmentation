
Code for "VLR Project by Avi and Nischal",


## Install
```
conda create -n FPSNet python=3.7
source activate FPSNet
cd /ROOT/
pip install -r requirements.txt
```

# Dataset
Download SemanticKITTI dataset from [official website](http://www.semantic-kitti.org/)
The dataset structure should be 
```
./
├── 
├── ...
└── path_to_data_shown_in_config/
      └── sequences
            ├── 00/           
            │   ├── velodyne/	
            |   |	   ├── 000000.bin
            |   |	   ├── 000001.bin
            |   |	   └── ...
            │   ├── labels/ 
            |   |      ├── 000000.label
            |   |      ├── 000001.label
            |   |      └── ...
            |   ├── calib.txt
            |   ├── poses.txt
            |   └── times.txt
            └── 08/
```

## Train
Revise dataset path in train.sh and run
```
cd /train/tasks/semantic
sh train.sh
```

## Inference and Test
Revise dataset path in test.sh and run
```
cd /train/tasks/semantic
sh test.sh

## Citation
If you use this code, please cite:
```

```
## Acknowledgement
Part of code is borrowed from [lidar-bonnetal](https://github.com/PRBonn/lidar-bonnetal), thanks for their sharing!
## Related Repos
- [SynLiDAR: Learning From Synthetic LiDAR Sequential Point Cloud for Semantic Segmentation](https://github.com/xiaoaoran/SynLiDAR)
- [Unsupervised Representation Learning for Point Clouds: A Survey](https://github.com/xiaoaoran/3d_url_survey)
