<p align="center">
  <h1 align="center">Vision-Guided Action: Enhancing 3D Human Motion Prediction with Gaze-informed Affordance in 3D Scenes</h1>
  <p align="center">
    <a>Ting Yu</a><sup>1</sup>
    &nbsp;·&nbsp;
    <a href="https://lin-yi-er.github.io/homepage/">Yi Lin</a><sup>1</sup>
    &nbsp;·&nbsp;
    <a>Jun Yu</a><sup>2</sup>
    &nbsp;·&nbsp;
    <a>Zhenyu Lou</a><sup>1</sup>
    <a>Qiongjie Cui</a><sup>1</sup>
  </p>
  <p align="center">
    <sup>1</sup>Hangzhou Normal University
    <br>
    <sup>2</sup>Harbin Institute of Technology
    <br>
    <sup>3</sup>Zhejiang University
    <br>
    <sup>4</sup>Singapore University of Technology and Design
  </p>
  <h3 align="center">CVPR 2025</h3>



## Setup

---

### Step 1: Install Required Dependencies

Begin by installing the required Python packages listed in `requirements.txt`:

```
pip install -r requirements.txt
```



### Step 2: Install PointNet++, SoftGroup, Affordancenet 

Clone the PointNet++ repository and follow the instructions provided in [this link](https://github.com/daerduoCarey/o2oafford/tree/main/exps):

```
git clone --recursive https://github.com/erikwijmans/Pointnet2_PyTorch
cd Pointnet2_PyTorch
```

**Important**: You need to modify the code in the repository to avoid issues with the build. Specifically:

- Comment out lines 100-101 in `sampling_gpu.cu`:

  ```
  # https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu#L100-L101
  ```

- Edit lines 196-198 in `pointnet2_modules.py` (located in `[PATH-TO-VENV]/lib64/python3.8/site-packages/pointnet2_ops/`):

  ```
  interpolated_feats = known_feats.repeat(1, 1, unknown.shape[1])
  ```



Clone the Softgroup repository and follow the instructions provided in [this link](https://github.com/thangvubk/SoftGroup.git):

~~~
git clone --recursive https://github.com/thangvubk/SoftGroup.git
cd Softgroup
~~~

Softgroup provides the custom dataset guide, so you can process scene data youself, or use the results we processed.


Clone the Affordancenet repository and follow the instructions provided in [this link](https://github.com/Gorilla-Lab-SCUT/AffordanceNet.git):

~~~
cd model
git clone --recursive https://github.com/Gorilla-Lab-SCUT/AffordanceNet.git
cd AffordanceNet
~~~

you can easily train the affordancenet following the official implementation.

1. download the affordance [dataset](https://drive.google.com/drive/folders/1s5W0Nfz9NEN8gP14tge8GuouUDXs2Ssq?usp=sharing) 

2. train

   ~~~bash
   python train.py config/dgcnn/estimation_cfg.py --work_dir TPATH_TO_LOG_DIR --gpu 0,1
   ~~~

3. test

   ~~~bash
   python test.py config/dgcnn/estimation_cfg.py --work_dir PATH_TO_LOG_DIR --gpu 0,1 --checkpoint PATH_TO_CHECKPOINT
   ~~~



After making the changes, run the following commands to install dependencies:

```
pip install -r requirements.txt
```



### Step 3: Install Additional Dependencies

Download and install the following dependencies:

- [Vposer](https://github.com/nghorbani/human_body_prior)
- [SMPL-X](https://github.com/vchoutas/smplx)



## Dataset

---

The GAP3DS method utilizes a standard-processed dataset. However, due to confidentiality constraints, we are unable to release the processed version.

To obtain the raw dataset, please follow the instructions provided in the official [GIMO repository](https://github.com/y-zheng18/GIMO?tab=readme-ov-file#dataset).

After downloading and unzipping the raw dataset, the directory structure should look like the following:

```
--data_root
     |--bedroom0122
           |--2022-01-21-194925
                 |--eye_pc
                 |--PV
                 |--smplx_local
                 |--transform_info.json
                 ...
           |--2022-01-21-195107
           ...
     |--bedroom0123
     |--bedroom0210
     |--classroom0219
     ...
```



On the first run, our code will automatically preprocess the data. Ensure that the **dataroot** is correctly set before running the program. After preprocessing, the dataset will be stored in the same location as the raw dataset, and the folder structure will be as follows:

```
--data_root
      |--SLICES_8s
            |--train
                 |--gazes.pth
                 |--joints_input.pth
                 |--joints_label.pth
                 |--poses_input.pth
                 |--poses_label.pth
                 |--scene_points_<sample_points>.pth
            |--test
                 |--gazes.pth
                 |--joints_input.pth
                 |--joints_label.pth
                 |--poses_input.pth
                 |--poses_label.pth
                 |--scene_points_<sample_points>.pth
     |--bedroom0122
     |--bedroom0123
     |--bedroom0210
     |--classroom0219
     ...
```



## Quickstart Guide

---

### Evaluation

To evaluate the model, execute the following command:

```
bash scripts/eval.sh
```

[the weight's results you can use](https://drive.google.com/file/d/1i2kASdLfNJ9tlftgtzWTsnJYY4Eo-OYM/view?usp=drive_link)

### Training

To train the model, use the following command:

```
bash scripts/train.sh
```
