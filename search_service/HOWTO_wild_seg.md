# stylebot_wild_seg

## Preparation
Start by cloning this repository. Besides, you can save time with a docker container of the same name. In order to get the image, contact @dhl. We recommand preprocessing dataset locally and training the model on the container.

## Previous version instruction
Below is about real data. Previously, we studied the case with "datasets" folder. Everything is the same except preprocessing datasets. Give it a try on the toy example before the real data.

### Preprocess dataset
- Use jupyter/preprocess-dhl_r1.ipynb for tutorial.
- If you want to do it fast, use preprocess.py.

## Setup env

### Set base dir path
Do the following at the root of the repository (should have `stylebot_wild_seg` foldername)
```sh
export BASEDIR=$PWD
```

### Setup python

If you need to install python, I recommend installing [Anaconda](https://www.anaconda.com/) with python 3.8 (as of 2021.07.10)

For Linux 
```sh
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Choose `yes` for the following question
```sh
Do you wish the installer to initialize Anaconda3
by running conda init? [yes|no]
[no] >>> yes
```
And then do `source ~/.bashrc` to have conda activated.

Optionally, you can setup conda environment other than base.


Set up required packages:
```sh
cd $BASEDIR
pip install -r requirements.txt
```

### Preprocess dataset
For aimlk/stylebot_sb1_data_unloader:0.0.6(data in a docker image),
1. run preprocess_masking.py.
2. Then, use jupyter/preprocess_annotation.ipynb.

If you need to run a jupyter notebook server, do:
```sh
jupyter notebook
```

Example:
```sh
dhl@bbox1:~/stylebot_wild_seg$ jupyter notebook
[I 22:49:35.441 NotebookApp] Writing notebook server cookie secret to /home/dhl/.local/share/jupyter/runtime/notebook_cookie_secret
[I 22:49:35.541 NotebookApp] Serving notebooks from local directory: /home/dhl/stylebot_wild_seg
[I 22:49:35.541 NotebookApp] Jupyter Notebook 6.4.0 is running at:
[I 22:49:35.541 NotebookApp] http://localhost:8888/?token=b6bf9c566f5d783f7ef10a4b144af32570fa44848525961e
[I 22:49:35.541 NotebookApp]  or http://127.0.0.1:8888/?token=b6bf9c566f5d783f7ef10a4b144af32570fa44848525961e
[I 22:49:35.541 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 22:49:35.543 NotebookApp] No web browser found: could not locate runnable browser.
[C 22:49:35.543 NotebookApp]

    To access the notebook, open this file in a browser:
        file:///home/dhl/.local/share/jupyter/runtime/nbserver-3679-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=b6bf9c566f5d783f7ef10a4b144af32570fa44848525961e
     or http://127.0.0.1:8888/?token=b6bf9c566f5d783f7ef10a4b144af32570fa44848525961e
```

Open the [notebook](./src/preprocess-dhl_r1.ipynb) and run it for actual preprocessing.


## Installation

Match CUDA - Pytorch version
- If you have CUDA 10.x, then the usual Pytorch install does the right work.
- If you have CUDA 11.1, reinstall Pytorch with the following:
```sh
pip uninstall -y torch torchvision 
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Install detectron2 precompiled version for CUDA 11
```sh
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```
- This is CUDA 11.1
- CUDA version has to match your CUDA 

Detectron2 for CUDA 10 and torch 1.9 (precompiled):
```sh
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.9/index.html
```

### Install source codes for reference

```sh
cd $BASEDIR/src/checkpoints/
git clone -b v0.5 https://github.com/facebookresearch/detectron2.git
rm -rf $BASEDIR/src/checkpoints/detectron2_repo
mv detectron2 detectron2_repo
cd detectron2_repo
pip install -e .
```

## Training w/ detectron2

### Modification of the original code

We modify the detectron2 code a bit by doing the following:
```sh
cd $BASEDIR/src/checkpoints/detectron2_repo/projects/PointRend
cp $BASEDIR/src/checkpoints/train_net.py ./
cp $BASEDIR/src/checkpoints/go_train.sh ./
cp $BASEDIR/src/checkpoints/Base-RCNN-FPN.yaml ../../configs/
```
* Configuration with Base-RCNN-FPN.yaml
```
DATASETS:
  TRAIN: ("stylebot",)
  TEST: ("stylebot_test",)
SOLVER:
  IMS_PER_BATCH: 2
  BASE_LR: 0.01
  STEPS: (60000, 80000)
  MAX_ITER: 90000
```
The function `register_coco_instances` loads a single dataset at a time. However, multiple datasets can be set to both TRAIN and TEST in DATASETS. 

In the SOLVER section, you can make control of the number of image per batch, learning rate, optimizing steps and iteration.
### Begin Training
```sh
cd $BASEDIR/src/checkpoints/detectron2_repo/projects/PointRend
bash ./go_train.sh
```
### Resume training
In some cases, you need to stop training and restart it. train_net.py has option for that. Add option argument to go_train.sh with `--resume`. 
```
python train_net.py --config-file ./configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml --num-gpus 1 --resume
```
If you need more description, run 
```sh
./train_net.py -h
```
## Evaluation
As the desciption, you can evaluate using the method of COCO evaluator manually by adding `--eval-only`
```
python train_net.py \
--config-file ./configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_1x_coco.yaml \
--eval-only /path/to/checkpoint_file
```
* /path/to/checkpoint_file: pth file's path
### Visualization
We modify the custom codes a bit by doing the following:
```sh
cd $BASEDIR/src/checkpoints/detectron2_repo/projects/PointRend
cp $BASEDIR/src/checkpoints/visualize.py ./
cp $BASEDIR/src/checkpoints/test.py ./
```
* visualize.py: the predection about a single image 
* test.py: visualizing and saving the results with pickles on multiple images.
### Set metadata 
In order to show category names on the visualization, you should set `thing_classes`. It is identical to the categories used during preprocessing. Just remove category ids from cats in [preprocess.py](https://github.com/AIML-K/stylebot_wild_seg/blob/main/src/preprocess.py).

```
metadata = MetadataCatalog.get("stylebot").set(thing_classes=[
'shoes',
'bag', 
'necklace',
'skin',
...
])
```
### Configure model weights
If you want to apply your own model, insert the path of model_final.pth on cfg.MODEL.WEIGHTS.

```
cfg.MODEL.WEIGHTS = "/stylebot_wild_seg/src/checkpoints/detectron2_repo/projects/PointRend/output/model_final.pth"
```
