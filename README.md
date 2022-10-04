**HuBMAP + HPA - Hacking the Human Body** - Rank 178/1175 (16%) My Profile [Here!](https://www.kaggle.com/sfgx8801234 "Here!")


Install step on server(conda)
````python
conda create --name env_lian_mmsegmentation python=3.7
conda activate env_lian_mmsegmentation
pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6/index.html
git clone https://github.com/open-mmlab/mmsegmentation.git 
cd mmsegmentation
git checkout -f cd18b6d
pip install -e .
````

Install step use pip in inference nootbook

Download Dataset [here!](https://drive.google.com/drive/folders/1gFFOV3eZkB5ISo5hIzPftnVXvhbV1zn9?usp=sharing "here!") and put in root directory
Use command to train swin transformer
````python
python tools/train.py configs/HuBMAP_Mulit/upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K.py --seed 69 --deterministic
````

The log file, config file,weight will save in ./work_dirs_HuBMAP_mulit/ upernet_swin_base_patch4_window12_512x512_160k_ade20k_pretrain_384x384_22K