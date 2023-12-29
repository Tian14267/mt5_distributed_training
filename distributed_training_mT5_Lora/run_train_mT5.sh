#TOT_CUDA="2,3"
#CUDA_VISIBLE_DEVICES="1,2" horovodrun -np 2 python mT5_train.py

#####  单卡
CUDA_VISIBLE_DEVICES=3 python mT5_train.py