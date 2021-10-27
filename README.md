# MGD_model_training

Platform: CUDA 10.2, Python 3.7.7

Python Dependencies:

torch,
torchvision,
datetime,
time,
utils,
apex,
tqdm,
pillow,
pandas,
opencv-python,
numpy

Getting start ==>

Install from Github:

git clone https://github.com/EysenYang/MGD_model_training

cd MGD_model_training

unzip data.zip

(To use the example data, please unzip the data.zip and leave it in the original path.)

Setting Environment Guide (Install from PyPi) ==>

pip3 install -r requirements.txt


Running the example script ==>

Training:

python3 training.py --train-file ./trn.csv --val-file ./val.csv --epochs 120 -j 2 --output-dir ./resnet34_Ep120_wd1e4

Predicting:

python3 predict.py 
