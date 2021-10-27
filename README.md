# MGD_model_training

Platform: CUDA 11.0, Python 3.7.7

Python Dependencies:

	torch==1.7.1+cu110,
	torchvision==0.8.2+cu110,
	sklearn==0.0
	tqdm==4.61.2,
	pillow==8.3.1,
	pandas==1.3.0,
	opencv-python==4.5.2.54,
	numpy==1.21.0

<== Getting start ==>

Install from Github:

	git clone https://github.com/EysenYang/MGD_model_training

	cd MGD_model_training

	unzip data.zip

(To use the example data, please unzip the data.zip and leave it in the original path.)

<== Setting Environment Guide (Install from PyPi) ==>

	pip3 install -r requirements.txt


<== Running the example script ==>

Training:

	python3 training.py --train-file ./trn.csv --val-file ./val.csv --epochs 120 -j 2 --output-dir ./resnet34_Ep120_wd1e4

Predicting:

	python3 predict.py 
