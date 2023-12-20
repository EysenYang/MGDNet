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

The final checkpoint files will be generated in the output folder and will appear in the form of model_xxx.pth.

Predicting:

	python3 predict.py 


Expected output:

	image_name	label	Normal	NMGO	MGA	MGO	MGAO	MGOO
	./data/N/1002.jpg	0	0.758973539	0.062725827	5.93E-05	3.37E-05	0.174801379	0.003406364
	./data/N_K/1940.jpg	1	0.001687979	0.926098049	3.35E-05	8.07E-06	0.000528317	0.07164406
	./data/W/1080.jpg	2	0.001754218	0.002595291	0.982559681	0.008812006	0.003825066	0.000453728
	./data/W_Z/551.jpg	3	0.000158101	0.000112213	0.021836495	0.976503253	0.001225252	0.000164684
	./data/Z/1135.jpg	4	0.069005467	0.132829428	0.396607846	0.091052495	0.13766253	0.172842279
	./data/Z_K/3082.jpg	5	0.020390244	0.142778814	0.077948295	0.352566421	0.16126579	0.24505043
	

The expected output will be a CSV file containing 8 columns of information, where the first column is the image name, the second column is the label, and the third to eighth columns are the predicted probabilities for the six subtypes of the current image.

Classification Standard:
Normal group: meibomian glands and ducts are normal. 
NMGO group: full meibomian glands with opening.
MGA group: tire-like meibomian gland epithelial cells are absent, the acinar wall disappears, becomes smaller, or the fiber cord changes.
MGO group: enlarged acinus and blocked meibum in the acinus.
MGAO group: shows the characteristics of both the atrophy group and obstruction group.
MGOO group: normal meibomian gland opening with highly dilated meibomian gland secretory tube and blocked meibomian gland acinus.

