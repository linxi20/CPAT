## 1. Environment Configuration
```
numpy==1.19.5 
matplotlib==3.3.4
pandas==1.1.5
scikit-learn==0.24.2 
scipy==1.5.4   
einops==0.4.1
torch==1.10.0
torchaudio==0.10.0
torchvision==0.11.0
cudatoolkit==11.3.1
gpu=NVIDIA Tesla P100-SXM2 GPU
gpu memory=16384 MiB
cpu=Intel(R) Xeon(R) Silver 4216 CPU @ 2.10GHz
```
1. Create a virtual environment: `conda create -n pytorch_cuda python=3.6`
2. Activate the environment:  `conda activate pytorch_cuda`
3. Install the gpu version of Pytorch: `conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3`
4. Install other required packages.
## 2. Data
The folder `./dataset/` holds the *8* benchmark datasets used in the experiment.
## 3. Code Download
```
git clone https://github.com/linxi20/CPAT.git
```
## 4. Experimental Parameters
The parameter settings for all datasets are stored in the *sh* script file under the folder `./scripts/`.
## 5. Model Training and Evaluation
1、We provide the running scripts for all benchmark tests under the folder `./scripts/`. You can reproduce the experiments results by the following example:
- Switch to the folder `CPAT/scripts`:
	```
	cd CPAT/scripts
	```
- Submit all the script files below:
	```
	# Multivariate forecasting with CPAT
	nohup ./illness.sh > illness.log 2>&1 
	nohup ./etth1.sh > etth1.log 2>&1 
	nohup ./etth2.sh > etth2.log 2>&1 
	nohup ./ettm1.sh > ettm1.log 2>&1 
	nohup ./ettm2.sh > ettm2.log 2>&1 
	nohup ./weather.sh > weather.log 2>&1 
	nohup ./electricity.sh > electricity.log 2>&1 
	nohup ./traffic.sh > traffic.log 2>&1 
	```
2、After the scripts runs, the folder `./logs/LongForecasting/` is created in the current directory, which stores the log files recording the training process. The experimental results can be viewed via `result.txt` after the training is completed and the prediction accuracy of the model can be evaluated based on the *MSE* and *MAE* metrics.
