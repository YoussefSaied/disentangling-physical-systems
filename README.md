# disentangling-SDSC-fall-2020

## Usage
### Prerequisites
* python=3.8
* cartopy (pip requires sudo rights to install it, conda does not)
* scikit-image
* numpy
* torch
* torchvision
* matplotlib
* scipy
* imageio
* pandas
* seaborn
* Pillow
* tqdm
* netCDF4
* pickle

### Repos used in the code


### Preparation and for Climate
N.B. The code contains a lot of 'hacky' parts as the b-VAE was used with climate data only in the final week before the submission so the focus was on results and analysis.
* Cloning repository: 
	Download Victor's repo which I slightly modified to add a fully connected architecture for the b-VAE and a dataloader to load my pre-processed pickled dataset, run the script `move_src.sh` to move these files from the folder `Climate/src`  to Victor's repo.
* Use the Basic\_data\_preparation notebook to preprocess the data and save the preprocessed data before training. 


### Climate Notebooks
* `PCA`: All the PCA calculations and figures done for the climate data.
* `Basic\_data\_preparation`: Pre-processing and pickling pre-processed data to use for training.

### Lorenz Notebooks
N.B. these notebooks use modified src files from this git repo: [Gilpin's FNN network](https://github.com/williamgilpin/fnn). The actual network was not used in the report.
* `disentangling-lorenz-FNN`: FNN based disentanglement for Lorenz and other figures and calculations.
* `disentangling-lorenz-VAE`: b-VAE based disentanglement for Lorenz and other figures and calculations.
* `disentangling-lorenz-PCA`: PCA based disentanglement for Lorenz and other figures and calculations.
* `disentangling-lorenz-basic`: Basic state space reconstruction calculations and figures.

### Notebooks not used in the report _not commented_
* `climate\_training\_DK`: DeepKoopman applied to the climate data
* `climate\_training\_DK\_noPC`: DeepKoopman applied to the climate data after subtracting the seasonal cycle
* `climate\_training\_DK_control`: DeepKoopman applied to the climate data with control variables (a model I created in an attempt to isolate the intermittent signals)
* `pendulum`: DeepKoopman applied to a pendulum.
