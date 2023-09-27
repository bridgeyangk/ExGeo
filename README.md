# ExGeo
![](https://img.shields.io/badge/python-3.8.13-green)
![](https://img.shields.io/badge/pytorch-1.12.1-green)
![](https://img.shields.io/badge/cudatoolkit-11.6.0-green)
![](https://img.shields.io/badge/cudnn-7.6.5-green)

This folder provides a reference implementation of **ExGeo**.


## Basic Usage

### Requirements

The code was tested with `python 3.8.13`, `pytorch 1.12.1`,  `cudatoolkit 11.6.0`, and `cudnn 7.6.5`. Install the dependencies via [Anaconda](https://www.anaconda.com/):

```shell
# create virtual environment
conda create --name ExGeo python=3.8.13

# activate environment
conda activate ExGeo

# install pytorch & cudatoolkit
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# install other requirements
conda install numpy pandas
pip install scikit-learn
```

### Run the code

```shell
# Open the "ExGeo" folder
cd ExGeo

# data preprocess (executing IP clustering). 
python generateidx.py --dataset "New_York"
python generateidx.py --dataset "Los_Angeles"
python generateidx.py --dataset "Shanghai"

python preprocess.py --dataset "New_York"
python preprocess.py --dataset "Los_Angeles"
python preprocess.py --dataset "Shanghai"
# run the model ExGeo
python main.py --dataset "New_York" --dim_in 30 --lr 2e-3 --saved_epoch 10
python main.py --dataset "Los_Angeles" --dim_in 30 --lr 2e-3 --saved_epoch 10
python main.py --dataset "Shanghai" --dim_in 51 --lr 1e-3 --saved_epoch 10

# load the checkpoint and then test
python test.py --dataset "New_York" --dim_in 30 --lr 2e-3 --load_epoch 100
python test.py --dataset "Los_Angeles" --dim_in 30 --lr 2e-3 --load_epoch 100
python test.py --dataset "Shanghai" --dim_in 51 --lr 1e-3 --load_epoch 70
```

## The description of hyperparameters used in main.py

| Hyperparameter   | Description                                                  |
| :--------------- | ------------------------------------------------------------ |
| seed             | the random number seed used for parameter initialization during training |
| model_name       | the name of model                                            |
| dataset          | the dataset used by main.py                                  |
| lambda_1         | the trade-off coefficient of data perturbation in loss function |
| lambda_2         | the trade-off coefficient of parameter perturbation in loss function |
| lr               | learning rate                                                |
| harved_epoch     | when how many consecutive epochs the performance does not increase, the learning rate is halved |
| early_stop_epoch | when how many consecutive epochs the performance does not increase, the training stops. |
| saved_epoch      | how many epochs to save checkpoint for the testing           |
| dim_in           | the dimension of input data                                  |
| dim_med          | the dimension of middle layers                               |
| dim_z            | the dimension of vector representation                       |
| eta              | magnitude of data disturbance                                |
| zeta             | magnitude of parameter disturbance                           |
| step             | times of gradient ascent in a single parameter disturbance   |
| mu               | inner learning rate of parameter disturbance                 |
| c_mlp            | when predicting if use collaborative_mlp or not              |
| epoch_threshold  | when we start adding perturbation both in data and parameter |



## Folder Structure

```tex
└── ExGeo
	├── datasets # Contains three large-scale real-world street-level IP geolocation datasets.
	│	|── New_York # Street-level IP geolocation dataset collected from New York City including 91,808 IP addresses.
	│	|── Los_Angeles # Street-level IP geolocation dataset collected from Los Angeles including 92,804 IP addresses.
	│	|── Shanghai # Street-level IP geolocation dataset collected from Shanghai including 126,258 IP addresses.
	├── lib # Contains model implementation files
	│	|── layers.py # The code of the attention mechanism.
	│	|── model.py # The core source code of proposed RIPGeo
	│	|── sublayers.py # The support file for layer.py
	│	|── utils.py # Auxiliary functions
	├── asset # Contains saved checkpoints and logs when running the model
	│	|── log # Contains logs when running the model 
	│	|── model # Contains the saved checkpoints
        ├── generateidx.py # generate the idx of traget nodes and landmark nodes
	├── preprocess.py # Preprocess dataset and execute IP clustering the for model running
	├── main.py # Run model for training and test
	├── test.py # Load checkpoint and then test
	└── README.md
```

## Dataset Information

The "datasets" folder contains three subfolders corresponding to three large-scale real-world street-level IP geolocation    datasets collected from New York City, Los Angeles and Shanghai. There are three files in each subfolder:

- data.csv    *# features (including attribute knowledge and network measurements) and labels (longitude and latitude) for street-level IP geolocation* 
- ip.csv    *# IP addresses*
- last_traceroute.csv    *# last four routers and corresponding delays for efficient IP host clustering*

The detailed **columns and description** of data.csv in New York dataset are as follows:

#### New York  

| Column Name                     | Data Description                                             |
| ------------------------------- | ------------------------------------------------------------ |
| ip                              | The IPv4 address                                             |
| as_mult_info                    | The ID of the autonomous system where IP locates             |
| country                         | The country where the IP locates                             |
| prov_cn_name                    | The state/province where the IP locates                      |
| city                            | The city where the IP locates                                |
| isp                             | The Internet Service Provider of the IP                      |
| vp900/901/..._ping_delay_time   | The ping delay from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._trace             | The traceroute list from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._tr_steps          | #steps of the traceroute from probing hosts "vp900/901/..." to the IP host |
| vp900/901/..._last_router_delay | The delay from the last router to the IP host in the traceroute list from probing hosts "vp900/901/..." |
| vp900/901/..._total_delay       | The total delay from probing hosts "vp900/901/..." to the IP host |
| longitude                       | The longitude of the IP (as label)                           |
| latitude                        | The latitude of the IP host (as label)                       |

PS: The detailed columns and description of data.csv in other two datasets are similar to New York dataset's.

