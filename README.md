# 2025-AGILE-AircraftTrajectoryPrediction

Code supporting the experiements presented on the paper "Experiments on Geospatial Data Modelling for Long-Term Trajectory Prediction of Aircrafts". The paper was accepted on the AGILE 2025 conference. The aim of the project was to explore geospatial data modeling, specially the H3 index, for trajectory predicitons of Aircraft using a CVAE-LSTM model.

## Pre-requisites
* Create a virtual environment with Python 3.11.10
* Create an account in Opensky (optional, for using the [download_raw.py](./data/download_raw.py))
    * store the credentials inside [secrets.conf](./src/secrets.conf)


## Installation

For installing all the dependencies, you can use the installation script on this repository. To do so, please execute the following code on a terminal **inside the folder of this project** on a virtual environment with Python 3.11.10:

```bash
chmod +x install.sh
bash install.sh
```

## Reproduce the Experiments
To reproduce the experiments of the paper, you can use the already downloaded data inside [src/data/raw/](./src/data/raw/). However, more data, or data from different airports can be download using the code provided in the [download_raw.py](./src/data/download_raw.py) file. To download the exact same dates used on the experiments for the paper, you can use the code as it is by executing the following commands:

```bash
source .venv/bin/activate
cd ./src/data/
python3 download_raw.py 
```

### Experiments
All the experiments were executed using the code on [experiments.ipynb](./experiments.ipynb). To try different configurations, you can change the parameters in the **Experiment Setup** section. 

#### Different H3 Resolutions
Here, we fixed IN_SEQ_LEN, TGT_SEQ_LEN, and SAMPLING_TIME, while trying H3_RESOLUTION between 5 and 11. To use different resolutions, change the H3_RESOLUTION parameter, to try different geodata representations, change the GEODATA_REPRESENTATION parameter.

Example of Experiment Setup:
```python
    GEODATA_REPRESENTATION : Literal["h3", "xy", "latlon"] = "h3"
    H3_RESOLUTION : Literal["5", "6", "7", "8", "9", "10", "11"] = "5"
    IN_SEQ_LEN = 5 #Fixed
    TGT_SEQ_LEN = 60 #Fixed
    SAMPLING_TIME = 60 #Fixed
```

#### Different Long-term Prediction Ranges
Here, we fixed IN_SEQ_LEN, and H3_RESOLUTION, while trying different TGT_SEQ_LEN, and SAMPLING_TIME. To use different geodata representations, change the GEODATA_REPRESENTATION parameter.

Example of Experiment Setup:
```python
GEODATA_REPRESENTATION : Literal["h3", "xy", "latlon"] = "h3"
H3_RESOLUTION : Literal["5", "6", "7", "8", "9", "10", "11"] = "5" #Fixed
IN_SEQ_LEN = 2 #Fixed
TGT_SEQ_LEN = 60
SAMPLING_TIME = 60
```

The different combinations we used for TGT_SEQ_LEN and SAMPLING_TIME are shown in the table bellow:

| TGT_SEQ_LEN | SAMPLING_TIME |
| --- | --- |
| 60 | 60 |
| 30 | 60 |
| 120 | 30 |
| 60 | 30 |
| 180 | 20 |
| 90 | 20 |
| 360 | 10 |
| 180 | 10 |
| 720 | 5 |
| 360 | 5 |
