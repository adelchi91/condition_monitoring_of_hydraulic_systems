# condition_monitoring_of_hydraulic_systems

The data used comes from `https://archive.ics.uci.edu/dataset/447/condition+monitoring+of+hydraulic+systems`

The idea is to create a model that predicts whether a valve is in good condition or not. The information can be found in `./data_subset/profile.txt`. The data we leveraged is 
* PS2 (Pressure (bar) 100Hz sampling) `./data_subset/PS2.txt`
* FS1 (Volume flow (l/min) 10Hz sampling) `./data_subset/FS1.txt`
* Profile : "valve condition" variable is the column of interest here.  
The valve is has optimal condition if equal to 100%.

The model is a binary classification model that can be launched via `./src/condition_monitoring_of_hydraulic_systems.py`. 

Two notebooks with some exploratory tests can be found in `./notebook/`.

A fastapi app which takes PS2 and FS1 timeseries as input is found in `./app/fast_api.py`. There may be a current bug in the preprocessing part, when tsfresh should be carried out. Ongoing further checks. 

A containerized version of the app can be retrieved using the `Dockerfile`. 