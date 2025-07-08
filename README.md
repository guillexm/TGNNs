I'v added my dataset work to the github under the folder dataworkspace. There are also two extra files/folders ("dataworkspace.egg-info" and pyproject.toml), these are to make dataworkspace into a python package, making import statements simpler inside the code.

The dataworkspace folder structure was build with the idea of having a generic model and dataset class (model.base.py and dataset.base.py) to allow for a lot of code reuse in case we want to run many models on many datasets. For now most of it is boilerplate code, what I've done so far is:
1. Downloaded most datasets
2. Formatted the data into panda dataframes that were exported to dataset_name.csv files.

Each dataset is under dataworkspace/datasets/DATASET_NAME, except for the metrla and and pemsbay datasets, they are under torch_traffic/DATASET_NAME

See below for the exact status as to all the datasets that I previously identified as interesting:

| Dataset         | Status                           |
|-----------------|----------------------------------|
| seattle_bike | <span style="color:#e74c3c;">Download unavailable when I checked</span> |
| pems | <span style="color:#e74c3c;">Download unavailable when I checked</span> |
| metrla | <span style="color:#2ecc71;">Clean dataframe in csv</span> |
| pems_bay | <span style="color:#2ecc71;">Clean dataframe in csv</span> |
| melpeds | <span style="color:#2ecc71;">Clean dataframe in csv</span> |
| large_st | <span style="color:#f1c40f;">Held off: Large Dataset</span> |
| louisville_cycling | <span style="color:#f1c40f;">Held off: Complex Dataset</span> |
| chicago_cycling | <span style="color:#f1c40f;">Held off: Complex Dataset</span> |
| opensky | <span style="color:#f1c40f;">Held off: Large and Complex Dataset</span> |
| ne_iso | <span style="color:#2ecc71;">Clean dataframe in csv</span> |
| ny_iso | <span style="color:#2ecc71;">Clean dataframe in csv</span> |
| elergone | <span style="color:#2ecc71;">Clean dataframe in csv</span> |
| camels_us | <span style="color:#f1c40f;">Held off: Large and Complex Dataset</span> |
| nyt_covid | <span style="color:#f1c40f;">Held off: Large and Complex Dataset</span> |
| financial_data | <span style="color:#f1c40f;">Held off: need to discuss specs</span> |


Explanation of the different status:
-Clean dataframe in csv: I have downloaded the dataset and turned it into a panda dataframe that I exported to a csv. This should make them very easy to work with whenever we need them.
-Helf off: Some datasets being hard to work with, I'm holding off on generating the dataframe until we are sure we will be using them.
-Download unavailable when I checked: Self explanatory. However the dataset could still be available at other sources.

