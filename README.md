
> **MageRec:A Parallel Mamba–Attention Gated Network for Sequential Recommendation**\

### environment dependencies

The following are the main runtime environment dependencies for running the repository：
- cuda 11.8
- python 3.10.14
- pytorch 2.3.0
- recbole 1.2.0
- mamba-ssm 2.2.2
- casual-conv1d 1.2.2
- numpy 1.26.4

If you are having trouble installing Mamba, please refer to the installation tutorial we wrote: [https://github.com/AlwaysFHao/Mamba-Install](https://github.com/AlwaysFHao/Mamba-Install).

You can also view detailed environment information in File [environment.yaml](environment.yaml).

###  DataSets
datasets are provided by [RecBole](https://github.com/RUCAIBox/RecSysDatasets)


##  Run
After preparing all the necessary files and runtime environment, please modify the configuration file path in [`🐍 run.py`](run.py) in the following format:
```python
config = Config(model=MageRec, config_file_list=['config/{dataset_name}.yaml'])
```
Just run it directly：
```shell
python run.py
```
If you want to continue checkpoint training, you need to add the model weight path to the `checkpoint_path` configuration item in the corresponding configuration file. 
```yaml
checkpoint_path: saved/model_weight_name.pth
```

### Baseline(Optional)
You can directly select the baseline model we have organized in the [`📁 baseline`](baseline) folder for training, taking `SASRec` as an example.
```shell
cd ./baseline/SASRec
python run.py
