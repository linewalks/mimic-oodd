# mimic-oodd
Out of Distribution Detection of MIMIC3


## Setting

1. Copy config.default.json to config.json
```
cp config.default.json config.json
```
2. change DB_URI in config.json

3. Install python packages.

```
pip install -r requirements.txt
```

## Run

#### RNN

```
# Run by default setting
python run_rnn.py

# See help for more tunable parameters.
python run_rnn.py --help
```

#### Cox

```
# Run by default setting
python run_cox.py

# See help for more tunable parameters.
python run_cox.py --help
```

#### DeepSurv

```
# Run by default setting
python run_deepsurv.py

# See help for more tunable parameters.
python run_deepsurv.py --help
```

#### NN

* Run Neural Network Model with Survival Data
  * Duration data will not be used.

```
# Run by default setting
python run_nn.py

# See help for more tunable parameters.
python run_nn.py --help
```

#### RNN Survival

* Run Neural Network Model with Survival Data
  * Duration data will not be used.

```
# Run by default setting
python run_rnn_survival.py

# See help for more tunable parameters.
python run_rnn_survival.py --help
```
