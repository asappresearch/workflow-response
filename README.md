# Workflow-guided Response Generation
This is the code repository for the paper "Workflow-Guided Response Generation for Task-Oriented Dialogue" by Anonymous et al.


## Installation

```bash
unzip workflow-response.zip
cd workflow-response
pyenv virtualenv workflow-response
pyenv activate workflow-response
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Install Quark inside the main repository and move the contents of the RL_code folder to the Quark folder:

```git clone git@github.com:GXimingLu/Quark.git
mv Quark/RL_code/* Quark/
```


## Data Processing

Download and create datasets for training.

```sh
bash scripts/dataproc/download_process_abcd.sh

```

## For training and evaluation, make sure that the trained model names are updated in the bash files / corresponding python scripts.

## Training

### Model Choice Options for Training
1. NoAction
2. ActionAware
3. Workflow Prediction
4. ActionPlan Oracle 
5. ActionPlan Cascade (only used for evaluation) 
6. ActionPlan All Future Oracle
7. Guideline Oracle


### Finetune base model

```sh
bash bash/distill_train.sh [1-7]
```
Options 1 - 7 determine the model to be trained.

### Train the reward model
Refer to ./RL_code/README.md for information on the adapted code files.
```sh
bash bash/block_workflow_scorer_run.sh 1
```

### Train the RL model 
```sh
bash bash/quark_run.sh
```

## Evaluation 
Evaluate the teacher-forcing based models
```sh
bash distill_evaluate.sh [1-7]
```
Options 1 - 7 determine the model to be evaluated.

Evaluate the RL model:
```sh
python eval/interactive_quark_eval.py
```

Human evaluations (given the human compliance annotation csv files in ./human_annotations/):
```sh
python eval/process_human_eval.py
```