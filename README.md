# Repository for Workflow-guided Response Generation


## Installation

```bash
git clone git@github.com:asappresearch/workflow-response.git
cd workflow-response
pyenv virtualenv workflow-response
pyenv activate workflow-response
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data Processing

Download and create datasets for training.

```sh
for dataset in abcd multi_woz; do
    bash scripts/dataproc/download_process_${dataset}.sh
done
```

## Training

### Finetune base model

```sh
bash bash/gpt2_train.sh
```

### Train reward model

```sh
bash bash/bert_reward/train.sh
```

### Train RL model

```sh
bash bash/quark_run.sh
```

## Evaluation

Evaluate RL model:
```sh
python eval/interactive_quark_eval.py
```

Human evaluations:
```sh
python eval/process_human_eval.py
```
