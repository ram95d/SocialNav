# Model training

## Trajectory prediction model

Train the model on the whole dataset.
Different models can be trained by changing the configuration file.
```shell
# Train new model.
python model-run.py fit -- --config model/configs/model.yaml

# Continue training from a checkpoint.
python model-run.py fit -- --config model/configs/model.yaml \
                           --ckpt_path logs/model/checkpoints/epoch-100.ckpt

# Fine-tune the model.
python model-run.py fit --ft_ckpt_path logs/model_name/checkpoints/epoch-100.ckpt \
                        -- \
                        --config model/configs/model-ft.yaml
```

Evaluate the model with the leave-one-out strategy.
Different models can be evaluated by changing the configuration file.
```shell
python model-run.py eval --eval_dataset model/data/one-out -- --config model/configs/model.yaml
```

Evaluation results will be saved in `logs/${model_name}-eval/results.yaml`.

## Train with synthetic dataset

Pretrain the model on the synthetic dataset.
```shell
python model-run.py fit -- --config model/configs/model-synth.yaml
```

Evaluate the model with the leave-one-out strategy, by fine-tuning
on the train split of the evaluation dataset.
```shell
python model-run.py eval --eval_dataset model/data/one-out \
                         --ft_ckpt_path logs/model_name/checkpoints/epoch-100.ckpt \
                         -- \
                         --config model/configs/model-ft.yaml
```

## Mask autoencoder

Train the mask autoencoder.
```shell
python -m model.mask_autoenc.train --config model/configs/autoenc/autoenc.yaml
```

## Monitor training

Monitor training with tensorboard.
```shell
tensorboard --logdir logs
```

## Datasets

The first time the model is run, the datasets will be constructed
from scratch, so it will take some time. The next time, the datasets
will be loaded from the cache.

## Configs

The `.yaml` config files are used to configure the model training.

Some things to take into consideration:
- `data.min_obs_len`: minimum number of observations in a trajectory
  when training the model. For fair comparison, `8` must be used.
  When training the model for deployment, you can use the value that
  will be used in the deployment, e.g., `2` for predicting the
  future trajectory as soon as possible. `model.obs_len` is related
  to this value, but it is used to define the maximum number of
  observations to use when predicting the future trajectory.
- `data.min_pred_len`: minimum number of available future positions
  when training the model. For fair comparison, `12` must be used.
  Similarly, `model.pred_len` must be `12` for evaluation.
- `data.fill_missing`: if `true`, missing observations in the
  trajectories in the dataset will be filled with a linear
  interpolation. If `false`, the trajectories with missing
  observations will be removed from the dataset. For fair
  evaluation, `false` must be used. For deployment, `true` is a
  reasonable choice, to exploit more data.
- `data.only_visible`: a visible pedestrian is defined as a
  pedestrian for which the last observation is known. If `true`,
  only the trajectories of visible pedestrians will be used in the
  dataset. For fair evaluation, `true` must be used. For deployment,
  `false` is a reasonable choice, to potentially exploit more data
  (in the case where the last observation is not known, but the
  future trajectory is known).
- `data.path`: path to the dataset. For evaluation using
  `python model-run.py eval`, the path is automatically overwritten
  for the specific evaluation dataset, according to the
  `--eval_dataset` argument (see above).
- `data.max_step_size`: maximum step size between two consecutive
  observations in a trajectory. If `null`, the step size is not
  limited. Note that it is a dataset dependent parameter, since it depends
  on the scale of the trajectories.
  For fair evaluation, the filtering is never applied to the
  test set, even if it is specified in the config file.
- `model.num_samples`: number of samples to generate when predicting.
  For evaluation, `1` must be used in the deterministic setting,
  and `20` in the stochastic setting.

### Evaluation configs

The configs used for evaluation must contain the following fields:
```yaml
model:
  init_args:
    obs_len: 8
    pred_len: 12
    num_samples: 20 # or 1
data:
  init_args:
    min_obs_len: 8
    min_pred_len: 12
    fill_missing: false
    only_visible: true
```
