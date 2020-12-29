# IQA with RankNet
The repo includes a simple framework to train a RankNet model for IQA tasks. It is the final project of CS386 (Digital Image Processing), SJTU.

## Dataset
In this project, we use dataset [ICME2020](https://qa4camera.github.io/). Please refer to the [sample dataset](./sample_dataset) folder for data organization.

In experiments, we split the dataset into two parts: 1-80 scenes are used for training and 81-100 scenes are used for validation.

## Train
### Using config files
Modify the configurations in .json config files, then run:
```
python train.py --config config.json
```

### Resuming from checkpoints
You can resume from a previously saved checkpoint by:
```
python train.py --resume path/to/checkpoint
```

### Using Multiple GPU
You can enable multi-GPU training by setting n_gpu argument of the config file to larger number. If configured to use smaller number of gpu than available, first n devices will be used by default. Specify indices of available GPUs by cuda environmental variable.

```
python train.py --device 2,3 -c config.json
```
This is equivalent to
```
CUDA_VISIBLE_DEVICES=2,3 python train.py -c config.py
```

## Test
You can test trained model by running [test.py](./test.py) passing path to the trained checkpoint by `--resume` argument.

## Reference
- Burges, Chris, et al. "Learning to rank using gradient descent." Proceedings of the 22nd international conference on Machine learning. 2005.
- Burges, Christopher JC. "From ranknet to lambdarank to lambdamart: An overview." Learning 11.23-581 (2010): 81.
- Ma, Kede, et al. "dipIQ: Blind image quality assessment by learning-to-rank discriminable image pairs." IEEE Transactions on Image Processing 26.8 (2017): 3951-3964.
- Su, Shaolin, et al. "Blindly Assess Image Quality in the Wild Guided by a Self-Adaptive Hyper Network." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.