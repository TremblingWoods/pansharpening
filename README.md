# pan-sharpening    Team-zhouman

|model|paper|
|-|-|
|panformer-AAAI| Pan-sharpening with Customized Transformer and Invertible Neural Network|
|GCPNet-TGRS|When Pansharpening Meets Graph Convolution Network and Knowledge Distillation|
|band_aware-TCI|PAN-guided band-aware multi-spectral feature enhancement for Pan-sharpening|
|distill-TGRS|Effective Pan-sharpening by Multi-Scale Invertible Neural Network and Heterogeneous Task Distilling|
|pan_unfolding-CVPR|Memory-augmented Deep Conditional Unfolding Network for Pan-sharpening|
|MutInf-CVPR|Mutual Information-driven Pan-sharpening|
|SFITNET-ECCV|Spatial-Frequency Domain Information Integration for Pan-sharpening|
|SFINet++-TPAMI|A General Spatial-Frequency Learning Framework for Satellite Image Pansharpening|
|mmnet-IJCV|Memory-augmented Deep Unfolding Network for Guided Image Super-resolution|
|LHFnet-ACM MM|Adaptively Learning Low-high Frequency Information Integration for Pan-sharpening|
|NormNet-ACM MM|Normalization-based Feature Selection and Restitution for Pan-sharpening|
|CINet-TGRS|Information Compensation and Integration Network for Pan-sharpening|



If you have any questions, please contact us (manman@mail.ustc.edu.cn)


This repository contains the implementation of the various algorithm for super-resolution of remote sensing images. The algorithm is trained using a deep neural network architecture and is implemented using PyTorch.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

In order to run this implementation, you need to have the following software and libraries installed:

- Python 3.7 or later
- PyTorch 1.6 or later
- CUDA (if using GPU)
- NumPy
- Matplotlib
- OpenCV
- PyYAML

### Installing

You can install the necessary packages using pip:

```python
pip install torch numpy matplotlib opencv-python pyyaml
```

### Prepare Dataset

All the training/testing data are avalibal at [Baidu Disk](https://pan.baidu.com/s/12XqgR3l4m-WBxGOw4W807A?pwd=i1dk) code: i1dk 

If you use this datasets, please cite our paper to acknowledge our efforts.


### Configuration

Before training the model, you need to configure the following options in the `option.yaml` file:


- `log_dir`: the directory to store the training log files.
- `checkpoint`: the directory to store the trained model parameters.
- `data_dir_train`: the directory of the training data.
- `data_dir_eval`: the directory of the evaluation data.

### Training the Model

To train the model, you can run the following command:

```
python main.py
```

### Testing the Model

To test the trained pan-sharpening model, you can run the following command:

```
python test.py
python py-tra/demo_deep_methods.py
```

### Configuration

The configuration options are stored in the `option.yaml` file. Here is an explanation of each of the options:

#### algorithm

- algorithm: The model for training

#### Logging

- `log_dir`: The location where the log files will be stored.

#### Model Weights

- `checkpoint`: The location to store the model weights.

#### Training Data

- `data_dir_train`: The location of the training data.
- `data_dir_eval`: The location of the test data.

#### Pretrain

- `pretrained`: Whether to use a pretrained model.
- `pre_sr`: The location of the pretrained model.
- `pre_folder`: The location where the pretrained models are stored.

#### Testing

- `algorithm`: The algorithm to use for testing.
- `type`: The type of testing, either `test` or `eval`.
- `data_dir`: The location of the test data.
- `source_ms`: The source of the multi-spectral data.
- `source_pan`: The source of the panchromatic data.
- `model`: The location of the model to use for testing.
- `save_dir`: The location to save the test results.

#### Data Processing

- `upscale`: The upscale factor.
- `batch_size`: The size of each batch.
- `patch_size`: The size of each patch.
- `data_augmentation`: Whether to use data augmentation.
- `n_colors`: The number of color channels.
- `rgb_range`: The range of the RGB values.
- `normalize`: Whether to normalize the data.

#### Training Hyperparameters

- `schedule.lr`: The learning rate.
- `schedule.decay`: The learning rate decay.
- `schedule.gamma`: The learning rate decay factor.
- `schedule.optimizer`: The optimizer to use, either `ADAM`, `SGD`, or `RMSprop`.
- `schedule.momentum`: The momentum for the `SGD` optimizer.
