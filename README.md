# Conditional Poisson Flow Generative Models

Pytorch implementation of the CoPFGM 2023 paper [include paper link]

by [Ioannis Georgiades]()

---

We propose a new **Conditional** Poisson flow generative model (**CoPFGM**) that extends the PFGM [repository](https://github.com/Newbeeer/Poisson_flow) and allows for conditional image sampling, provided proper label
dataset and configured model. Currently the repository supports the following datasets:

* **MNIST** : Common dataset of handwritten digits provided by Tensorflow Datasets
* **Dilbert - Faces** : Custom dataset of Dilbert comic strips faces


Experimentally, CoPFGM achieves the conditional behavior by introducing the
label information of each image during training and sampling by augmenting 
.

---



*Acknowledgement:* Our implementation relies on the repo [Poisson Flow Generative Models](https://github.com/Newbeeer/Poisson_flow)


## Dependencies

The code is tested on Ubuntu 20.04,
Python 3.10.12, CUDA Version 11.8

```sh
pip install -r requirements.txt
```



## Usage

Train and evaluate our models through `main.py`.

```sh
python3 main.py:
  --config: Training configuration.
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval
  --workdir: Working directory
```

For example, to train a new CoPFGM model on MNIST dataset, one could execute 

```sh
python3 main.py --config ./configs/poisson/mnist_config.py --mode train \
--workdir CoPFGM_MNIST
```

And similarly, to evaluate the model, one could execute

```sh
python3 main.py --config ./configs/poisson/faces_config.py --mode train \
--workdir CoPFGM_Dilbert
```
  **Note**: During training and sampling the default conditioned class is
  the first class of the dataset for example the number 0 for MNIST. To change the conditioned class, please
  modify the `config.sampling.target` parameter in the config files or
  modify the command by adding `--config.sampling.target [ClassID]`.

 `config` is the path to the config file. The prescribed config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

For a custom dataset, you can run the following script, which will
give you some PFGM hyper-parameter suggestions, based on the average data norm of the dataset and the data dimension: 

```shell
python3 hyper-parameters.py 
  --data_norm: Average data norm of the dataset 
  --data_dim: Data dimension
```


*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` .


## Tips

- **Important** : We use a large batch (e.g. current `training.batch_size=4096` for MNIST) to calculate the Poisson field for each mini-batch samples (e.g. `training.small_batch_size=128` for MNIST). To adjust GPU memory cost, please modify the `training.batch_size` parameter in the config files. The same applies for the dilbert dataset.
  

## Checkpoints

Please place the pretrained checkpoints under the directory `workdir/checkpoints`, e.g., `CoPFGM_MNIST/checkpoints`.  

To generate and evaluate the FID/IS of  (10k) samples of the CoPFGM you could execute:

```shell
python3 main.py --config ./configs/poisson/mnist_config.py --mode eval \ 
--workdir CoPFGM_MNIST --config.eval.enable_sampling --config.eval.num_samples 10000
```

To only generate and visualize 100 samples you could execute:

```shell
python3 main.py --config ./configs/poisson/mnist_config.py --mode eval \ 
--workdir CoPFGM_MNIST --config.eval.enable_sampling --config.eval.save_images --config.eval.batch_size 100
```

The samples will be saved to `CoPFGM_MNIST/eval/ode_images_{ckpt}.png`.

Important, depending on the config and chekpoint counter, you might need 
to adjust also in the config the 

```shell
... --config.eval.begin_ckpt x --config.eval.end_ckpt y
```

All checkpoints are provided in this [Google drive folder](https://drive.google.com/drive/folders/1v4u0OhZ0rxjgch51pZLySztMQATQQOeK?usp=sharing).

| Dataset              | Checkpoint path                                              |    Invertible?     |  IS  |  FID  | NFE (RK45) |
| -------------------- | :----------------------------------------------------------- | :----------------: | :--: | :---: | :--------: |
| CIFAR-10             | [`poisson/cifar10_ddpmpp/`](https://drive.google.com/drive/folders/1UBRMPrABFoho4_laa4VZW733RJ0H_TI0?usp=sharing) | :heavy_check_mark: | 9.65 | 2.48  |    ~104    |
| CIFAR-10             | [`poisson/cifar10_ddpmpp_deep/`](https://drive.google.com/file/d/1BeJGD0WP230u8nkHEWqywOvhH2_5F-Q0/view?usp=sharing) | :heavy_check_mark: | 9.68 | 2.35  |    ~110    |
| LSUN bedroom $256^2$ | [`poisson/bedroom_ddpmpp/`](https://drive.google.com/drive/folders/1uFmlcBTQmUI_ZfyUiYoR54H4V2uBsuS7?usp=sharing) | :heavy_check_mark: |  -   | 13.66 |    ~122    |
| CelebA $64^2$        | [`poisson/celeba_ddpmpp/`](https://drive.google.com/drive/folders/1LjplqjwIfZbp6LeK3_M2rIW-CaVhgn6p?usp=sharing) | :heavy_check_mark: |  -   | 3.68  |    ~110    |



### FID statistics

Please find the statistics for FID scores in the following links:

[CIFAR-10](https://drive.google.com/file/d/1YyympxZ95l6_ane0TxYt94yqeiGcOBNG/view?usp=sharing),  [CelebA 64](https://drive.google.com/file/d/1dzSsmBvJOjDy12VzdypWDVYBF8b9yRkm/view?usp=sharing), [LSUN bedroom 256](https://drive.google.com/file/d/16zTW5DhwmK4Hl-Vhez9LDyqN-CXi4Lhi/view?usp=sharing)





<center><img src="assets/pfgm_cat.gif" width="750" height="250"/></center>







