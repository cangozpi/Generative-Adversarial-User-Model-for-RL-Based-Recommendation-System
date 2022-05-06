# Generative Adversarial User Model for RL Based Recommendation System
Implementation of the paper ["Generative Adversarial User Model  for RL Based recommendation System"](https://arxiv.org/abs/1812.10613) in Pytorch as part of a group project in Koc University COMP547 _Unsupervised Deep Learning Course_.

---

## __TODOs__:
* Create Pytorch <u>_Dataloaders_ for the datasets</u>
* Implement <u>_GAN model_ using pytorch</u>
* Create <u>_Training & Validation & Evaluation Loops_</u>
* Implement _main.py_ to instantiate training of the model according to the passed in command line arguments and the _config.yaml_.
* First <u>_Training_</u> of the model
* <u>_Replicate Baseline_</u> Results in the paper
* Perform <u>_Hyperparameter search_</u>
* __(Optional)__ RL agent implementation (if time permits)

---

## __To Install:__
* Using conda Environments:
    ```bash
    $ conda create --name <env> --file requirements.txt
    ```
* Using pip to install:
    ```bash
    $ python -m pip install -r requirements.txt
    ```
---


## __To Run:__
1. Running steps are yet to come
    ```
    $ 
    ```

---

## __File Structure:__
* __main.py__: 
    
    Parses command line arguments and reads in the model hyperparameters from _config.yaml_ to initiate training/testing accordingly.

* __train.py__:

    Implements training loop for the model.

* __test.py__:

    Implements testing loop for the model.

* __data.py__:

    Implements dataloaders of the datasets.

* __model/__ -->
    * __generator.py__:

        Implements Generator model that is part of the GAN model.

    * __discriminator.py__: 

        Implements Discriminator model that is part of the GAN model.

    * __gan.py__:         

        Implements the GAN model using the Generator & the Discriminator models that are implemented in _generator.py_ and _discriminator.py_.

* __config.yaml__: 

    Entails Hyperparameters of the model.

---
