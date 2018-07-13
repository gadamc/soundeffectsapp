# SoundFX

An application to match unlabeled images from videos to unlabeled sound files. A prototype for an automatic sound-effects insertion tool for sound effect engineers and independent film makers. 

[http://audioinsert.xyz](http://audioinsert.xyz)

## Installation and Setup

Following [AWS setup instructions](https://docs.google.com/presentation/d/1EjBfDnIauu9L5LIH_79XqIWkWfCeB1AA99Q7rD75W_I/edit#slide=id.p) with an Ubuntu 16 server, the following commands install the necessary packages. This runs on an t2.medium instance with a 16 GB EBS volume.

```
1. conda install numpy scipy pandas scikit-learn flask sqlalchemy

2. conda install pytorch-cpu torchvision-cpu -c pytorch

3. conda install -c conda-forge tensorflow

4. conda install pillow
5. conda install sqlalchemy-utils
```

Run the server

```
./run_app.py
```



