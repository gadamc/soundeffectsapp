# SoundFX

An application for sound-effect engineers and independent film makers to match unlabeled images from videos to unlabeled sound files. 

[http://audioinsert.xyz](http://audioinsert.xyz)

## Installation and Setup

This runs on an AWS t2.medium instance with a 16 GB EBS volume running Ubuntu 16.

```
1. conda install numpy scipy pandas scikit-learn flask sqlalchemy
2. conda install pytorch-cpu torchvision-cpu -c pytorch
3. conda install -c conda-forge tensorflow
4. conda install pillow
```

Run the server

```
./run_app.py
```



