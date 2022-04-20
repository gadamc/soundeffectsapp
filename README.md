# SoundFX

An application for sound-effect engineers and independent film makers to match unlabeled images from videos to unlabeled sound files. 

Technically, this application defines a simple neural network that learns to pair still images to unlabeled sound files via a latent representation of the image and sound file generated by neural networks. 
A pre-trained CNN (VGGnet) was used to feed-forward video frames to final layer of the network (before classification layer). Similarly, the sound from the video
were converted into a spectrogram and fed-forward through a different pre-trained neutral network (VGGish - trained specifically for spectrogram classification) to the final pre-classification layer. The 2-layer perceptron network defined was then trained to reproduce the final layer outputs from the sound NN using the final layer outputs from the image NN as input.

An out-of-sample test was performed where images were passed through the VGGnet and the trained simple neural network defined here. The output was then matched (cosine similarity) to a set of sound files, represented by the final layer output from VGGish network applied to each sound file. Though the sample size was small, the test exhibited skill better than random selection. 

This project was produced in ~2 weeks time as part of the Insight Data Science program in Seattle in the summer of 2018.

## Installation
These instructions work for Ubuntu 16. This application was developed on Mac OSX High Sierra.

```
1. conda install numpy scipy pandas scikit-learn flask sqlalchemy
2. conda install pytorch-cpu torchvision-cpu -c pytorch (OSX: conda install pytorch torchvision -c pytorch)
3. conda install -c conda-forge tensorflow
4. conda install pillow
```

## Run

```
./run_app.py
```



