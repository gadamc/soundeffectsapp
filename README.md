# Audioeinfuengung

An application to demonstrate the matching of images and videos with a an appropriate sound from a library of unlabled sounds. A prototype for an automatic sound-effects insertion tool for sound effect engineers and independent film makers. 

## Installation and Setup

Following [AWS setup instructions](https://docs.google.com/presentation/d/1EjBfDnIauu9L5LIH_79XqIWkWfCeB1AA99Q7rD75W_I/edit#slide=id.p) with an Ubuntu 16 server, the following commands install the necessary packages. 

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

