import os
import glob
import json

import numpy  as np
from PIL import Image

import torch
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from soundeffectsapp import app
from soundeffectsapp.video_features.feature_extractor import YouTube8MFeatureExtractor

import tensorflow as tf

print('Setting up Model')
SOUNDFILE_FEATURE_DIR = 'soundeffectsapp/static/audio_samples/features'
SOUNDFILE_WAV_DIR = '../static/audio_samples/wav'

SOUNDFILE_LIST = glob.glob(os.path.join(SOUNDFILE_FEATURE_DIR,'*.json'))

print('Found {} sound feature files in {}'.format(len(SOUNDFILE_LIST), SOUNDFILE_FEATURE_DIR))
#print(SOUNDFILE_LIST)

VIDEO_MODEL_DIR =  os.path.join(app.config['CWD'], 'soundeffectsapp/video_features/')
TORCHMODELPARAMS = {"simple":os.path.join(app.config['CWD'], 'soundeffectsapp/models/simple_model_jun21.pt')}
TORCHMODELPARAMS['mlp_2layer_selu'] = os.path.join(app.config['CWD'], 'soundeffectsapp/models/MLP_2layer_SELU_jun14.pt')

TORCHMODEL_SIMPLE = torch.nn.Sequential(
    torch.nn.Linear(1024, 128),
    torch.nn.Tanh()
)

TORCHMODEL_MLP_2LAYER_SELU = torch.nn.Sequential(
    torch.nn.Linear(1024, 512),
    torch.nn.SELU(),
    torch.nn.Linear(512, 256),
    torch.nn.SELU(),
    torch.nn.Linear(256, 128),
    torch.nn.Tanh()
)

TORCHMODEL_SIMPLE.load_state_dict(torch.load(TORCHMODELPARAMS['simple']))
#TORCHMODEL_MLP_2LAYER_SELU.load_state_dict(torch.load(TORCHMODELPARAMS['mlp_2layer_selu']))

TORCHMODEL = TORCHMODEL_SIMPLE
TORCHMODEL.eval()

_SOUNDFILE_FEATURES = {}
for sf in SOUNDFILE_LIST:
    sfdict = json.load(open(sf,'r'))
    _SOUNDFILE_FEATURES[sf] = sfdict

#print(_SOUNDFILE_FEATURES)

# def _scale(v, mean=128., variance=128.0):
#     return (v - mean)/variance
# def _unscale(v, mean=128., variance=128.):
#     return variance*v + mean

# def _int64_list_feature(int64_list):
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


# def _bytes_feature(value):
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def _make_bytes(int_array):
#   if bytes == str:  # Python2
#     return ''.join(map(chr, int_array))
#   else:
#     return bytes(int_array)


def _clip_n_scale(features, min_value=-2.0, max_value=2.0):
  assert features.dtype == 'float32'
  features = np.clip(features, min_value, max_value)
  _range = (max_value - min_value)/2.
  return features / _range

# def _quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0, return_as_bytes=False):
#   """Quantizes float32 `features` into string."""
#   assert features.dtype == 'float32'
#   assert len(features.shape) == 1  # 1-D array
#   features = np.clip(features, min_quantized_value, max_quantized_value)
#   quantize_range = max_quantized_value - min_quantized_value
#   features = (features - min_quantized_value) * (255.0 / quantize_range)
#   features = [int(round(f)) for f in features]

#   if return_as_bytes:
#     return _make_bytes(features)
#   else:
#     return features

def _getFeaturesFromImage(inputfile):
    extractor = YouTube8MFeatureExtractor(VIDEO_MODEL_DIR)
    im = np.array(Image.open(inputfile))
    #features = _scale(np.array(_quantize(extractor.extract_rgb_frame_features(im)), dtype = np.float32))
    features = extractor.extract_rgb_frame_features(im)
    features = _clip_n_scale(features)

    #features = _scale(np.clip(extractor.extract_rgb_frame_features(im), -2.0, 2.0), 0, 4.0)
    return features[np.newaxis, :]

def _calcCosSim(audio_features, audio_key = 'mean_audio'):
    '''
    audio_key = 'mean_audio' or 'median_audio'
    '''
    sf_similarities = []
    for k,v in _SOUNDFILE_FEATURES.items():
        mean_audio = np.array(v[audio_key]).reshape(1,-1)
        cossim = cosine_similarity(audio_features, mean_audio)
        sf_similarities.append((v['filename'], cossim[0][0], k))
    return sf_similarities


def _replacepath(sftuple):
    fname, costheta, fpath = sftuple
    linkpath = os.path.join(SOUNDFILE_WAV_DIR, os.path.splitext(os.path.basename(fpath))[0])
    return(fname, costheta, linkpath)

def runModel(filename):
    
    inputpath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    print(inputpath)
    _, fileext = os.path.splitext(inputpath)
    fileext = fileext.strip('.')

    if fileext not in app.config['ALLOWED_EXTENSIONS']:
        print(app.config['ALLOWED_EXTENSIONS'])
        return [['invalid file type', -1]]

    #first step is to extract features
    if fileext in ('jpg', 'jpeg'):
        features = _getFeaturesFromImage(inputpath)
    else:
        return [['not yet implemented', -1]]
        #features = getAverageFeaturesFromVideo(inputpath)

    #now pass the video features to the torch model and extract
    #the audio feature predictions

    #print(features.tolist())
    #hh = np.histogram(features,bins=255)
    #print(hh[0])

    audio_features = TORCHMODEL(torch.from_numpy(features))
    #print(audio_features)

    sfsims = _calcCosSim(audio_features.detach().numpy())
    sfsims.sort(key = lambda x:x[1], reverse=True)
    sfsims = list(map(_replacepath, sfsims))
    
    return sfsims



