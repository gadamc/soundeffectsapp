import os
from flask import Flask
app = Flask(__name__)
app.config['CWD'] = os.getcwd()
from soundeffectsapp import views

