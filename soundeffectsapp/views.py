import os

from flask import render_template
from flask import request, redirect, url_for, flash
from soundeffectsapp import app

from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2

from werkzeug import secure_filename

import soundeffectsapp.a_Model as videoToAudioModel


user = 'adam' #add your username here (same as previous postgreSQL)                      
host = 'localhost'
dbname = 'birth_db'
db = create_engine('postgres://%s%s/%s'%(user,host,dbname))
con = None
con = psycopg2.connect(database = dbname, user = user)

UPLOAD_FOLDER = os.path.join(app.config['CWD'], 'input_videos')
ALLOWED_EXTENSIONS = set(['mp4', 'jpg', 'jpeg'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template("input.html")


@app.route('/input')
def cesareans_input():
    return render_template("input.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
      uploadFile = request.files['file']
      if uploadFile.filename == '':
        flash('No selected file')
        return redirect(request.url)

      if uploadFile and allowed_file(uploadFile.filename):
        #uploadFile.save(secure_filename(uploadFile.filename))
        filename = secure_filename(uploadFile.filename)
        uploadFile.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('uploaded_file',
                                    filename=filename))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    #return 'uploaded {}'.format(filename)

    ## could use this to return sound file? 
    # no, probably just want to populate a list with links to the soundeffect files
    the_result = videoToAudioModel.runModel(filename)
    print(the_result)
    #return str(the_result)
    #the_result_urls = map(lambda x: url_for('static', filename=x[0])) 


    return render_template("output.html", resultsFromModel = the_result)
    
    #return send_from_directory(app.config['UPLOAD_FOLDER'],
    #                           filename)

# @app.route('/output')
# def cesareans_output():
#   #pull 'birth_month' from input field and store it
#   patient = request.args.get('birth_month')
#     #just select the Cesareans  from the birth dtabase for the month that the user inputs
#   query = "SELECT index, attendant, birth_month FROM birth_data_table WHERE delivery_method='Cesarean' AND birth_month='%s'" % patient
#   print(query)
#   query_results=pd.read_sql_query(query,con)
#   print(query_results)
#   births = []
#   for i in range(0,query_results.shape[0]):
#       births.append(dict(index=query_results.iloc[i]['index'], attendant=query_results.iloc[i]['attendant'], birth_month=query_results.iloc[i]['birth_month']))
#       the_result = ''
#   #return render_template("output.html", births = births, the_result = the_result)
#   the_result = ModelIt(patient,births)
#   return render_template("output.html", births = births, the_result = the_result)



