from flask import Flask, request, jsonify, redirect, Response, url_for, send_from_directory, render_template, flash
from flask_mysqldb import MySQL 
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required
from flask_cors import CORS
from datetime import datetime
import numpy as np
import decimal
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './file_uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['css'] = "./templates/css/layouts"
app.config['img'] = "./templates/img"
app.config['CROPPED'] = "./cropped_photos"

@app.route("/")
def landing():
    return ("landing")

@app.route('/img/<file_name>')
def home_page(file_name):
    return send_from_directory(app.config['img'], file_name)

def allowed_file(filename):
    #split the filename so you return the first item in the list to check if its allowed
    return '.' in filename and \
       filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'users_img' not in request.files:
            return redirect('/')
        
        uploaded_img = request.files['users_img']
        if uploaded_img.filename == '':
            return redirect('/')
        
        if uploaded_img and allowed_file(uploaded_img.filename):
            uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_img.filename))
            return redirect('/')
        
        