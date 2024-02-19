from flask import Flask, request, jsonify, redirect, Response, url_for, send_from_directory, render_template, flash
from flask_mysqldb import MySQL 
from flask_jwt_extended import JWTManager
from flask_cors import CORS
from datetime import datetime
import numpy as np
import decimal
from datetime import datetime
import os
from object_detection_classification.object_detection_classification import get_class_list, detect_and_classify
from text_detection.detect_text import get_text_detection
from recipe_recommendation.tf_idf.recommend_recipes import get_recipes
from chatbot.chatbot_response import get_bot_response

app = Flask(__name__)
CORS(app)

class_labels = []
recipe_list = []

@app.route("/")
def landing():
    return ("landing")

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        start_time = datetime.now()

        print("Received request", request.files)
        images = []
        for key in request.files:
            image_files = request.files.getlist(key)
            images.extend(image_files)
        print("Images received:", images)
    
        classifications = []
        global recipe_list

        for image in images:
            print("Performing object detection on image:", image)

            # Detect and classify the objects in the image
            object_detection_results = detect_and_classify(image)
            print("Object detection results:", object_detection_results)
            classifications.append(object_detection_results)
            
            # Perform text detection on the image
            print("Performing text detection on image:", image)
            text_detection_results = get_text_detection(image)
            print("Text detection results:", text_detection_results)
        
        # Extract the class labels from object detection results
        class_labels = [result['class_label'] for result_list in classifications for result in result_list]
        print("Class labels:", class_labels)
        
        # Combine results from object detection and text detection
        combined_results = class_labels + text_detection_results

        print("Combined results:", combined_results)

        # Calculate the total time taken
        end_time = datetime.now()
        total_time = end_time - start_time
        print("Total time taken:", total_time)

        print("Getting recipes")
        # Use combined results as query for recipes
        recipe_list = get_recipes(combined_results)
        print("Recipe list:", recipe_list)

        # Return a JSON response with recipes
        return jsonify({'recipes': recipe_list}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/ingredients", methods=['GET'])
def get_available_ingredients():
    class_list = get_class_list()
    return jsonify({'ingredients': class_list})

@app.route("/chatbotresponse", methods=['POST'])
def get_response():
    userText = request.json.get('msg')
    print("User Text: ", userText)
    return jsonify({'message': get_bot_response(userText)}) 

@app.route('/recipe', methods=['GET'])
def get_recipe():    
    return jsonify({'recipes': recipe_list})   

if __name__=="__main__":
    app.run(debug=True)
    