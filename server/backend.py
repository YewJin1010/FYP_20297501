from flask import Flask, request, jsonify, redirect, Response, url_for, send_from_directory, render_template, flash
from flask_cors import CORS
from datetime import datetime
import numpy as np
import decimal, os, re
from datetime import datetime
from object_detection_classification.object_detection_classification import get_class_list, detect_and_classify
from text_detection.detect_text import get_text_detection
from recipe_recommendation.tf_idf.recommend_recipes import query_recipes
from chatbot.chatbot_response import get_bot_response
from database.database_ingredients import extract_ingredients_from_text
from spellchecker import SpellChecker 

app = Flask(__name__)
CORS(app)

recipe_list = []

@app.route("/")
def landing():
    return ("landing")

@app.route('/upload_object', methods=['POST'])
def upload_object_detection():
    print("Object detection request received")
    try:
        start_time = datetime.now()

        print("Received request", request.files)
        images = []
        for key in request.files:
            image_files = request.files.getlist(key)
            images.extend(image_files)
        print("Images received:", images)
    
        classifications = []

        for image in images:
            print("Performing object detection on image:", image)

            # Detect and classify the objects in the image
            object_detection_results = detect_and_classify(image)
            if isinstance(object_detection_results, str):
                print("Error:", object_detection_results)
                return jsonify({'error': object_detection_results}), 400  # Return error response
            
            print("Object detection results:", object_detection_results)
            classifications.append(object_detection_results)
            
        # Extract the class labels from object detection results
        class_labels = [result['class_label'] for result_list in classifications for result in result_list]
        print("Class labels:", class_labels)
        
        # Calculate the total time taken
        end_time = datetime.now()
        total_time = end_time - start_time
        print("Total time taken:", total_time)

        # Return a JSON response with recipes
        return jsonify({'class_labels': class_labels}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/upload_text', methods=['POST'])
def upload_text_detection():
    print("Text detection request received")
    try:
        start_time = datetime.now()

        print("Received request", request.files)
        images = []
        for key in request.files:
            image_files = request.files.getlist(key)
            images.extend(image_files)
        print("Images received:", images)
        results = []
    
        for image in images:
            print("Performing text detection on image:", image)
            result = get_text_detection(image)
            if isinstance(result, str):
                print("Error:", result)
                return jsonify({'error': result}), 400  # Return error response
            print("Results:", result)

            results.append(result)
        text_detection_results = [item for sublist in results for item in sublist]
        print("Text detection results:", text_detection_results)

        # Calculate the total time taken
        end_time = datetime.now()
        total_time = end_time - start_time
        print("Total time taken:", total_time)

        # Return a JSON response with recipes
        return jsonify({'text_detection_results': text_detection_results}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_recipes', methods=['POST'])
def get_recipes():
    global recipe_list
    try: 
        combined_results = request.json
        print("Combined results:", combined_results)

        # Use combined results as query for recipes
        recipe_list = query_recipes(combined_results)
        print("Recipe list:", recipe_list)

        return jsonify({'recipes': recipe_list}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/ingredients", methods=['GET'])
def get_available_ingredients():
    class_list = get_class_list()
    return jsonify({'ingredients': class_list})

def autocorrect_text(text):
    spell = SpellChecker()
    misspelled = text.split()
    corrected_text = []
    for word in misspelled:
        corrected_text.append(spell.correction(word))

    print("Original text: ", text)
    print("Corrected text: ", " ".join(corrected_text))
    return " ".join(corrected_text)

@app.route("/chatbotresponse", methods=['POST'])
def get_response():
    userText = request.json.get('msg')

    # Autocorrect user input
    userText = autocorrect_text(userText)

    response, intents = get_bot_response(userText)
    print("Response: ", response)
    print("Intents: ", intents)

    if intents[0]['intent'] == 'RequestIngredientRecipe':
        found_ingredients = []
        global recipe_list

        found_ingredients = extract_ingredients_from_text(userText)
        print("Found ingredients: ", found_ingredients)
        if found_ingredients:
            print("Found ingredients: ", found_ingredients)
            recipe_list = query_recipes(found_ingredients)
            return jsonify({'recipes': recipe_list})
        else: 
            response = "Sorry, I didn't catch that. Could you please specify the ingredient you would like to use?"
            return jsonify({'recipes': response})

    return jsonify({'message': response}) 

@app.route('/recipe', methods=['GET'])
def get_recipe_response():    
    return jsonify({'recipes': recipe_list})   

if __name__=="__main__":
    app.run(debug=True)
    