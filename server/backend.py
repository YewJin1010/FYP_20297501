from flask import Flask, request, jsonify, redirect, Response, url_for, send_from_directory, render_template, flash
from flask_cors import CORS
from datetime import datetime
import numpy as np
import decimal, os, re
from datetime import datetime
from object_detection_classification.object_detection_classification import get_class_list, detect_and_classify
from text_detection.detect_text import get_text_detection
from recipe_recommendation.t5.generate_recipes import generate_recipe
from chatbot.preprocess_text import autocorrect_text
from chatbot.chatbot_response import get_bot_response


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

def format_ingredients(combined_results):
    # Remove repeating ingredients from the list
    combined_results = list(set(combined_results))
    ingredients = ', '.join(combined_results)
    return ingredients

@app.route('/get_recipes', methods=['POST'])
def get_recipes():
    global recipe_list
    try: 
        combined_results = request.json
        print("Combined results:", combined_results)

        ingredients = format_ingredients(combined_results)
        print("Ingredients:", ingredients)

        # Use combined results as query for recipes
        recipe_list = generate_recipe(ingredients)
        print("Recipe list:", recipe_list)

        return jsonify({'recipes': recipe_list}), 200
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': 'Internal server error'}), 500

@app.route("/ingredients", methods=['GET'])
def get_available_ingredients():
    class_list = get_class_list()
    # Remove underscores from class labels
    class_list = [re.sub(r'_', ' ', class_label) for class_label in class_list]

    # Capitalize the first letter of each word
    class_list = [class_label.title() for class_label in class_list]
                  
    print("Class list:", class_list)
    return jsonify({'ingredients': class_list})

@app.route("/chatbotresponse", methods=['POST'])
def get_response():
    userText = request.json.get('msg')

    # Autocorrect user input
    userText = autocorrect_text(userText)

    if userText is None:
        return jsonify({'message': "Sorry, I didn't catch that."})
    
    else: 
        response, intents = get_bot_response(userText)
        print("Response: ", response)
        print("Intents: ", intents)

        if intents[0]['intent'] == 'RequestIngredientRecipe':
            found_ingredients = []
            global recipe_list

            #found_ingredients = extract_ingredients_from_text(userText)
            print("Found ingredients: ", found_ingredients)
            if found_ingredients:
                print("Found ingredients: ", found_ingredients)
                recipe_list = generate_recipe(found_ingredients)
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
    