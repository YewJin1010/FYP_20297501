import React, { useState, useEffect } from 'react';
import './UploadImage.css';
import axios from 'axios';

function UploadImage() {
  const [isPopupOpen, setPopupOpen] = useState(false);
  const [isInfoOpen, setInfoOpen] = useState(false);
  const [ingredients, setIngredients] = useState([]);
  const [previewImagesObject, setPreviewImagesObject] = useState([]);
  const [previewImagesText, setPreviewImagesText] = useState([]);
  const [fileObjectsObject, setFileObjectsObject] = useState([]);
  const [fileObjectsText, setFileObjectsText] = useState([]);
  const [mode, setMode] = useState('Object');
  const [showProgressBar, setShowProgressBar] = useState(false);
  const [backgroundOpacity, setBackgroundOpacity] = useState(1);
  const [buttonsDisabled, setButtonsDisabled] = useState(false);
  
  // Log current mode
  useEffect(() => {
    console.log('Mode1:', mode);
  }, [mode]);

  // Log file objects and preview images
  useEffect(() => {
      console.log('File objects Object:', fileObjectsObject);
      console.log('Preview images Object:', previewImagesObject);
      console.log('Number of images Object:', previewImagesObject.length);

      console.log('File objects Text:', fileObjectsText);
      console.log('Preview images Text:', previewImagesText);
      console.log('Number of images Text:', previewImagesText.length);
  }, [fileObjectsObject, previewImagesObject, fileObjectsText, previewImagesText]);

  // Change detector mode
  const handleDetectChange = () => {
    setMode(mode === 'Object' ? 'Text' : 'Object')
  }

  // Handle file change
  const handleFileChange = (event) => {
    if (event.target.files) {
        let files = Array.from(event.target.files);
        let blobs = files.map(file => URL.createObjectURL(file)); // Convert files to Blobs
        
        if (mode === 'Object') {
            setPreviewImagesObject(prevImages => [...prevImages, ...blobs]);
            let newFileObjects = files.map(file => new File([file], file.name, { type: file.type })); // Create File objects
            setFileObjectsObject(prevFileObjects => [...prevFileObjects, ...newFileObjects]);
        } else {
            setPreviewImagesText(prevImages => [...prevImages, ...blobs]);
            let newFileObjects = files.map(file => new File([file], file.name, { type: file.type })); // Create File objects
            setFileObjectsText(prevFileObjects => [...prevFileObjects, ...newFileObjects]);
        }
    } 
};

  const handleRemoveImage = (index) => {
    if (mode === 'Object') {
      let updatedImages = [...previewImagesObject];
      updatedImages.splice(index, 1);
      setPreviewImagesObject(updatedImages);

      let updatedFileObjects = [...fileObjectsObject];
      updatedFileObjects.splice(index, 1);
      setFileObjectsObject(updatedFileObjects);
    } else {
      let updatedImages = [...previewImagesText];
      updatedImages.splice(index, 1);
      setPreviewImagesText(updatedImages);

      let updatedFileObjects = [...fileObjectsText];
      updatedFileObjects.splice(index, 1);
      setFileObjectsText(updatedFileObjects);
    }
  };

  const openPopup = () => {
    console.log('Opening pop-up');
    axios.get('/ingredients')
      .then(response => {
        console.log('Response from server:', response);
        setIngredients(response.data.ingredients);
        console.log('Ingredients:', ingredients);
        setPopupOpen(true);
      })
      .catch(error => {
        console.error('Error getting ingredients:', error);
      });
  };  

  const closePopup = () => {
    console.log('Closing pop-up');
    setPopupOpen(false);
  }

  const openInfo = () => {
    console.log('Opening info');
    setInfoOpen(true);
  }

  const closeInfo = () => {
    console.log('Closing info');
    setInfoOpen(false);
  }

  
  const submitImages = async () => {

    console.log('Submitting images');
    console.log('File objects Object:', fileObjectsObject);
    console.log('File text Object:', fileObjectsText);

    let object_results = [];
    let text_results = [];

    if (fileObjectsObject.length === 0 && fileObjectsText.length === 0) {
        alert('Please select an image to upload');
        return;
    }

    setShowProgressBar(true);
    const progressBar = document.querySelector('.progress-bar');
    progressBar.style.width = '0%';
    setBackgroundOpacity(0.5);
    setButtonsDisabled(true);
 
    try{
      var formDataObject = new FormData();
      fileObjectsObject.forEach((file, index) => {
        formDataObject.append(`image_${index}`, file);
      });	
    
      var formDataText = new FormData();
      fileObjectsText.forEach((file, index) => {
        formDataText.append(`image_${index}`, file);
      });
      progressBar.style.width = '10%';
      // Submit object images
      if (formDataObject && Array.from(formDataObject.keys()).length > 0) {
          let responseObject = await axios.post('/upload_object', formDataObject, {
              headers: {
                  'Content-Type': 'multipart/form-data'
              }
          });
          console.log('Response from server for object images:', responseObject);
          if (responseObject.status === 200) {
              console.log('Object images uploaded successfully');
              object_results = responseObject.data.class_labels;
              console.log('Object results FRONTEND:', object_results);
              progressBar.style.width = '40%';
          }
      }

      // Submit text images
      if (formDataText && Array.from(formDataText.keys()).length > 0) {
          let responseText = await axios.post('/upload_text', formDataText, {
              headers: {
                  'Content-Type': 'multipart/form-data'
              }
          });
          console.log('Response from server for text images:', responseText);
          if (responseText.status === 200) {
              console.log('Text images uploaded successfully');
              text_results = responseText.data.text_detection_results;
              console.log('Text results FRONTEND:', text_results);
              progressBar.style.width = '60%';
          }
      }

      // Combine results and send to backend
    let combined_results = object_results.concat(text_results);
    console.log('Combined results:', combined_results);
    progressBar.style.width = '80%';
    
    try {
      let response = await axios.post('/get_recipes', combined_results);
      console.log('Response from get_recipes:', response);
      if (response.status === 200) {
          console.log('Recipes received successfully:', response.data);
          progressBar.style.width = '90%';
      }
    } catch (error) {
        console.error('Error getting recipes:', error);
    }
    progressBar.style.width = '100%';
    //window.location.href = '/chatbot';
        
    } catch (error) {
      console.error(error);
      let errorMessage = "An error occurred";
      if (error.response && error.response.data && error.response.data.error) {
        let fullErrorMessage = error.response.data.error.toString();
        let match = fullErrorMessage.match(/cannot identify image file/i);
        if (match) {
            errorMessage = match[0];
        }
    }
    alert(errorMessage); 
    }
  }
  return (
    <div>
      <div className="progress" id="progressBar" style={{ visibility: showProgressBar ? 'visible' : 'hidden' }}>
        <div className="progress-bar" role="progressbar" style={{ width: '0%' }} aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
      </div>
      <div className="uploadimg_bg" style={{ opacity: backgroundOpacity }}>
      <nav className='nav'>
          <h3 style={{ marginLeft: "45%" }}>
            FYP 20297501 DEMO
          </h3>
        </nav>
      <div className="gui-bg">
        <div className="gui-grid">
          <div className="left-side">
          <h3 className='mode-text'>{mode} Mode</h3>
            <div className="upload">
              <div className="upload-wrapper">
                <div className="upload-img" style={{ maxHeight: '200px', overflow: 'auto' }}>
                  {(mode === 'Object' ? previewImagesObject : previewImagesText).map((image, index) => (
                    <div key={index} className="uploaded-img">
                      <img src={image} alt={`Preview ${index}`}/>
                      <button type="button" className="remove-btn" onClick={() => handleRemoveImage(index)}>
                        <b><u className='remove-btn-text'>x</u></b>
                      </button>
                    </div>
                  ))}
                </div>
                <div className="upload-info">
                  <p>
                    <span className="upload-info-value">{mode === 'Object' ? previewImagesObject.length : previewImagesText.length}</span> file(s) uploaded.
                  </p>
                </div>
                  <div className="upload-area" onClick={() => document.getElementById('upload-input').click()}>
                    <div className="upload-area-img">
                      <img src="assets/upload.png" alt="upload icon" style={{ maxHeight: '100px'}}/>
                    </div>
                    <p className="upload-area-text">Select images</p>
                  </div>
                  <input type="file" className="visually-hidden" id="upload-input" multiple onChange={handleFileChange} disabled={buttonsDisabled}/>
                </div>
              </div>
            </div>
            <div className="right-side">
              <h3 className='menu-text'>Menu</h3>
              <div className="action-btns">
                <div className="button-grid">
                  <button className="info-btn" onClick={openInfo} disabled={buttonsDisabled}>Info</button>
                  <button className="submit-btn" onClick={submitImages} disabled={buttonsDisabled}>Submit</button>
                  <button className="openPopup" onClick={openPopup} disabled={buttonsDisabled}>Available Ingredients</button>
                  <button className="change-btn" onClick={handleDetectChange} disabled={buttonsDisabled}>Change Detector</button>
                </div>
                <div className="popup" style={{ display: isPopupOpen ? 'block' : 'none' }}>
                  <div className="popup-content">
                    <h2>Available Ingredients</h2>
                    <div className="ingredients-container">
                      <ul className="ingredients-list">
                        {ingredients.map((ingredient, index) => (
                          <li key={index}>{ingredient}</li>
                        ))}
                      </ul>
                    </div>
                    <button className="close-btn" onClick={closePopup}>Close</button>
                  </div>
                </div>
                <div className="info" style={{ display: isInfoOpen ? 'block' : 'none' }}>
                  <div className="info-content">
                    <h2>Need Help?</h2>
                    <div className="info-container">	
                      <ul className='info-list'>
                        <li>
                          <b><u>Upload Image</u></b><br/>
                          - Upload an image of your ingredients to get started. <br />
                        </li>
                        <li>
                          <b><u>Change Detector</u></b><br/>
                          - Click on "Change Detector" to switch between detecting objects and text. <br />
                          - Your current mode is displayed at the top. <br />
                          - Your images will be saved when you switch modes. <br />
                        </li>
                        <li>
                          <b><u>Submit</u></b><br/>
                          - Click on "Submit" to get recipes based on the detected ingredients. <br />
                          - The submit button will be disabled until an image is uploaded. <br />
                          - Remember to provide as many ingredients as possible for better results. <br />
                        </li>
                        <li>
                          <b><u>Available Ingredients</u></b><br/>
                          - Click on "Available Ingredients" to view a list of ingredients that can be detected. <br />
                        </li>
                        <li>
                          <b><u>Missing Something?</u></b><br/>
                          - The more ingredients you provide, the better the recipes will be.<br />
                          - If you are missing an ingredient, you can type it in the text area.<br />
                          - The ingredients are seperated by commas.<br />
                        </li>
                      </ul>
                    </div>
                    <button className="close-btn" onClick={closeInfo}>Close</button>
                  </div>
                </div>
                <div className="addons">
                  <label htmlFor="addon-input" className='addons-title'>Missing something?</label>
                  <textarea className="addon-input" rows="4" cols="60" disabled={buttonsDisabled}  placeholder= "1 cup flour, 2 teaspoons cinnamon, grapes"></textarea>
              </div>
              </div>
            </div>
        </div>
      </div>
    </div>
  </div>
  );
}

export default UploadImage;
