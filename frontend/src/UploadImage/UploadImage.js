import React, { useState, useEffect } from 'react';
import './UploadImage.css';
import axios from 'axios';

function UploadImage() {
  const [isPopupOpen, setPopupOpen] = useState(false);
  const [ingredients, setIngredients] = useState([]);
  const [previewImagesObject, setPreviewImagesObject] = useState([]);
  const [previewImagesText, setPreviewImagesText] = useState([]);
  const [fileObjectsObject, setFileObjectsObject] = useState([]);
  const [fileObjectsText, setFileObjectsText] = useState([]);
  const [mode, setMode] = useState('Object');
 
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
        setPopupOpen(true);
      })
      .catch(error => {
        console.error('Error getting ingredients:', error);
      });
    setPopupOpen(true);
  }

  const closePopup = () => {
    console.log('Closing pop-up');
    setPopupOpen(false);
  }

  
  const submitImages = async () => {

    const progressBar = document.querySelector('.progress-bar');
    progressBar.style.width = '0%';

    console.log('Submitting images');
    console.log('File objects Object:', fileObjectsObject);
    console.log('File text Object:', fileObjectsText);

    let object_results = [];
    let text_results = [];

    if (fileObjectsObject.length === 0 && fileObjectsText.length === 0) {
        alert('Please select an image to upload');
        return;
    }
 
    try{
      var formDataObject = new FormData();
      fileObjectsObject.forEach((file, index) => {
        formDataObject.append(`image_${index}`, file);
      });	
      progressBar.style.width = '20%';
      var formDataText = new FormData();
      fileObjectsText.forEach((file, index) => {
        formDataText.append(`image_${index}`, file);
      });
      
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
              
          }
      }

      // Combine results and send to backend
    let combined_results = object_results.concat(text_results);
    console.log('Combined results:', combined_results);
    
    
    try {
      let response = await axios.post('/get_recipes', combined_results);
      console.log('Response from get_recipes:', response);
      if (response.status === 200) {
          console.log('Recipes received successfully:', response.data);
          
      }
    } catch (error) {
        console.error('Error getting recipes:', error);
    }
    
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
    <div className="background">
      <br /><br /><br /><br /><br />
      
      <div class="progress">
          <div class="progress-bar" role="progressbar" style={{ width: '0%' }} aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"></div>
      </div>
      <h3 className='mode-text'>{mode} Mode</h3>
      <div className="upload">
        <div className="upload-wrapper">
          <div className="upload-img">
          {(mode === 'Object' ? previewImagesObject : previewImagesText).map((image, index) => (
              <div key={index} className="uploaded-img">
                <img src={image} alt={`Preview ${index}`} />
                <button type="button" className="remove-btn" onClick={() => handleRemoveImage(index)}>
                  <b className='remove-btn-text'>x</b>
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
              <img src="assets/upload.png" alt="upload icon" />
            </div>
            <p className="upload-area-text">Select images</p>
          </div>
          <input type="file" className="visually-hidden" id="upload-input" multiple onChange={handleFileChange} />
        </div>
      </div>
      <div className="action-buttons">
        <div className="popup-container">
          <button className='openPopup' onClick={openPopup}>Available Ingredients</button>
            {isPopupOpen && (
              <div id="popup" className="popup">
                <div className="popup-content">
                  <span className="close" onClick={closePopup}>&times;</span>
                  <p className='ingre-title'>Ingredients available <br></br>for detection:</p>
                  <ul className='ingre-items'>
                    {ingredients.map((ingredient, index) => (
                      <li key={index}>{ingredient}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}    
        </div>
        <button className="submit-btn" onClick={submitImages}>Submit</button>       
        <button className="info-btn">Info</button>   
        <button className='change-btn' onClick={handleDetectChange}>Change Detector</button>
        </div>
    </div>
  );
}

export default UploadImage;
