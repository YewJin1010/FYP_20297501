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

  // Use a single state for preview images and file objects
  useEffect(() => {
    console.log('Mode:', mode);
  }, [mode]);

  // Reset preview images when mode changes
  useEffect(() => {
    if (mode === 'Object') {
      setPreviewImagesText([]);
      setFileObjectsText([]);
  } else {
      setPreviewImagesObject([]);
      setFileObjectsObject([]);
  }
}, [mode]);

  const handleDetectChange = () => {
    setMode(mode === 'Object' ? 'Text' : 'Object')
    console.log('Mode:', mode);
  }

  const handleFileChange = (event) => {
    if (event.target.files) {
        let files = Array.from(event.target.files);
        let blobs = files.map(file => URL.createObjectURL(file)); // Convert files to Blobs
        
        if (mode === 'Object') {
            setPreviewImagesObject(prevImages => [...prevImages, ...blobs]);
            let newFileObjects = files.map(file => new File([file], file.name, { type: file.type })); // Create File objects
            setFileObjectsObject(prevFileObjects => [...prevFileObjects, ...newFileObjects]);
            console.log('File objects:', fileObjectsObject);
        } else {
            setPreviewImagesText(prevImages => [...prevImages, ...blobs]);
            let newFileObjects = files.map(file => new File([file], file.name, { type: file.type })); // Create File objects
            setFileObjectsText(prevFileObjects => [...prevFileObjects, ...newFileObjects]);
            console.log('File objects:', fileObjectsText);
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
    /*
    console.log('Submitting images');
    console.log('File objects:', fileObjects); // Check if fileObjects contains data
    if (fileObjects.length === 0) {
        alert('Please select an image to upload');
        return;
    }

    var formData = new FormData();
    fileObjects.forEach((file, index) => {
      formData.append(`image_${index}`, file);
    });
    
    let uploadRoute = '/upload_object';
    if (mode === 'Text') {
      uploadRoute = '/upload_text';
    }

    try {
      let response = await axios.post(uploadRoute, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
      console.log('Response from server:', response);
      if (response.status === 200) {
        console.log('Images uploaded successfully');
        window.location.href = '/chatbot';
      }
    } 
    catch (error) {
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
    */
  }

  return (
    <div className="background">
      <br /><br /><br /><br /><br />
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
