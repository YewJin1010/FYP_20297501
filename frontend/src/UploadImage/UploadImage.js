import './UploadImage.css'; 
import axios from 'axios'; 
import React, { useState } from 'react';

function UploadImage(){
    const [previewSrc, setPreviewSrc] = useState('');
    const [error, setError] = useState('');

    const showPreview = (event) => {
        if(event.target.files.length > 0){
            var src = URL.createObjectURL(event.target.files[0]);
            var preview = document.getElementById("file-ip-1-preview");
            preview.src = src;
            preview.style.display = "block";
        } 
    }

    const handleSubmit = () => {
        if (previewSrc) {
            // Add logic for submitting the image, if needed
            console.log('Image submitted!');
        } else {
            alert('Please upload an image before submitting.');
        }
    }

    return (
    <div className="UploadImage">
        <div className="center">
            <div className="form-input">
                <div className="preview">
                    <img id='file-ip-1-preview' src='' alt='Preview'></img>
                    <label htmlFor="file-ip-1">Upload Image</label>
                    <input type="file" id="file-ip-1" accept="image/*" onChange={showPreview}></input>
                    <button className="action-button" onClick={handleSubmit}>Submit</button>
                </div>                
            </div>
        </div>
    </div>
    );
}
export default UploadImage;
