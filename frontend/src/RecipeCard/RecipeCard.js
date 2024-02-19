import React, { useState, useEffect } from 'react';
import './RecipeCard.css';

const RecipeCard = ({ recipe }) => {
  const [imageSrc, setImageSrc] = useState('');

  useEffect(() => {
    const fetchImage = async () => {
      try {
        console.log('Fetching image:', recipe.image);

        const response = await fetch(recipe.image);
  
        if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          setImageSrc(url);
        } else {
          console.error(`Failed to fetch image. Status: ${response.status}`);
        }
      } catch (error) {
        console.error('Error fetching image:', error);
      }
    };
  
    if (recipe.image) {
      fetchImage();
    }
  }, [recipe.image]);

  return (
    <div className="recipe-card">
      <div className="image-container">
        {imageSrc ? (
          <img src={imageSrc} alt={recipe.title} />
        ) : (
          <p>Loading image...</p>
        )}
      </div>
      <div className="recipe-details">
        <h2>{recipe.title}</h2>
        <p>Source: {recipe.source}</p>
        <p>Tags: {recipe.tags.join(', ')}</p>
        <a href={recipe.url} target="_blank" rel="noopener noreferrer">
          View Recipe
        </a>
      </div>
    </div>
  );
};

export default RecipeCard;
