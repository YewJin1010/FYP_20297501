import React from 'react';
import './RecipeCard.css';

const RecipeCard = ({ recipe }) => {
  return (
    <div className="recipe-card">
      <div className="image-container">
        <img
          src={recipe.image} 
          alt={recipe.title}
        />
      </div>
      <div className="recipe-details">
        <h2>{recipe.title}</h2>
        <p>Source: {recipe.source}</p>
        <p>Tags: {recipe.tags.join(', ')}</p>
        {/* Add more details as needed */}
        <a href={recipe.url} target="_blank" rel="noopener noreferrer">
          View Recipe
        </a>
      </div>
    </div>
  );
};

export default RecipeCard;
