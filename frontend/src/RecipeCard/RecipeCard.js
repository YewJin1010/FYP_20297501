import React from 'react';
import './RecipeCard.css';

const RecipeCard = ({ recipe }) => {
  return (
    <div className="recipe-card">
      <h2>{recipe.title}</h2>
      <p>Source: {recipe.source}</p>
      <p>Tags: {recipe.tags.join(', ')}</p>
      {/* Add more details as needed */}
      <a href={recipe.url} target="_blank" rel="noopener noreferrer">
        View Recipe
      </a>
    </div>
  );
};

export default RecipeCard;
