import React from 'react';
import RecipeCard from '../RecipeCard/RecipeCard'; 
import './Discover.css'; // You can add global styles in App.css
import recipesData from './recipes.json';

const Discover = () => {
    return (
      <div className="discover">
        {/* Render the first two recipes */}
        {recipesData.slice(0, 2).map((recipe, index) => (
          <RecipeCard key={index} recipe={recipe} />
        ))}
      </div>
    );
  };
  
export default Discover;