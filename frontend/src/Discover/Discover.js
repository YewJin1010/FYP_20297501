import React, { useEffect, useState } from 'react';
import RecipeCard from '../RecipeCard/RecipeCard'; 
import './Discover.css'; // You can add global styles in App.css
import recipesData from './recipes.json';

const Discover = () => {
  const [randomRecipes, setRandomRecipes] = useState([]);

 
  useEffect(() => {
    // Shuffle the recipesData array
    const shuffledRecipes = [...recipesData].sort(() => Math.random() - 0.5);

    // Take the first two recipes from the shuffled array
    const selectedRecipes = shuffledRecipes.slice(0, 9);

    // Set the state to trigger a re-render
    setRandomRecipes(selectedRecipes);
  }, []); // Empty dependency array ensures this runs only once on mount

  return (
    <div className="discover">
      {/* Render the randomly selected recipes */}
      {randomRecipes.map((recipe, index) => (
        <RecipeCard key={index} recipe={recipe} />
      ))}
    </div>
  );
};

export default Discover;
