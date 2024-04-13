import './App.css';
import React, { useState, useEffect } from 'react';

function App() {
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const cakeImages = [
    "assets/cake_1.jpg",
    "assets/cake_2.jpg",
    "assets/cake_3.jpg",
    "assets/cake_4.jpg",
    "assets/cake_5.jpg"
  ];

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentImageIndex(prevIndex => (prevIndex + 1) % cakeImages.length);
    }, 3000); // Change image every 3 seconds

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="App">
      <div className="landing_bg">
        <nav className='nav'>
          <h3 style={{ marginLeft: "45%" }}>
            FYP 20297501 DEMO
          </h3>
        </nav>
        <div className="left-section">
          <header className="header">
            <h1>Inspire me TODAY!</h1> {/* Title */}
          </header>
          <p>
            Indulge in a world of deliciousness!<br />
            Creating new recipes from ingredients at hand can be a daunting task.<br />
            Let us help you create your dream cake with our user-friendly tools and <br />
            make your baking experience a breeze.
          </p>
          <br />
          <a href="http://localhost:3000/UploadImage" className="next-btn">
            Get Started Now!
          </a>
          <div className="recipe-flow">
            <img src="assets/recipe_flow.png" alt="Recipe Flow" className="recipe-flow-image" />
          </div>
          <p className='recipe-flow-paragraph'>
            Its as easy as 1, 2, 3! <br />
            Simply upload an image of your ingredients and let our AI do the rest. <br />
          </p>
        </div>
        <div className="right-section">
          <div className="cake-container">
            {cakeImages.map((image, index) => (
              <img
                key={index}
                src={image}
                alt="Cake Image"
                className={`cake-image ${index === currentImageIndex ? 'fade-in' : 'fade-out'}`}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
