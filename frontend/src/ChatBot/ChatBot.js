import React, { useState, useRef, useEffect } from 'react';
import './ChatBot.css';
import axios from 'axios';

function getRandomColor() {
  // Generate a random hexadecimal color code
  return '#' + Math.floor(Math.random()*16777215).toString(16);
}

function ChatBot() {
  const [messages, setMessages] = useState([]);
  const userInputRef = useRef(null);
  const [inputValue, setInputValue] = useState('');

  const recipeFormat = (recipe, index) => {
    const ingredientsArray = recipe.ingredients.split(';'); // Split ingredients by semicolon
    const ingredientsList = ingredientsArray.map((ingredient, idx) => (
      <p key={`ingredient_${idx}`}>{ingredient.trim()}</p> // Trim whitespace around each ingredient
    ));

    const splitDirections = recipe.directions.split('.').filter(sentence => sentence.trim() !== ''); // Split directions by '.' and filter out empty sentences
    const indexedDirections = splitDirections.map((sentence, index) => `${index + 1}. ${sentence.trim()}`); // Add index number to each sentence
    const numberedDirections = indexedDirections.map((direction, idx) => (
      <p key={`direction_${idx}`}>{direction}.</p> // Wrap each direction in a paragraph tag to create line break
    ));
    const recipeMessage = (
      <div key={`recipe_${index}`}>
        <p><strong>Recipe {index + 1}:</strong> {recipe.title}</p>
        <p><strong>Ingredients:</strong><br />{ingredientsList}</p>
        <p><strong>Directions:</strong><br />{numberedDirections}</p>
      </div>
    );
    return recipeMessage;
  }

  useEffect(() => {
    const formatRecipesIntoMessages = (recipes) => {
      let newMessages = [...messages]; // Copy existing messages

       // Check if there are any recipes
      if (recipes.length > 0) {
        // Add a greeting message if there are recipes
        const greetingMessage = {
          sender: "Bot",
          message: <p>Based on your ingredients, here are some recipes you can try:</p>,
          type: "bot",
          time: getTime()
        };
        newMessages.push(greetingMessage);

        recipes.forEach((recipe, index) => {
          const recipeMessage = recipeFormat(recipe, index);
          newMessages.push({ sender: "Bot", message: recipeMessage, type: "bot", time: getTime() });
        });
        setMessages(newMessages); // Update messages state with new recipe messages
      };
    };
      
    // Function to fetch recipes when component mounts
    const fetchRecipes = async () => {
      try {
        const response = await axios.get('/recipe');
        console.log('Recipes:', response.data.recipes);
        formatRecipesIntoMessages(response.data.recipes);
      } catch (error) {
        console.error('Error fetching recipes:', error);
      }
    };
    // Call the function to fetch recipes when component mounts
    fetchRecipes();
  }, []);

  useEffect(() => {
    const handleKeyPress = (event) => {
      if (event.key === 'Enter') {
        if (event.shiftKey) {
          // Handle shift enter behavior
        } else {
          sendMessage(); // Call sendMessage directly
          // clear the input field
          setInputValue('');
        }
      }
    };
    userInputRef.current.addEventListener('keypress', handleKeyPress);
    return () => {
      userInputRef.current.removeEventListener('keypress', handleKeyPress);
    }
  }, []);
  
  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && event.shiftKey) {
      event.preventDefault();
      setInputValue(inputValue + '\n');
    }
  }
  const getTime = () => {
    const now = new Date();
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
  };

  const appendMessage = (sender, message, type) => {
    setMessages(prevMessages => [...prevMessages, { sender, message, type, time: getTime() }]);
  };

  const sendMessage = () => {
    const userMessage = userInputRef.current.value.trim();
    if (userMessage !== "") {  
      console.log('Sending message:', userMessage);
      axios.post('/chatbotresponse', {msg: userMessage})
        .then(response => {
          console.log('Response from server:', response);
          const botMessage = response.data.message;
          appendMessage("Bot", botMessage, "bot");

          if (response.data.recipes){
            const recipes = response.data.recipes;
            console.log('Recipes:', recipes);
    
            let newMessages = [...messages];
            if (recipes.length > 0) {
              const greetingMessage = {
                sender: "Bot",
                message: <p>Here are some recipes you can try:</p>,
                type: "bot",
                time: getTime()
              };
              newMessages.push(greetingMessage);
      
              recipes.forEach((recipe, index) => {
                const recipeMessage = recipeFormat(recipe, index);
                newMessages.push({ sender: "Bot", message: recipeMessage, type: "bot", time: getTime() });
              });
              setMessages(newMessages);
            }
          }
        })  
        .catch(error => {
          console.error('Error sending message:', error);
        });
      appendMessage("You", userMessage, "user");
      setInputValue('');
    }
  };

  useEffect(() => {
    // Scroll to the bottom whenever messages change
    const chatBox = document.getElementById("chatBox");
    chatBox.scrollTop = chatBox.scrollHeight;
  }, [messages]);


  return (
    <div className="background">
      <div className="chat-box" id="chatBox">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.type}`}>
            {msg.type === 'bot' && (
              <div className="bot-message-container">
                <img className="bot-profile-image" src="assets/robot.jpg" alt="chatbot"></img>
                <div className="bot-message-bubble">
                  <div className = "bot-message-sender">{msg.sender}</div>
                  <div className="bot-message-text">{msg.message}</div>
                  <div className="bot-message-time">{msg.time}</div>
                </div>
              </div>
            )}
            {msg.type === 'user' && (
              <div className="user-message-container">
                <img className="user-profile-image" src="assets/user.jpeg" alt="user"></img>
                <div className="user-message-bubble">
                  <div className = "user-message-sender">{msg.sender}</div>
                  <div className="user-message-text">{msg.message}</div>
                  <div className="user-message-time">{msg.time}</div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="input-container">
        <textarea
          className='user-input' 
          type="text" 
          ref={userInputRef} 
          value = {inputValue}
          onChange = {handleInputChange}
          onKeyDown = {handleKeyDown}
          placeholder="Type your message..." 
        />
        <button className="send-btn" onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default ChatBot;