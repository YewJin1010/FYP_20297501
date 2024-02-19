import React, { useState, useRef, useEffect } from 'react';
import './ChatBot.css';
import axios from 'axios';

function ChatBot() {
  const [messages, setMessages] = useState([]);
  const userInputRef = useRef(null);
  const [recipes, setRecipes] = useState([]);

  useEffect(() => {
    const formatRecipesIntoMessages = (recipes) => {
      let newMessages = [...messages]; // Copy existing messages
      recipes.forEach((recipe, index) => {
        const recipeMessage = (
          <div key={`recipe_${index}`}>
            <p><strong>Recipe {index + 1}:</strong> {recipe.title}</p>
            <p><strong>Ingredients:</strong><br />{recipe.ingredients}</p>
            <p><strong>Directions:</strong><br />{recipe.directions}</p>
          </div>
        );
        newMessages.push({ sender: "Bot", message: recipeMessage, type: "bot", time: getTime() });
      });
      setMessages(newMessages); // Update messages state with new recipe messages
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
        })
        .catch(error => {
          console.error('Error sending message:', error);
        });
  
      appendMessage("You", userMessage, "user");
      userInputRef.current.value = "";
    }
  };

  useEffect(() => {
    // Scroll to the bottom whenever messages change
    const chatBox = document.getElementById("chatBox");
    chatBox.scrollTop = chatBox.scrollHeight;
  }, [messages]);


  return (
    <div className="chat-container">
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
        <input type="text" ref={userInputRef} placeholder="Type your message..." />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default ChatBot;