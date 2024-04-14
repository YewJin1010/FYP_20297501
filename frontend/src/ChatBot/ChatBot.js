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

  useEffect(() => {
    const fetchRecipes = async () => {
      try {
        const response = await axios.get('/recipe');
        console.log('Recipes:', response.data.recipes);
        formatRecipesIntoMessages(response.data.recipes);
      } catch (error) {
        console.error('Error fetching recipes:', error);
      }
    };
    
    fetchRecipes();
  }, []);

  const formatRecipesIntoMessages = (recipes) => {
    let newMessages = [];
  
    if (recipes.length > 0) {
      const recipe = recipes[0];
      const [title, directions] = recipe.split("directions: "); // Split the recipe into title and directions and remove 'directions:'
  
      const cleanTitle = title.replace('title:', '').trim(); // Remove 'title:' 
      const cleanDirections = directions
        .slice(1, -1) // Remove opening and closing quotes 
        .replace(/', '/g, '') // Remove all occurrences of ', '
        .split('. ') // Split by '. ' to get individual sentences
        .flatMap(sentence => sentence.split('.').map(step => step.trim())) // Split each sentence by '.' and flatten the resulting array
        .filter(sentence => sentence !== ''); // Remove any empty strings

      const titleMessage = {
        sender: "Bot",
        message: (
          <div>
            <p><strong>Title:</strong> {cleanTitle}</p>
            <p><strong>Directions:</strong></p>
            <ol>
            {cleanDirections.map((step, index) => (
              <li key={index}>
                <p>{step}</p>
              </li>
            ))}
            </ol>
          </div>
        ),
        type: "bot",
        time: getTime()
      };
  
      newMessages.push(titleMessage);
    } else {
      const noRecipesMessage = {
        sender: "Bot",
        message: <p>No recipes found.</p>,
        type: "bot",
        time: getTime()
      };
      newMessages.push(noRecipesMessage);
    }
  
    setMessages(newMessages);
};
  
  
  const getTime = () => {
    const now = new Date();
    const hours = now.getHours().toString().padStart(2, '0');
    const minutes = now.getMinutes().toString().padStart(2, '0');
    return `${hours}:${minutes}`;
  };

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
                const recipeMessage = formatRecipesIntoMessages(recipe, index);
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