import React, { useState, useRef, useEffect } from 'react';
import './ChatBot.css';
import axios from 'axios';

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

  function generateCakeList() {
    const cakeList = ['Chocolate cake', 'Vanilla cake', 'Strawberry cake', 'Red velvet cake', 'Carrot cake', 'Cheesecake', 'Ice cream cake'];
    const renderedCakeList = cakeList.map((cakeName, index) => {
      const imageUrl = `assets/chef_${index + 1}.jpg`;
      return (
        <li key={index}>
          <div className="previous-discussion">
            <img className="profile-picture" src={imageUrl} alt={cakeName}></img>
            <div className="discussion-info">
              <p><b>{cakeName}</b></p>
              <p className="time">7:30 p.m. 12/4/2024</p>
            </div>
          </div>
        </li>
      );
    });
  
    return renderedCakeList;
  }  

  return (
    <div className="chat-bg">
      <nav className='nav'>
          <h3 style={{ marginLeft: "45%" }}>
            FYP 20297501 DEMO
          </h3>
        </nav>
      <div className="chat-grid">
        <div className="titles-column">
        <h2 className='title-header'>Previous Discussions</h2>
        <div className='title-container'>
          <ul className='title-list'>
            {generateCakeList()}
          </ul>
        </div>
    </div>
        <div className="chat-column">
          <div className="chat-box" id="chatBox">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                {msg.type === 'bot' && (
                  <div className="bot-message-container">
                    <img className="bot-profile-image" src="assets/chef_4.jpg" alt="chatbot" />
                    <div className="bot-message-bubble">
                      <div className="bot-message-sender">{msg.sender}</div>
                      <div className="bot-message-text">{msg.message}</div>
                      <div className="bot-message-time">{msg.time}</div>
                    </div>
                  </div>
                )}
                {msg.type === 'user' && (
                  <div className="user-message-container">
                    <img className="user-profile-image" src="assets/user.jpeg" alt="user" />
                    <div className="user-message-bubble">
                      <div className="user-message-sender">{msg.sender}</div>
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
              className="user-input"
              type="text"
              ref={userInputRef}
              value={inputValue}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="Type your message..."
            />
            <button className="send-btn" onClick={sendMessage}>
              Send
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatBot;