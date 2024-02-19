import './Login.css';
import axios from 'axios'; 
import React, { useState } from 'react';

function Login() {
  const [User, setUser] = useState(''); //the Username state
  const [Pwd, setPwd] = useState(''); // the Password state

  const HandleSubmit = (e) => {
    e.preventDefault();
    axios.post('/login', { Username: User, Password: Pwd }) // Syntax is Key: useState value
      .then(response => {
        console.log('Id', response.data.exists);
        if (response.data.exists === 1) {
          window.location.href = 'http://localhost:3000/Menu';
          console.log("Succesful");
        } 
      })
      .catch(error => {
        alert('Wrong Credentials! Try again.');
        console.error('Error:', error);
      });
  }

  return (
    
    <div className="Login typewriter" class = "background">&nbsq;
      <div class= "container" id="container">
          <div class = "form-container">
            <form action="#">
            <h1>Login</h1>
           
            <form onSubmit={HandleSubmit} className="alignLeft">
              <label>
                <input value={User} onChange={(e) => setUser(e.target.value)} type="text" name="Username" placeholder='Username'/>
              </label><br></br>
              <br></br>
              <label>
                <input value={Pwd} onChange={(e) => setPwd(e.target.value)} type="password" name="Password" placeholder='Password'/>
              </label><br></br><br></br>
              <button type="submit" className='loginBtn' style={{ marginBottom: 50 }}>Login</button>
            </form>
              <a href="http://localhost:3000/CreateAccount" className="alignRight" class = "createaccounttext">Create Account</a>
            </form>
          </div>
          <div class="overlay-conatiner">
            <div class="overlay">
            <div class="overlay-panel overlay-right">
            <h1>FYP 20297501</h1>
            <p>This project integrates object detection and classification with natural language processing to provide recipe recommendations. </p>
            </div>
            </div>
          </div>
      </div>
      <div id="grad1"></div>
    </div>
  );
}

export default Login;
