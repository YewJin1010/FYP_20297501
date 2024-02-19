import './CreateAccount.css';
import axios from 'axios'; 
import React, { useState } from 'react';

function CreateAccount() {
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
    
    <div className="App typewriter" class = "background">&nbsq;
      <div class= "container" id="container">
          <div class = "form-container">
            <form action="#">
            <h1>Create Account</h1>
           
            <form onSubmit={HandleSubmit} className="alignLeft">
              <label>
                <input value={User} onChange={(e) => setUser(e.target.value)} type="text" name="Username" placeholder='Username'/>
              </label><br></br>
              <br></br>
              <label>
                <input value={Pwd} onChange={(e) => setPwd(e.target.value)} type="password" name="Password" placeholder='Password'/>
              </label><br></br><br></br>
              <button type="submit" className='loginBtn' style={{ marginBottom: 50 }}>Sign Up</button>
            </form>
            <a href="http://localhost:3000/Login" className="alignRight" class = "logintext">Already have an account?</a>
            </form>
          </div>
          <div class="overlay-conatiner">
            <div class="overlay">
            <div class="overlay-panel overlay-right">
            <h1>Why have an Account?</h1>
            <p>Creating an account enables personalisation such as saving favorite recipes and selecting vegan or nut-free options.</p>
            </div>
            </div>
          </div>
      </div>
      <div id="grad1"></div>
    </div>
  );
}

export default CreateAccount;
