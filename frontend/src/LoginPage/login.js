import React, {useState} from 'react';
import './login.css';
import axios from 'axios'; 

function Login() {

const [User, setUser] = useState(''); //the Username state
const [Pwd, setPwd] = useState(''); // the Password state


//this just logs the data (username and password into the console,if wanna compare with database i think need to do it here)
const HandleSubmit = (e)  => {
  e.preventDefault();
  axios.post('/login', {Username: User, Password: Pwd}) //Syntax is Key: useState value
  .then(response => {
    console.log('Id', response.data.exists);
    if (response.data.exists == 1 ){
      window.location.href = 'http://localhost:3000/adminMenu';
      console.log("Admin");
    } else if (response.data.exists == 2){
      window.location.href = 'http://localhost:3000/studentMenu';
      console.log("Student");
    } else if (response.data.exists == 3){
      window.location.href = 'http://localhost:3000/teacherMenu';
      console.log("Teacher");
    }
  })
  .catch(error => {
    alert('Wrong Credentials! Try again.');
    console.error('Error:', error);
  });
}

return (
  <div className="Login">
    <div className='containerLogin'>
      <div className='row'>

        <div className='col-6 loginForm'>
          <div className='cardLogin'>
          <h1 class="alignLeft">Welcome Back!</h1>
          <p class="subAlignLeft">Enter Your Username & Password</p>

          <form onSubmit = {HandleSubmit} class="alignLeft">
             <label> Username &nbsp;
                <input value = {User} onChange={(e)=>setUser(e.target.value)}type="text" name="Username" /> 
             </label><br></br>
             <label> Password &nbsp;&nbsp;
                <input value = {Pwd} onChange={(e)=>setPwd(e.target.value)} type="Password" name="Password" />
             </label><br></br><br></br>
                <button type="submit" className='loginBtn' style={{marginBottom:50}}> Login </button> 
          </form>

          </div>
        </div>
      </div>    
      
    </div>

    <div id="grad1"></div>
  </div>
);
}

export default Login;