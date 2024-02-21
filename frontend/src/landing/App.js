import './App.css';
import axios from 'axios'; 
import React, { useState } from 'react';

function App() {


  return (
    
    <div className="App typewriter" class = "background">
    <nav style={{ height: '50px', position: 'fixed', top: '0', width: '100%', backgroundColor: 'black' }}>
      <div className="nav-container" style={{ paddingTop: 0, height: '55px', backgroundColor: 'orange', display: 'flex', justifyContent: 'flex-end' }}>
        <ul className="nav-items" style={{ listStyleType: 'none', margin: 0, padding: 0, display: 'flex', alignItems: 'center', height: '100%', color: 'black' }}>
          <li><h3 style={{padding: '15px', fontSize: '20px', fontFamily: 'Georgia, serif', padding: '20px 40px 10px 40px'}}>Inspire me Today!</h3></li>
          <li><a href="/" style={{ textDecoration: 'none', color: 'black', padding: '15px 30px 15px 30px', display: 'block'}} onMouseOver={(e) => e.target.style.backgroundColor = '#ffcc66'} onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}><b>Home</b></a></li>
          <li><a href="/studentNotification" style={{ textDecoration: 'none', color: 'black', padding: '15px 30px 15px 30px', display: 'block', backgroundColor: '#ffcc66'}} onMouseOver={(e) => e.target.style.backgroundColor = '#ffcc66'} onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}><b>About</b></a></li>
          <li><a href="/studentsetting" style={{ textDecoration: 'none', color: 'black', padding: '15px 30px 15px 30px', display: 'block', backgroundColor: '#ffcc66'}} onMouseOver={(e) => e.target.style.backgroundColor = '#ffcc66'} onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}><b>Contact</b></a></li>
          <li><a href="/StudentInformation" style={{ textDecoration: 'none', color: 'black', padding: '15px 30px 15px 30px', display: 'block', backgroundColor: '#ffcc66'}} onMouseOver={(e) => e.target.style.backgroundColor = '#ffcc66'} onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}><b>Setting</b></a></li>
          <li><a href="/login" style={{ textDecoration: 'none', color: 'black', padding: '15px 30px 15px 30px', display: 'block', backgroundColor: '#ffcc66'}} onMouseOver={(e) => e.target.style.backgroundColor = '#ffcc66'} onMouseOut={(e) => e.target.style.backgroundColor = 'transparent'}><b>Log Out</b></a></li>
        </ul>
      </div>
    </nav>
      <br></br><br></br><br></br><br></br><br></br>
      <h1 class = "title">FYP 20297501 DEMO</h1>
      <br></br><br></br><br></br><br></br><br></br><br></br>
      <a href="http://localhost:3000/UploadImage" class = "landing-btn">Ready? Let's Bake</a>
    </div>
  );
}

export default App;
