import './Menu.css';
import { useState } from 'react';

function studentMenu() {


  return (
    <div className="Menu">
      <div class="containerMenu">
        <nav>
          <a href='/Menu' className='logo'></a>
            <div class="topnav">
              <a href="/">Log Out</a>
              <a href="/information">Information</a>
              <a href="/setting">Setting</a>
              <a href="/Menu">Home</a>
            </div>
        </nav>

        <div className='card cardTop'>
          <div class="inner_part">
            <label for="slideImg" class="img1">
              <img class="img_1"></img>
            </label>
            <div class="content content_1">
              <div class="title" style={{marginLeft:50, textAlign:'left'}}>Schedule</div>
              <div class="text" style={{marginLeft:50}}>
                   Access your upcoming lessons and stay on top of your schedule. 
                   Keep track of your booked classes and never miss a session again.
              </div>
              <a href='/studentschedule'><button>Check Schedule</button></a>
            </div>
          </div>
        </div>      

        <div className='card'>
          <div class="inner_part">
            <label for="slideImg" class="img2">
              <img class="img_2"></img>
            </label>
            <div class="content content_1">
              <div class="title" style={{marginLeft:50, textAlign:'left'}}>Bookings</div>
              <div class="text" style={{marginLeft:50}}>
                   Take your music skills to the next level with our experienced teachers. 
                   Book lessons for guitar, piano, drums, and more, with our easy-to-use system. 
                   All levels and styles are welcome - start pursuing your passion today!
              </div>   
              <a href='/chooseclass'><button>Book Now</button></a>
            </div>
          </div>
        </div>  

        <div className='card'>
          <div class="inner_part">
            <label for="slideImg" class="img3">
              <img class="img_3"></img>
            </label>
            <div class="content content_1">
              <div class="title" style={{marginLeft:50, textAlign:'left'}}>Transaction History</div>
              <div class="text" style={{marginLeft:50}}>
                  Keep track of your transactions with ease. Check your transaction history to 
                  stay on top of your finances and monitor your spending. 
              </div>
              <a href='/studenttransaction'><button>Check History</button></a>
            </div>
          </div>
        </div>  

        <div className='card'>
          <div class="inner_part">
            <label for="slideImg" class="img4">
              <img class="img_4"></img>
            </label>
            <div class="content content_1">
              <div class="title" style={{marginLeft:50, textAlign:'left'}}>FAQ</div>
              <div class="text" style={{marginLeft:50}}>
                  Have questions? Check out our FAQ section for answers to our most commonly asked questions.
                  From billing and payments to troubleshooting and support, our FAQ has you covered. 
                  Browse our FAQs to get the answers you need quickly and easily.
              </div>
              <a href='/studentfaq'><button>Need Help?</button></a>
            </div>
          </div>
        </div>  

        

    </div>

      <div id="grad1"></div>
    </div>
  );
}

export default studentMenu;