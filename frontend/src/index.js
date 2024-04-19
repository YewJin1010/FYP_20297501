import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';

import App from './landing/App';
import UploadImage from './UploadImage/UploadImage';
import ChatBot from './ChatBot/ChatBot';

import reportWebVitals from './reportWebVitals';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
  },

  {
    path: '/UploadImage',
    element: <UploadImage />,
  },
  {
    path: '/ChatBot',
    element: <ChatBot />,
  },
]);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
