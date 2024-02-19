import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';

import App from './landing/App';
import Menu from './Menu/Menu';
import Login from './Login/Login';
import CreateAccount from './CreateAccount/CreateAccount';
import UploadImage from './UploadImage/UploadImage';
import RecipeCard from './RecipeCard/RecipeCard';
import Discover from './Discover/Discover';
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
    path: '/menu',
    element: <Menu />,
  },
  {
    path: '/login',
    element: <Login />,
  },
  {
    path: '/createaccount',
    element: <CreateAccount />,
  },
  {
    path: '/UploadImage',
    element: <UploadImage />,
  },
  {
    path: '/RecipeCard',
    element: <RecipeCard />,
  },
  {
    path: '/Discover',
    element: <Discover />,
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
