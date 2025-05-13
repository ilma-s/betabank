"use client";

import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';
import axios from 'axios';

interface AuthContextType {
  isAuthenticated: boolean;
  token: string | null;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [token, setToken] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  // Configure axios to use token for all requests
  useEffect(() => {
    if (token) {
      console.log("Setting auth header with token");
      // Set default headers for all requests
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
      // Add an interceptor to handle 401 errors globally
      const interceptorId = axios.interceptors.response.use(
        (response) => response,
        async (error) => {
          // If we get a 401 error, log out the user
          if (error.response && error.response.status === 401) {
            console.log("401 error detected in interceptor, logging out");
            // Clear the token and auth state
            setIsAuthenticated(false);
            setToken(null);
            localStorage.removeItem('isAuthenticated');
            localStorage.removeItem('authToken');
          }
          return Promise.reject(error);
        }
      );
      
      // Clean up the interceptor when unmounting
      return () => {
        console.log("Cleaning up auth interceptor");
        axios.interceptors.response.eject(interceptorId);
        delete axios.defaults.headers.common['Authorization'];
      };
    } else {
      // Remove Authorization header if token is null
      console.log("Removing auth header as token is null");
      delete axios.defaults.headers.common['Authorization'];
    }
  }, [token]);

  // Check if user is already authenticated on load
  useEffect(() => {
    console.log("Checking localStorage for auth token");
    const authStatus = localStorage.getItem('isAuthenticated');
    const storedToken = localStorage.getItem('authToken');
    
    if (authStatus === 'true' && storedToken) {
      console.log("Found token in localStorage, restoring session");
      setIsAuthenticated(true);
      setToken(storedToken);
    } else {
      console.log("No valid token found in localStorage");
    }
    
    // Mark initialization as complete
    setIsInitialized(true);
  }, []);

  const login = async (username: string, password: string) => {
    try {
      console.log("Attempting login for user:", username);
      // Authenticate with the backend
      const response = await axios.post('http://localhost:8000/token', 
        new URLSearchParams({
          'username': username,
          'password': password
        }),
        {
          headers: {
            'Content-Type': 'application/x-www-form-urlencoded'
          }
        }
      );
      
      // Store the token and set authenticated state
      const receivedToken = response.data.access_token;
      console.log("Login successful, received token");

      // First update the headers
      console.log("Setting auth header immediately after login");
      axios.defaults.headers.common['Authorization'] = `Bearer ${receivedToken}`;
      
      // Then update the state
      setToken(receivedToken);
      setIsAuthenticated(true);
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('authToken', receivedToken);
      return true;
    } catch (error) {
      console.error("Login failed:", error);
      return false;
    }
  };

  const logout = () => {
    // First remove the headers
    delete axios.defaults.headers.common['Authorization'];
    
    // Then update state
    setIsAuthenticated(false);
    setToken(null);
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('authToken');
  };

  // Show a loading state until we've checked localStorage
  if (!isInitialized) {
    return <div className="min-h-screen flex items-center justify-center bg-[#261436]">
      <div className="text-white">Loading...</div>
    </div>;
  }

  return (
    <AuthContext.Provider value={{ isAuthenticated, token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
} 