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

  // Check if user is already authenticated on load
  useEffect(() => {
    const authStatus = localStorage.getItem('isAuthenticated');
    const storedToken = localStorage.getItem('authToken');
    
    if (authStatus === 'true' && storedToken) {
      setIsAuthenticated(true);
      setToken(storedToken);
    }
  }, []);

  const login = async (username: string, password: string) => {
    // For demo purposes, we'll keep the hardcoded admin credentials
    // and also try to authenticate with the backend
    if (username === 'admin' && password === 'admin') {
      setIsAuthenticated(true);
      // Create a dummy token for admin user
      const dummyToken = 'admin-demo-token-12345';
      setToken(dummyToken);
      localStorage.setItem('isAuthenticated', 'true');
      localStorage.setItem('authToken', dummyToken);
      return true;
    }
    
    try {
      // Try to get a token from the backend
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
    setIsAuthenticated(false);
    setToken(null);
    localStorage.removeItem('isAuthenticated');
    localStorage.removeItem('authToken');
  };

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