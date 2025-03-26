"use client";

import React, { useState } from 'react';
import { useAuth } from '@/context/AuthContext';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login } = useAuth();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!username || !password) {
      setError('Please enter both username and password');
      return;
    }
    
    const success = login(username, password);
    
    if (!success) {
      setError('Invalid credentials. Use admin/admin to login.');
    }
  };

  return (
    <div className="bg-[#261436] min-h-screen flex items-center justify-center">
      <Card className="bg-[#F1E6EA] w-[400px]">
        <CardHeader>
          <CardTitle className="text-[#261436]">Login</CardTitle>
          <CardDescription className="text-[#261436]">
            Enter your credentials to access the Transaction Generator
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="username" className="text-[#261436]">Username</label>
              <Input
                id="username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="bg-white border-gray-300"
                placeholder="admin"
              />
            </div>
            <div className="space-y-2">
              <label htmlFor="password" className="text-[#261436]">Password</label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-white border-gray-300"
                placeholder="admin"
              />
            </div>
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <Button type="submit" className="w-full bg-[#261436] text-white">
              Login
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
} 