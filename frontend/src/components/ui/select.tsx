"use client"

import React from 'react';

interface SelectProps {
  children: React.ReactNode;
  onValueChange: (value: string) => void;
  value: string;
}

export function Select({ children, onValueChange, value }: SelectProps) {
  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    onValueChange(event.target.value);
  };

  return (
    <div className="relative">
      <select value={value} onChange={handleChange} className="w-full text-left">
        {children}
      </select>
    </div>
  );
}

export function SelectTrigger({ children }: { children: React.ReactNode }) {
  return <button className="w-full text-left">{children}</button>;
}

interface SelectContentProps {
  children: React.ReactNode;
  className?: string;
}

export function SelectContent({ children, className }: SelectContentProps) {
  return <div className={className}>{children}</div>;
}

interface SelectItemProps extends React.OptionHTMLAttributes<HTMLOptionElement> {
  children: React.ReactNode;
  value: string;
  className?: string;
}

export function SelectItem({ children, value, className = '', ...props }: SelectItemProps) {
  return (
    <option value={value} className={`px-4 py-2 text-black hover:bg-gray-100 cursor-pointer ${className}`} {...props}>
      {children}
    </option>
  );
}
