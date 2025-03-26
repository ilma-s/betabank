"use client"

import React, { useState, useRef, useEffect } from 'react';

interface SelectProps {
  children: React.ReactNode;
  onValueChange: (value: string) => void;
  value: string;
}

export function Select({ children, onValueChange, value }: SelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const selectRef = useRef<HTMLDivElement>(null);
  
  // Find trigger and content among children
  const childrenArray = React.Children.toArray(children);
  const triggerChild = childrenArray.find(
    child => React.isValidElement(child) && child.type === SelectTrigger
  );
  const contentChild = childrenArray.find(
    child => React.isValidElement(child) && child.type === SelectContent
  );
  
  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (selectRef.current && !selectRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Extract SelectItems from content for creating hidden native select
  const selectItems: React.ReactElement[] = [];
  if (React.isValidElement(contentChild)) {
    React.Children.forEach(contentChild.props.children, child => {
      if (React.isValidElement(child) && child.type === SelectItem) {
        selectItems.push(child);
      }
    });
  }

  const handleTriggerClick = () => {
    setIsOpen(!isOpen);
  };

  const handleItemClick = (itemValue: string) => {
    onValueChange(itemValue);
    setIsOpen(false);
  };

  return (
    <div className="relative" ref={selectRef}>
      {/* Hidden native select for form submission */}
      <select 
        value={value} 
        onChange={(e) => onValueChange(e.target.value)} 
        className="sr-only"
        aria-hidden="true"
      >
        <option value="" disabled={value !== ""}>Select a value</option>
        {selectItems.map((item, index) => (
          <option key={index} value={item.props.value}>
            {item.props.children}
          </option>
        ))}
      </select>
      
      {/* Custom trigger button */}
      <div onClick={handleTriggerClick}>
        {triggerChild}
      </div>
      
      {/* Dropdown content */}
      {isOpen && (
        <div className="absolute mt-1 w-full z-10">
          {React.isValidElement(contentChild) && React.cloneElement(contentChild, {
            children: React.Children.map(contentChild.props.children, child => {
              if (React.isValidElement(child) && child.type === SelectItem) {
                return React.cloneElement(child, {
                  onClick: () => handleItemClick(child.props.value),
                  ...child.props
                });
              }
              return child;
            })
          })}
        </div>
      )}
    </div>
  );
}

export function SelectTrigger({ children, className }: { children: React.ReactNode, className?: string }) {
  return <div className={`cursor-pointer ${className || ''}`}>{children}</div>;
}

interface SelectContentProps {
  children: React.ReactNode;
  className?: string;
}

export function SelectContent({ children, className }: SelectContentProps) {
  return <div className={className}>{children}</div>;
}

interface SelectItemProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  value: string;
  className?: string;
}

export function SelectItem({ children, value, className = '', onClick, ...props }: SelectItemProps) {
  return (
    <div 
      role="option" 
      data-value={value}
      className={`cursor-pointer ${className}`}
      onClick={onClick}
      {...props}
    >
      {children}
    </div>
  );
}

interface SelectValueProps {
  placeholder?: string;
  className?: string;
}

export function SelectValue({ placeholder, className }: SelectValueProps) {
  return <span className={`block ${className || ''}`}>{placeholder}</span>;
}
