import { PersonaWithDataset } from '@/types/persona';
import { Transaction } from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const getToken = () => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('authToken');
    if (!token) {
      throw new Error('401: No authentication token found');
    }
    return token;
  }
  return '';
};

const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const error = await response.json();
    if (response.status === 401) {
      throw new Error('401: ' + (error.detail || 'Unauthorized'));
    }
    throw new Error(error.detail || 'API request failed');
  }
  return response.json();
};

export const getPersonas = async (): Promise<{ personas: PersonaWithDataset[] }> => {
  const response = await fetch(`${API_BASE_URL}/personas`, {
    headers: {
      Authorization: `Bearer ${getToken()}`,
    },
  });

  return handleResponse(response);
};

export const createPersonaWithDistribution = async (
  name: string,
  distribution: Record<string, number>
): Promise<{ id: number; message: string }> => {
  const response = await fetch(`${API_BASE_URL}/create-persona`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify({
      name,
      description: `Custom persona with defined distribution`,
      distribution,
    }),
  });

  return handleResponse(response);
};

export const createPersonaWithDataset = async (
  name: string,
  description: string,
  dataset: Transaction[]
): Promise<{ id: number; message: string }> => {
  const response = await fetch(`${API_BASE_URL}/create-persona`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${getToken()}`,
    },
    body: JSON.stringify({
      name,
      description,
      dataset,
    }),
  });

  return handleResponse(response);
}; 