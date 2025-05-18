'use client'

import { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/card';
import { getPersonas } from '@/lib/api';
import { PersonaWithDataset } from '@/types/persona';
import { useAuth } from '@/context/AuthContext';
import { useRouter } from 'next/navigation';

export function PersonaList() {
  const [personas, setPersonas] = useState<PersonaWithDataset[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    const fetchPersonas = async () => {
      if (!isAuthenticated) {
        router.push('/');
        return;
      }

      try {
        const response = await getPersonas();
        setPersonas(response.personas);
      } catch (err) {
        if (err instanceof Error && err.message.includes('401')) {
          logout();
          router.push('/');
        } else {
          setError(err instanceof Error ? err.message : 'Failed to load personas');
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchPersonas();
  }, [isAuthenticated, router, logout]);

  if (isLoading) {
    return <div className="text-center py-8">Loading personas...</div>;
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <div className="text-red-500 font-medium">Error: {error}</div>
        <button 
          onClick={() => window.location.reload()} 
          className="mt-4 text-sm text-gray-600 hover:text-gray-800"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!personas.length) {
    return (
      <div className="text-center py-8 text-gray-600">
        No personas found. Create your first persona to get started!
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {personas.map((persona) => (
        <Card key={persona.id}>
          <CardHeader>
            <CardTitle>{persona.name}</CardTitle>
            <CardDescription>{persona.description}</CardDescription>
          </CardHeader>
          <CardContent>
          </CardContent>
        </Card>
      ))}
    </div>
  );
} 