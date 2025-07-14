"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { getPersonas } from "@/lib/api";
import { PersonaWithDataset } from "@/types/persona";
import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";

export function PersonaList() {
  const [personas, setPersonas] = useState<PersonaWithDataset[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const { isAuthenticated, logout } = useAuth();
  const router = useRouter();

  useEffect(() => {
    const fetchPersonas = async () => {
      if (!isAuthenticated) {
        router.push("/");
        return;
      }

      try {
        const response = await getPersonas();
        setPersonas(response.personas || []);
      } catch (err) {
        if (err instanceof Error && err.message.includes("401")) {
          logout();
          router.push("/");
        } else {
          setError(
            err instanceof Error ? err.message : "Failed to load personas"
          );
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchPersonas();
  }, [isAuthenticated, router, logout]);

  if (isLoading) {
    return (
      <div className="text-center py-8 text-[#F1E6EA]">Loading personas...</div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <div className="text-red-500 font-medium">Error: {error}</div>
        <button
          onClick={() => window.location.reload()}
          className="mt-4 text-sm text-[#F1E6EA] hover:text-[#a78bfa]"
        >
          Try Again
        </button>
      </div>
    );
  }

  if (!personas || !personas.length) {
    return (
      <section className="w-full max-w-4xl mx-auto px-4 py-12">
        <h1 className="text-3xl font-bold mb-8 text-[#F1E6EA]">Personas</h1>
        <div className="text-center py-8 text-[#a78bfa]">
          No personas found. Create your first persona to get started!
        </div>
      </section>
    );
  }

  return (
    <section className="w-full max-w-6xl mx-auto px-4 py-12">
      <h1 className="text-3xl font-bold mb-8 text-[#F1E6EA]">Personas</h1>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {personas.map((persona) => (
          <Card
            key={persona.id}
            className="bg-[#F1E6EA] border border-[#a78bfa] rounded-xl shadow-md hover:shadow-xl transition-shadow duration-200 cursor-pointer"
          >
            <CardHeader>
              <CardTitle className="text-lg font-semibold text-[#261436]">
                {persona.name}
              </CardTitle>
              <CardDescription className="text-[#261436]">
                {persona.description}
              </CardDescription>
            </CardHeader>
          </Card>
        ))}
      </div>
    </section>
  );
}
