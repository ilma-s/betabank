"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { PersonaList } from "@/components/persona-list";
import { useAuth } from "@/context/AuthContext";

export default function PersonasPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    router.push("/");
    return null;
  }

  return (
    <div className="bg-[#261436] container mx-auto py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-3xl font-bold">Personas</h1>
        <Link href="/personas/create">
          <Button>Create Custom Persona</Button>
        </Link>
      </div>

      <div className="bg-[#261436] container mx-auto py-8">
        <PersonaList />
      </div>
    </div>
  );
}
