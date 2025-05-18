"use client";

import { UploadJson } from '@/components/upload-json';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/context/AuthContext';
import { useState } from 'react';
import { createPersonaWithDistribution } from '@/lib/api';
import { useToast } from '@/components/ui/use-toast';

interface Category {
  name: string;
  percentage: number;
}

export default function CreatePersonaPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuth();
  const { toast } = useToast();
  const [personaName, setPersonaName] = useState('');
  const [categories, setCategories] = useState<Category[]>([
    { name: 'Transport', percentage: 10 },
    { name: 'Shopping', percentage: 20 },
    { name: 'Groceries', percentage: 15 },
    { name: 'Utilities', percentage: 10 },
    { name: 'Dining', percentage: 10 },
    { name: 'Salary', percentage: 10 },
    { name: 'ATM Withdrawals', percentage: 15 },
    { name: 'Subscriptions', percentage: 10 },
  ]);
  const [newCategory, setNewCategory] = useState('');
  const [newPercentage, setNewPercentage] = useState('');
  const [isCreating, setIsCreating] = useState(false);

  if (!isAuthenticated) {
    router.push('/');
    return null;
  }

  const handlePercentageChange = (index: number, value: string) => {
    const newValue = parseFloat(value) || 0;
    const newCategories = [...categories];
    newCategories[index].percentage = newValue;
    setCategories(newCategories);
  };

  const handleDeleteCategory = (index: number) => {
    setCategories(categories.filter((_, i) => i !== index));
  };

  const handleAddCategory = () => {
    if (newCategory && newPercentage) {
      setCategories([
        ...categories,
        { name: newCategory, percentage: parseFloat(newPercentage) || 0 }
      ]);
      setNewCategory('');
      setNewPercentage('');
    }
  };

  const handleCreatePersona = async () => {
    if (!personaName || totalPercentage !== 100) return;

    setIsCreating(true);
    try {
      const distribution = categories.reduce((acc, cat) => {
        acc[cat.name] = cat.percentage / 100;
        return acc;
      }, {} as Record<string, number>);

      await createPersonaWithDistribution(personaName, distribution);
      
      toast({
        title: "Success",
        description: "Custom persona created successfully",
      });

      router.push('/personas');
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to create persona",
        variant: "destructive",
      });
    } finally {
      setIsCreating(false);
    }
  };

  const totalPercentage = categories.reduce((sum, cat) => sum + cat.percentage, 0);

  return (
    <div className="bg-[#261436] min-h-screen">
      <div className="container mx-auto py-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-3xl font-bold text-white">Create Custom Persona</h1>
          <Link href="/personas">
            <Button variant="outline" className="bg-[#F1E6EA] text-[#261436] hover:bg-[#F1E6EA]/90">
              Back to Personas
            </Button>
          </Link>
        </div>
        
        <Tabs defaultValue="distribution" className="space-y-4">
          <TabsList className="bg-[#F1E6EA] w-full">
            <TabsTrigger 
              value="distribution" 
              className="flex-1 data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:text-[#261436]"
            >
              Create via Distribution
            </TabsTrigger>
            <TabsTrigger 
              value="json" 
              className="flex-1 data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:text-[#261436]"
            >
              Upload JSON Dataset
            </TabsTrigger>
          </TabsList>

          <TabsContent value="distribution">
            <Card className="bg-[#F1E6EA]">
              <CardContent className="pt-6">
                <div className="space-y-6">
                  <div>
                    <Label htmlFor="personaName" className="text-[#261436]">Persona Name</Label>
                    <Input
                      id="personaName"
                      value={personaName}
                      onChange={(e) => setPersonaName(e.target.value)}
                      className="bg-white text-[#261436] border-[#261436]/20"
                      placeholder="Enter persona name"
                    />
                  </div>

                  <div>
                    <h3 className="text-[#261436] font-semibold mb-4">Category Distribution</h3>
                    <div className="space-y-4">
                      {categories.map((category, index) => (
                        <div key={index} className="flex items-center gap-4">
                          <div className="flex-1">
                            <Label className="text-[#261436]">{category.name}</Label>
                          </div>
                          <div className="w-32">
                            <Input
                              type="number"
                              value={category.percentage}
                              onChange={(e) => handlePercentageChange(index, e.target.value)}
                              className="bg-white text-[#261436] border-[#261436]/20"
                              min="0"
                              max="100"
                              step="0.1"
                            />
                          </div>
                          <div className="w-8 text-[#261436]">%</div>
                          <Button
                            variant="ghost"
                            onClick={() => handleDeleteCategory(index)}
                            className="text-red-500 hover:text-red-700"
                          >
                            ×
                          </Button>
                        </div>
                      ))}

                      <div className="flex items-center gap-4 mt-4">
                        <Input
                          placeholder="New category"
                          value={newCategory}
                          onChange={(e) => setNewCategory(e.target.value)}
                          className="flex-1 bg-white text-[#261436] border-[#261436]/20"
                        />
                        <Input
                          type="number"
                          placeholder="Percentage"
                          value={newPercentage}
                          onChange={(e) => setNewPercentage(e.target.value)}
                          className="w-32 bg-white text-[#261436] border-[#261436]/20"
                          min="0"
                          max="100"
                          step="0.1"
                        />
                        <div className="w-8 text-[#261436]">%</div>
                        <Button
                          onClick={handleAddCategory}
                          className="bg-[#261436] text-white hover:bg-[#261436]/90"
                        >
                          Add
                        </Button>
                      </div>

                      <div className="flex justify-between items-center mt-6 text-[#261436] font-semibold">
                        <span>Total:</span>
                        <span className={totalPercentage === 100 ? 'text-green-600' : 'text-red-600'}>
                          {totalPercentage.toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>

                  <Button
                    className="w-full bg-[#261436] text-white hover:bg-[#261436]/90"
                    disabled={totalPercentage !== 100 || !personaName || isCreating}
                    onClick={handleCreatePersona}
                  >
                    {isCreating ? 'Creating...' : 'Create Persona'}
                  </Button>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="json">
            <Card className="bg-[#F1E6EA]">
              <CardContent className="pt-6">
                <div className="mb-6">
                  <h2 className="text-xl font-semibold mb-2 text-[#261436]">Upload Transaction Dataset</h2>
                  <p className="text-[#261436]/80">
                    Upload a JSON file containing transaction data to create a custom persona.
                    The file should contain an array of transactions with the following required fields:
                  </p>
                  <ul className="list-disc list-inside mt-2 text-[#261436]/80">
                    <li>transactionAmount (object with amount and currency)</li>
                    <li>bookingDateTime (ISO date string)</li>
                    <li>category (string)</li>
                  </ul>
                </div>
                
                <UploadJson onSuccess={() => router.push('/personas')} />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
} 