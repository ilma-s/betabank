"use client";

import { useState, useMemo } from "react";
import { Trash2, Settings, Search, FilterIcon, X, ChevronDown, ChevronUp } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import { DistributionEditor } from "./DistributionEditor";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "./ui/badge";

interface Batch {
  id: number;
  name: string;
  persona_name: string;
  persona_id: number;
  created_at: string;
  transaction_count: number;
}

interface BatchListProps {
  batches: Batch[];
  onBatchClick: (batchId: number) => void;
  onBatchDeleted: () => void;
  onBatchUpdated: () => void;
  token: string;
}

export function BatchList({
  batches,
  onBatchClick,
  onBatchDeleted,
  onBatchUpdated,
  token,
}: BatchListProps) {
  const [isDeleting, setIsDeleting] = useState<number | null>(null);
  const { toast } = useToast();
  const [showFilters, setShowFilters] = useState(false);
  const [searchName, setSearchName] = useState("");
  const [selectedPersona, setSelectedPersona] = useState("all");
  const [selectedDate, setSelectedDate] = useState("");

  // Get unique persona names for the filter dropdown
  const personaOptions = useMemo(() => {
    const uniquePersonas = Array.from(new Set(batches.map(batch => batch.persona_name)));
    return uniquePersonas.sort();
  }, [batches]);

  // Filter batches based on search criteria
  const filteredBatches = useMemo(() => {
    return batches.filter(batch => {
      // Name filter
      if (searchName && !batch.name.toLowerCase().includes(searchName.toLowerCase())) {
        return false;
      }
      
      // Persona filter
      if (selectedPersona !== "all" && batch.persona_name !== selectedPersona) {
        return false;
      }
      
      // Date filter
      if (selectedDate) {
        const batchDate = new Date(batch.created_at).toISOString().split('T')[0];
        const selectedDateObj = new Date(selectedDate);
        const batchDateObj = new Date(batch.created_at);
        
        // Compare only the date part (ignore time)
        if (
          batchDateObj.getFullYear() !== selectedDateObj.getFullYear() ||
          batchDateObj.getMonth() !== selectedDateObj.getMonth() ||
          batchDateObj.getDate() !== selectedDateObj.getDate()
        ) {
          return false;
        }
      }
      
      return true;
    });
  }, [batches, searchName, selectedPersona, selectedDate]);

  // Format date for display
  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      if (isNaN(date.getTime())) {
        return "Invalid Date";
      }
      return date.toLocaleDateString('en-GB', {
        day: '2-digit',
        month: '2-digit',
        year: 'numeric'
      });
    } catch (error) {
      console.error("Error formatting date:", error);
      return "Invalid Date";
    }
  };

  const handleDelete = async (e: React.MouseEvent, batchId: number) => {
    e.stopPropagation(); // Prevent batch click when clicking delete

    if (isDeleting) return; // Prevent multiple deletes

    if (!confirm("Are you sure you want to delete this batch?")) {
      return;
    }

    setIsDeleting(batchId);

    try {
      const response = await fetch(`/api/batches/${batchId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) {
        throw new Error("Failed to delete batch");
      }

      toast({
        title: "Success",
        description: "Batch deleted successfully",
      });

      onBatchDeleted(); // Refresh the batches list
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete batch",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(null);
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 mb-2">
        <Button
          variant={!showFilters ? "outline" : "secondary"}
          className="flex items-center gap-2 text-[#261436] border-[#261436]/20 hover:bg-[#F1E6EA]"
          onClick={() => setShowFilters(!showFilters)}
        >
          <FilterIcon className="h-4 w-4" />
          <span className="font-semibold">Filters</span>
          {showFilters ? (
            <ChevronUp className="h-4 w-4" />
          ) : (
            <ChevronDown className="h-4 w-4" />
          )}
        </Button>

        {(searchName || selectedPersona !== "all" || selectedDate) && (
          <div className="flex gap-1 items-center flex-wrap flex-1">
            {searchName && (
              <Badge variant="outline" className="bg-[#F1E6EA] text-[#261436] border-[#261436]/20 text-xs">
                {searchName}
                <X 
                  className="h-3 w-3 ml-1 cursor-pointer" 
                  onClick={() => setSearchName("")}
                />
              </Badge>
            )}
            {selectedPersona !== "all" && (
              <Badge variant="outline" className="bg-[#F1E6EA] text-[#261436] border-[#261436]/20 text-xs">
                {selectedPersona}
                <X 
                  className="h-3 w-3 ml-1 cursor-pointer" 
                  onClick={() => setSelectedPersona("all")}
                />
              </Badge>
            )}
            {selectedDate && (
              <Badge variant="outline" className="bg-[#F1E6EA] text-[#261436] border-[#261436]/20 text-xs">
                {new Date(selectedDate).toLocaleDateString('en-GB')}
                <X 
                  className="h-3 w-3 ml-1 cursor-pointer" 
                  onClick={() => setSelectedDate("")}
                />
              </Badge>
            )}
            <Button
              variant="secondary"
              size="sm"
              className="text-[#261436] hover:bg-[#F1E6EA] h-6 px-2 text-xs"
              onClick={() => {
                setSearchName("");
                setSelectedPersona("all");
                setSelectedDate("");
              }}
            >
              Clear
            </Button>
          </div>
        )}
      </div>

      {showFilters && (
        <div className="space-y-2 p-2 bg-white/50 rounded-md mb-2">
          <div>
            <Label htmlFor="name-filter" className="text-[#261436] text-xs">Name</Label>
            <Input
              id="name-filter"
              placeholder="Search batches"
              value={searchName}
              onChange={(e) => setSearchName(e.target.value)}
              className="mt-1 text-[#261436] placeholder:text-[#261436]/50 bg-white h-8 text-sm"
            />
          </div>
          
          <div>
            <Label htmlFor="persona-filter" className="text-[#261436] text-xs">Persona</Label>
            <Select value={selectedPersona} onValueChange={setSelectedPersona}>
              <SelectTrigger id="persona-filter" className="mt-1 text-[#261436] bg-white h-8 text-sm">
                <SelectValue placeholder="All Personas" />
              </SelectTrigger>
              <SelectContent className="bg-white border border-[#261436]/20 shadow-md">
                <SelectItem value="all" className="text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer">
                  All Personas
                </SelectItem>
                {personaOptions.map((persona) => (
                  <SelectItem 
                    key={persona} 
                    value={persona}
                    className="text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer"
                  >
                    {persona}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          
          <div>
            <Label htmlFor="date-filter" className="text-[#261436] text-xs">Date</Label>
            <Input
              id="date-filter"
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              className="mt-1 text-[#261436] bg-white h-8 text-sm"
            />
          </div>
        </div>
      )}

      <div className="space-y-2">
        {filteredBatches.map((batch) => (
          <div
            key={batch.id}
            onClick={() => onBatchClick(batch.id)}
            className="p-3 rounded border cursor-pointer group bg-white text-[#261436] hover:bg-gray-100"
          >
            <div className="flex flex-col h-full">
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="font-medium">{batch.name}</h3>
                  <p className="text-sm opacity-80">{batch.persona_name}</p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={(e) => handleDelete(e, batch.id)}
                    disabled={isDeleting === batch.id}
                    className="p-1.5 rounded-full hover:bg-red-100 text-red-600 hover:text-red-800"
                    title="Delete batch"
                  >
                    {isDeleting === batch.id ? (
                      <span className="text-sm text-gray-500">Deleting...</span>
                    ) : (
                      <Trash2 className="w-4 h-4" />
                    )}
                  </button>
                </div>
              </div>
              <div className="flex justify-between text-xs mt-2 opacity-80">
                <span>{formatDate(batch.created_at)}</span>
                <span>{batch.transaction_count} transactions</span>
              </div>
            </div>
          </div>
        ))}
        {filteredBatches.length === 0 && (
          <div className="text-center py-4 text-gray-500 text-sm">
            {batches.length === 0 ? "No batches found" : "No batches match the filter criteria"}
          </div>
        )}
      </div>
    </div>
  );
}
