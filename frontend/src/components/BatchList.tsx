"use client";

import { useState } from "react";
import { Trash2, Settings } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import { Button } from "@/components/ui/button";
import { DistributionEditor } from "./DistributionEditor";

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
    <div className="space-y-4">
      {batches.map((batch) => (
        <div
          key={batch.id}
          onClick={() => onBatchClick(batch.id)}
          className="p-4 bg-white rounded-lg shadow hover:shadow-md transition-shadow cursor-pointer relative group"
        >
          <div className="flex justify-between items-start">
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-semibold text-lg">{batch.name}</h3>
                <div className="flex gap-1">
                  <button
                    onClick={(e) => handleDelete(e, batch.id)}
                    disabled={isDeleting === batch.id}
                    className={`p-1.5 rounded-full transition-opacity ${
                      isDeleting === batch.id
                        ? "opacity-50"
                        : "opacity-0 group-hover:opacity-100"
                    } hover:bg-red-100`}
                    title="Delete batch"
                  >
                    <Trash2 className="w-4 h-4 text-red-600" />
                  </button>
                </div>
              </div>
              <p className="text-sm text-gray-600">{batch.persona_name}</p>
              <p className="text-sm text-gray-500">
                {new Date(batch.created_at).toLocaleDateString('en-GB')} •{" "}
                {batch.transaction_count} transactions
              </p>
            </div>
          </div>
        </div>
      ))}
      {batches.length === 0 && (
        <div className="text-center py-8 text-gray-500">No batches found</div>
      )}
    </div>
  );
}
