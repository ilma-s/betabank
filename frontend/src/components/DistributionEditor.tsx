import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Trash2, Loader2 } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Checkbox } from "@/components/ui/checkbox";
import { useToast } from "@/components/ui/use-toast";
import axios, { AxiosError } from "axios";

interface DistributionEditorProps {
  personaId: number;
  personaName: string;
  initialDistribution?: Record<string, number>;
  onClose: () => void;
  token: string;
  batchId?: number;
  currentDistribution?: Record<string, number>;
  onDistributionUpdated?: () => void;
  isNewPersona?: boolean;
}

interface ApiError {
  detail: string;
}

export function DistributionEditor({
  personaId,
  personaName,
  initialDistribution,
  onClose,
  token,
  batchId,
  currentDistribution,
  onDistributionUpdated,
  isNewPersona = false,
}: DistributionEditorProps) {
  const { toast } = useToast();
  const [distribution, setDistribution] = useState<Record<string, number>>(
    currentDistribution ||
      initialDistribution || {
        Transport: 0.1,
        Shopping: 0.2,
        Groceries: 0.15,
        Utilities: 0.1,
        Dining: 0.1,
        Salary: 0.1,
        "ATM Withdrawals": 0.15,
        Subscriptions: 0.1,
      }
  );
  const [displayValues, setDisplayValues] = useState<Record<string, string>>(
    Object.entries(currentDistribution || initialDistribution || {}).reduce(
      (acc, [key, value]) => ({
        ...acc,
        [key]: (value * 100).toFixed(1),
      }),
      {}
    )
  );
  const [newCategory, setNewCategory] = useState("");
  const [newPercentage, setNewPercentage] = useState("");
  const [saveForTraining, setSaveForTraining] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [regenerateBatch, setRegenerateBatch] = useState(!!batchId);

  const total = Object.values(distribution).reduce((sum, val) => sum + val, 0);
  const isValid = Math.abs(total - 1) < 0.01; // Allow small rounding errors
  const totalPercentage = (total * 100).toFixed(1);
  const isExactly100 = totalPercentage === "100.0";

  const handlePercentageChange = (category: string, value: string) => {
    // Validate input is a valid number
    if (value !== "" && isNaN(parseFloat(value))) {
      return;
    }

    setDisplayValues((prev) => ({
      ...prev,
      [category]: value,
    }));

    if (value === "") {
      setDistribution((prev) => ({
        ...prev,
        [category]: 0,
      }));
      return;
    }

    const percentage = parseFloat(value) / 100;
    if (isNaN(percentage)) return;

    setDistribution((prev) => ({
      ...prev,
      [category]: percentage,
    }));
  };

  const handleAddCategory = () => {
    if (!newCategory || !newPercentage) return;

    const percentage = parseFloat(newPercentage) / 100;
    if (isNaN(percentage)) return;

    setDistribution((prev) => ({
      ...prev,
      [newCategory]: percentage,
    }));

    setDisplayValues((prev) => ({
      ...prev,
      [newCategory]: newPercentage,
    }));

    setNewCategory("");
    setNewPercentage("");
  };

  const handleRemoveCategory = (category: string) => {
    const newDistribution = { ...distribution };
    delete newDistribution[category];
    setDistribution(newDistribution);

    const newDisplayValues = { ...displayValues };
    delete newDisplayValues[category];
    setDisplayValues(newDisplayValues);
  };

  const handleSubmit = async () => {
    if (!isValid) {
      toast({
        title: "Error",
        description: "Distribution percentages must sum to 100%",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);
    try {
      // Convert display values (percentages) back to decimals and validate
      const cleanDistribution = Object.entries(displayValues).reduce(
        (acc, [key, value]) => {
          // Parse the percentage and convert to decimal, ensuring it's a number
          const decimal = Number((parseFloat(value) / 100).toFixed(4));
          if (isNaN(decimal)) {
            throw new Error(`Invalid number for category ${key}: ${value}`);
          }
          // Ensure it's stored as a number, not a string
          acc[key] = decimal;
          return acc;
        },
        {} as Record<string, number>
      );

      if (isNewPersona) {
        // Create new persona with distribution
        const response = await axios.post(
          `http://localhost:8000/personas`,
          {
            name: personaName,
            distribution: cleanDistribution,
            save_for_training: saveForTraining,
          },
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        // Generate initial transactions for the new persona
        const generateResponse = await axios.get(
          `http://localhost:8000/generate/${response.data.id}`,
          {
            params: {
              batch_name: `${personaName} - Initial Batch`,
              months: 3, // Default to 3 months of data
            },
            headers: {
              Authorization: `Bearer ${token}`,
            },
          }
        );

        toast({
          title: "Success",
          description: "New persona created with initial transactions",
        });

        // Redirect to the batch page
        window.location.href = `/batch/${generateResponse.data.batch_id}`;
      } else {
        // Update existing persona distribution
        const response = await axios.patch(
          `http://localhost:8000/personas/${personaId}/distribution`,
          cleanDistribution,
          {
            headers: {
              Authorization: `Bearer ${token}`,
            },
            params: {
              save_for_training: saveForTraining,
              batch_id: regenerateBatch ? batchId : undefined,
            },
          }
        );

        if (regenerateBatch && batchId && !response.data.batch_regenerated) {
          throw new Error("Batch was not regenerated");
        }

        toast({
          title: "Success",
          description:
            regenerateBatch && batchId
              ? "Distribution updated and batch regenerated successfully"
              : "Distribution updated successfully",
        });
      }

      if (onDistributionUpdated) {
        onDistributionUpdated();
      }

      onClose();
    } catch (error) {
      console.error("Error updating distribution:", error);
      const axiosError = error as AxiosError<ApiError>;
      const errorMessage =
        axiosError.response?.data?.detail ||
        (error instanceof Error
          ? error.message
          : "Failed to update distribution");

      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return isNewPersona ? (
    <div className="mt-6">
      <div className="bg-white rounded-lg p-6 border border-gray-200">
        <div className="mb-6">
          <h2 className="text-lg font-medium mb-2 text-[#261436]">
            Define Distribution for New Persona
          </h2>
          <p className="text-sm text-[#261436]/70">
            Adjust category percentages or add new categories. Total must equal
            100%.
          </p>
        </div>

        <div className="grid gap-4">
          {/* Existing Categories */}
          <div className="grid gap-2">
            {Object.entries(distribution).map(([category, percentage]) => (
              <div key={category} className="flex items-center gap-2">
                <span className="w-40 truncate text-[#261436]">{category}</span>
                <Input
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  value={displayValues[category] || ""}
                  onChange={(e) =>
                    handlePercentageChange(category, e.target.value)
                  }
                  className="w-24 bg-white border-gray-300 text-[#261436]"
                />
                <span className="text-[#261436]">%</span>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleRemoveCategory(category)}
                >
                  <Trash2 className="h-4 w-4 text-[#261436]" />
                </Button>
              </div>
            ))}
          </div>

          {/* Add New Category */}
          <div className="flex items-center gap-2 pt-4 border-t">
            <Input
              placeholder="New Category"
              value={newCategory}
              onChange={(e) => setNewCategory(e.target.value)}
              className="w-40 bg-white border-gray-300 text-[#261436] placeholder:text-[#261436]/50"
            />
            <Input
              type="number"
              min="0"
              max="100"
              step="0.1"
              placeholder="Percentage"
              value={newPercentage}
              onChange={(e) => setNewPercentage(e.target.value)}
              className="w-24 bg-white border-gray-300 text-[#261436] placeholder:text-[#261436]/50"
            />
            <span className="text-[#261436]">%</span>
            <Button
              variant="outline"
              size="icon"
              onClick={handleAddCategory}
              disabled={!newCategory || !newPercentage}
            >
              <Plus className="h-4 w-4 text-[#261436]" />
            </Button>
          </div>

          {/* Total Percentage */}
          <div className="flex items-center gap-2 pt-4 border-t">
            <span className="font-medium text-[#261436]">Total:</span>
            <span
              className={`${isExactly100 ? "text-[#261436]" : "text-red-600"}`}
            >
              {totalPercentage}%
            </span>
          </div>

          {/* Options
          <div className="flex flex-col gap-4 pt-4 border-t">
            <div className="flex items-center space-x-2">
              <Checkbox
                id="save-training"
                checked={saveForTraining}
                onCheckedChange={(checked) => setSaveForTraining(checked as boolean)}
              />
              <label
                htmlFor="save-training"
                className="text-sm font-medium leading-none text-[#261436] peer-disabled:cursor-not-allowed peer-disabled:opacity-70"
              >
                Save this distribution for training
              </label>
            </div>
          </div> */}
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <Button
            variant="outline"
            onClick={onClose}
            className="text-[#261436]"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!isValid || isSubmitting}
            className="bg-[#261436] text-white"
          >
            {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {isSubmitting ? "Creating..." : "Create Persona"}
          </Button>
        </div>
      </div>
    </div>
  ) : (
    <Dialog open={true} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Edit Distribution for {personaName}</DialogTitle>
          <DialogDescription>
            Adjust category percentages to modify transaction patterns. Total must equal 100%.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-4">
          {/* Existing Categories */}
          <div className="grid gap-2">
            {Object.entries(distribution).map(([category, percentage]) => (
              <div key={category} className="flex items-center gap-2">
                <span className="w-40 truncate text-[#261436]">{category}</span>
                <Input
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  value={displayValues[category] || ""}
                  onChange={(e) =>
                    handlePercentageChange(category, e.target.value)
                  }
                  className="w-24 bg-white border-gray-300 text-[#261436]"
                />
                <span className="text-[#261436]">%</span>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => handleRemoveCategory(category)}
                >
                  <Trash2 className="h-4 w-4 text-[#261436]" />
                </Button>
              </div>
            ))}
          </div>

          {/* Total Percentage */}
          <div className="flex items-center gap-2 pt-4 border-t">
            <span className="font-medium text-[#261436]">Total:</span>
            <span
              className={`${isExactly100 ? "text-[#261436]" : "text-red-600"}`}
            >
              {totalPercentage}%
            </span>
          </div>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={onClose}
            className="text-[#261436]"
          >
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!isValid || isSubmitting}
            className="bg-[#261436] text-white"
          >
            {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {isSubmitting ? "Updating..." : "Update Distribution"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
