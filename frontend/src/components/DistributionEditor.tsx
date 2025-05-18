import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Plus, Trash2, Loader2, Info } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useToast } from "@/components/ui/use-toast";
import axios from "axios";
import { Label } from "@/components/ui/label";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

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
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [useForTraining, setUseForTraining] = useState(true);

  const total = Object.values(distribution).reduce((sum, val) => sum + val, 0);
  const isValid = Math.abs(total - 1) < 0.01;
  const totalPercentage = (total * 100).toFixed(1);
  const isExactly100 = totalPercentage === "100.0";

  const handlePercentageChange = (category: string, value: string) => {
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

  const handleSave = async () => {
    if (!isExactly100) {
      toast({
        title: "Error",
        description: "Total percentage must equal 100%",
        variant: "destructive",
      });
      return;
    }

    setIsSubmitting(true);
    try {
      await axios.patch(
        `http://localhost:8000/personas/${personaId}/distribution`,
        {
          distribution,
          batchId,
          useForTraining,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      toast({
        title: "Success",
        description: "Distribution updated successfully",
      });

      onDistributionUpdated?.();
    } catch {
      toast({
        title: "Error",
        description: "Failed to update distribution",
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
          <div className="grid gap-2">
            {Object.entries(distribution).map(([category]) => (
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

          <div className="flex items-center gap-2 pt-4 border-t">
            <span className="font-medium text-[#261436]">Total:</span>
            <span
              className={`${isExactly100 ? "text-[#261436]" : "text-red-600"}`}
            >
              {totalPercentage}%
            </span>
          </div>

          <div className="flex items-center gap-2 pt-4">
            <input
              type="checkbox"
              id="useForTraining"
              checked={useForTraining}
              onChange={(e) => setUseForTraining(e.target.checked)}
              className="h-4 w-4 rounded border-[#261436]/20 text-[#261436] focus:ring-[#261436]"
            />
            <Label
              htmlFor="useForTraining"
              className="text-sm font-medium text-[#261436] cursor-pointer flex items-center gap-2"
            >
              Use this distribution update to improve future transaction
              generations
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Info className="h-4 w-4 text-[#261436]/70" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      When enabled, this distribution update will be used to
                      train the model, helping it generate more accurate
                      transactions in future batches. The model will learn your
                      preferred category distributions and apply them to new
                      transaction sets.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </Label>
          </div>
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
            onClick={handleSave}
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
            Adjust category percentages to modify transaction patterns. Total
            must equal 100%.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-4">
          <div className="grid gap-2">
            {Object.entries(distribution).map(([category]) => (
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

          <div className="flex items-center gap-2 pt-4 border-t">
            <span className="font-medium text-[#261436]">Total:</span>
            <span
              className={`${isExactly100 ? "text-[#261436]" : "text-red-600"}`}
            >
              {totalPercentage}%
            </span>
          </div>

          <div className="flex items-center gap-2 pt-4">
            <input
              type="checkbox"
              id="useForTraining"
              checked={useForTraining}
              onChange={(e) => setUseForTraining(e.target.checked)}
              className="h-4 w-4 rounded border-[#261436]/20 text-[#261436] focus:ring-[#261436]"
            />
            <Label
              htmlFor="useForTraining"
              className="text-sm font-medium text-[#261436] cursor-pointer flex items-center gap-2"
            >
              Use this distribution update to improve future transaction
              generations
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger>
                    <Info className="h-4 w-4 text-[#261436]/70" />
                  </TooltipTrigger>
                  <TooltipContent>
                    <p className="max-w-xs">
                      When enabled, this distribution update will be used to
                      train the model, helping it generate more accurate
                      transactions in future batches. The model will learn your
                      preferred category distributions and apply them to new
                      transaction sets.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            </Label>
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
            onClick={handleSave}
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
