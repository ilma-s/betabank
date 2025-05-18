import { useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "./ui/textarea";
import { Alert, AlertDescription, AlertTitle } from "./ui/alert";
import { createPersonaWithDataset } from "@/lib/api";
import { useToast } from "@/components/ui/use-toast";
import { Info } from "lucide-react";
import { useAuth } from "@/context/AuthContext";

interface UploadJsonProps {
  onSuccess?: () => void;
}

interface ValidationError {
  message: string;
  line?: number;
  position?: number;
}

export function UploadJson({ onSuccess }: UploadJsonProps) {
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [error, setError] = useState<ValidationError | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [jsonPreview, setJsonPreview] = useState<string | null>(null);
  const { toast } = useToast();
  const { logout } = useAuth();
  const router = useRouter();

  const validateJsonSyntax = (jsonString: string): { isValid: boolean; error?: ValidationError } => {
    try {
      // First try to parse the JSON
      const parsed = JSON.parse(jsonString);
      
      // Check if it's properly formatted (no trailing commas, proper quotes, etc.)
      if (jsonString.includes(",]") || jsonString.includes(",}")) {
        return {
          isValid: false,
          error: { message: "Invalid JSON: Found trailing comma" }
        };
      }
      
      // Check for single quotes instead of double quotes
      if (jsonString.match(/'[^']*'/)) {
        return {
          isValid: false,
          error: { message: "Invalid JSON: Use double quotes (\") instead of single quotes (')" }
        };
      }
      
      // Check for unquoted property names - only check actual object properties
      const unquotedPropRegex = /[{,]\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:/g;
      let match;
      while ((match = unquotedPropRegex.exec(jsonString)) !== null) {
        // Skip if the property is actually quoted
        const beforeMatch = jsonString.substring(0, match.index + match[0].indexOf(match[1]));
        if (!beforeMatch.endsWith('"')) {
          return {
            isValid: false,
            error: { message: `Invalid JSON: Property names must be enclosed in double quotes: "${match[1]}"` }
          };
        }
      }

      return { isValid: true };
    } catch (e) {
      if (e instanceof Error) {
        // Try to extract line and position information from the error message
        const posMatch = e.message.match(/position (\d+)/);
        const position = posMatch ? parseInt(posMatch[1]) : undefined;
        
        // Calculate line number from position
        let line;
        if (position) {
          line = jsonString.substring(0, position).split('\n').length;
        }
        
        return {
          isValid: false,
          error: {
            message: e.message.replace(/^JSON\.parse: /, 'Invalid JSON: '),
            line,
            position
          }
        };
      }
      return {
        isValid: false,
        error: { message: "Invalid JSON format" }
      };
    }
  };

  const validateTransactionData = (data: any): { isValid: boolean; error?: ValidationError } => {
    if (!Array.isArray(data)) {
      return {
        isValid: false,
        error: { message: "Invalid format: File must contain an array of transactions" }
      };
    }

    if (data.length === 0) {
      return {
        isValid: false,
        error: { message: "Invalid dataset: File contains no transactions" }
      };
    }

    const requiredFields = [
      "transactionAmount",
      "bookingDateTime",
      "category",
      "creditorName",
      "creditorAccount",
      "debtorName",
      "debtorAccount",
      "remittanceInformationUnstructured"
    ];

    for (let i = 0; i < data.length; i++) {
      const transaction = data[i];
      
      // Check required fields
      const missingFields = requiredFields.filter(field => !(field in transaction));
      if (missingFields.length > 0) {
        return {
          isValid: false,
          error: {
            message: `Transaction ${i + 1}: Missing required fields: ${missingFields.join(", ")}`,
            line: i + 1
          }
        };
      }

      // Validate transactionAmount structure
      if (!transaction.transactionAmount?.amount || !transaction.transactionAmount?.currency) {
        return {
          isValid: false,
          error: {
            message: `Transaction ${i + 1}: transactionAmount must contain 'amount' and 'currency'`,
            line: i + 1
          }
        };
      }

      // Validate amount format
      const amount = parseFloat(transaction.transactionAmount.amount);
      if (isNaN(amount)) {
        return {
          isValid: false,
          error: {
            message: `Transaction ${i + 1}: Invalid amount format`,
            line: i + 1
          }
        };
      }

      // Validate date format
      if (isNaN(Date.parse(transaction.bookingDateTime))) {
        return {
          isValid: false,
          error: {
            message: `Transaction ${i + 1}: bookingDateTime must be a valid ISO date string`,
            line: i + 1
          }
        };
      }

      // Validate account structures
      if (!transaction.creditorAccount?.iban || !transaction.debtorAccount?.iban) {
        return {
          isValid: false,
          error: {
            message: `Transaction ${i + 1}: Both creditor and debtor accounts must contain IBAN`,
            line: i + 1
          }
        };
      }
    }

    return { isValid: true };
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    setError(null);
    setJsonPreview(null);
    
    if (!selectedFile) {
      setError({ message: "Please select a file" });
      setFile(null);
      return;
    }

    if (selectedFile.type !== "application/json") {
      setError({ message: "Invalid file type: Please select a JSON file" });
      setFile(null);
      return;
    }

    try {
      const fileContent = await selectedFile.text();
      
      // Validate JSON syntax
      const syntaxValidation = validateJsonSyntax(fileContent);
      if (!syntaxValidation.isValid) {
        setError(syntaxValidation.error!);
        setFile(null);
        return;
      }

      // Parse and validate transaction data
      const jsonData = JSON.parse(fileContent);
      const dataValidation = validateTransactionData(jsonData);
      if (!dataValidation.isValid) {
        setError(dataValidation.error!);
        setFile(null);
        return;
      }

      // If all validations pass, set the file and show a preview
      setFile(selectedFile);
      setJsonPreview(JSON.stringify(jsonData[0], null, 2)); // Show first transaction as preview
    } catch (err) {
      setError({ message: err instanceof Error ? err.message : "Failed to read file" });
      setFile(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !name) {
      setError({ message: "Please provide both a name and a JSON file" });
      return;
    }

    setIsLoading(true);
    try {
      const fileContent = await file.text();
      const jsonData = JSON.parse(fileContent);

      const result = await createPersonaWithDataset(
        name,
        description,
        jsonData
      );
      toast({
        title: "Success",
        description: "Custom persona created successfully",
      });

      // Reset form
      setFile(null);
      setName("");
      setDescription("");
      setError(null);
      setJsonPreview(null);

      // Call onSuccess callback if provided
      onSuccess?.();
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Failed to create persona";
      
      // Handle authentication errors
      if (errorMessage.includes('401')) {
        logout();
        router.push('/');
        toast({
          title: "Authentication Error",
          description: "Your session has expired. Please log in again.",
          variant: "destructive",
        });
      } else {
        setError({ message: errorMessage });
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="name" className="text-[#261436]">
          Persona Name
        </Label>
        <Input
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          placeholder="Enter persona name"
          required
          className="bg-white text-[#261436] border-[#261436]/20"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="description" className="text-[#261436]">
          Description (Optional)
        </Label>
        <Textarea
          id="description"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          placeholder="Describe your persona"
          rows={3}
          className="bg-white text-[#261436] border-[#261436]/20"
        />
      </div>

      <div className="space-y-4">
        <div className="bg-white rounded-lg p-4 border border-[#261436]/20">
          <div className="flex items-start gap-3 mb-4">
            <Info className="w-5 h-5 text-[#261436] mt-1 flex-shrink-0" />
            <div className="text-sm text-[#261436]">
              <p className="font-medium mb-2">JSON Format Requirements:</p>
              <ul className="list-disc pl-4 space-y-1">
                <li>Must be a valid JSON array of transactions</li>
                <li>Use double quotes for strings and property names</li>
                <li>No trailing commas</li>
                <li>Each transaction must include required fields</li>
                <li>Dates must be in ISO format</li>
                <li>Numbers must be valid (no leading zeros)</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <Label htmlFor="file" className="text-[#261436]">
            Transaction Dataset (JSON)
          </Label>
          <Input
            id="file"
            type="file"
            accept="application/json"
            onChange={handleFileChange}
            required
            className="bg-white text-[#261436] border-[#261436]/20"
          />
        </div>
      </div>

      {error && (
        <Alert variant="destructive" className="bg-white border-[#261436] border">
          <AlertTitle className="text-[#261436]">Validation Error</AlertTitle>
          <AlertDescription className="text-[#261436] font-medium">
            {error.message}
            {error.line && (
              <div className="mt-1 text-sm">
                Line: {error.line}
                {error.position && `, Position: ${error.position}`}
              </div>
            )}
          </AlertDescription>
        </Alert>
      )}

      {jsonPreview && (
        <div className="space-y-2">
          <Label className="text-[#261436]">Preview (First Transaction)</Label>
          <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto text-sm text-[#261436] border border-[#261436]/20">
            {jsonPreview}
          </pre>
        </div>
      )}

      <Button
        type="submit"
        disabled={isLoading || !file || !name}
        className="w-full bg-[#261436] text-white hover:bg-[#261436]/90"
      >
        {isLoading ? "Creating..." : "Create Persona"}
      </Button>
    </form>
  );
}
