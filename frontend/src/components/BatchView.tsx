import { useState, useMemo, useEffect, useCallback } from "react";
import { Transaction, BatchExplanationData } from "@/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BatchAnalytics } from "./BatchAnalytics";
import { TransactionList } from "./TransactionList";
import { Button } from "@/components/ui/button";
import { Download, Settings, Loader2 } from "lucide-react";
import { DistributionEditor } from "./DistributionEditor";
import axios from "axios";
import { useToast } from "@/components/ui/use-toast";
import { BatchExplanation } from "./BatchExplanation";
import { Skeleton } from "./ui/skeleton";
import { BatchEvaluation } from "./BatchEvaluation";

interface BatchViewProps {
  transactions: Transaction[];
  batchName: string;
  createdAt: string;
  months: number;
  onExport: (format: "json" | "csv" | "excel") => void;
  onTransactionUpdated?: () => void;
  personaId: number;
  personaName: string;
  batchId: number;
  token: string;
}

export function BatchView({
  transactions,
  onExport,
  onTransactionUpdated,
  personaId,
  personaName,
  batchId,
  token,
}: BatchViewProps) {
  const { toast } = useToast();
  const [showDistributionEditor, setShowDistributionEditor] = useState(false);
  const [batchExplanation, setBatchExplanation] =
    useState<BatchExplanationData | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);

  const fetchBatchExplanation = useCallback(async () => {
    setLoadingExplanation(true);
    try {
      const response = await axios.get(
        `http://localhost:8000/batches/${batchId}/explanation`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      setBatchExplanation(response.data);
    } catch {
      toast({
        title: "Error",
        description: "Failed to load batch explanation",
        variant: "destructive",
      });
    } finally {
      setLoadingExplanation(false);
    }
  }, [batchId, token, toast]);

  useEffect(() => {
    fetchBatchExplanation();
  }, [batchId, fetchBatchExplanation]);

  const currentDistribution = useMemo(() => {
    const categoryCount: Record<string, number> = {};
    const totalTransactions = transactions.length;

    transactions.forEach((tx) => {
      categoryCount[tx.category] = (categoryCount[tx.category] || 0) + 1;
    });

    return Object.entries(categoryCount).reduce((acc, [category, count]) => {
      acc[category] = count / totalTransactions;
      return acc;
    }, {} as Record<string, number>);
  }, [transactions]);

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div className="flex gap-2">
          <Button
            onClick={() => onExport("json")}
            variant="outline"
            className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
          >
            <Download className="mr-2 h-4 w-4" />
            Download as JSON
          </Button>
          <Button
            onClick={() => onExport("csv")}
            variant="outline"
            className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
          >
            <Download className="mr-2 h-4 w-4" />
            Download as CSV
          </Button>
          <Button
            onClick={() => onExport("excel")}
            variant="outline"
            className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
          >
            <Download className="mr-2 h-4 w-4" />
            Download as Excel
          </Button>
        </div>
        <div className="flex gap-2">
          <Button
            onClick={() => setShowDistributionEditor(true)}
            variant="outline"
            className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
          >
            <Settings className="mr-2 h-4 w-4" />
            Edit Distribution
          </Button>
        </div>
      </div>

      <Tabs defaultValue="transactions" className="w-full">
        <TabsList className="bg-[#F1E6EA] mb-4">
          <TabsTrigger
            value="transactions"
            className="data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:text-[#261436]"
          >
            Transactions
          </TabsTrigger>
          <TabsTrigger
            value="analytics"
            className="data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:text-[#261436]"
          >
            Analytics
          </TabsTrigger>
          <TabsTrigger
            value="batchanalysis"
            className="data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:text-[#261436]"
          >
            Batch Analysis
          </TabsTrigger>
        </TabsList>
        <TabsContent value="transactions" className="mt-0">
          <TransactionList
            transactions={transactions}
            onTransactionUpdated={onTransactionUpdated}
            token={token}
          />
        </TabsContent>
        <TabsContent
          value="analytics"
          className="mt-4 bg-white rounded-md p-4 text-[#261436]"
        >
          <BatchAnalytics transactions={transactions} />
        </TabsContent>
        <TabsContent
          value="batchanalysis"
          className="mt-4 bg-white rounded-md p-4 text-[#261436]"
        >
          {loadingExplanation ? (
            <div className="space-y-4">
              <Skeleton className="h-8 w-3/4" />
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-24 w-full" />
            </div>
          ) : batchExplanation ? (
            <BatchExplanation {...batchExplanation} token={token} />
          ) : (
            <div className="text-center py-8">
              <p className="text-muted-foreground mb-4">
                No analysis data available
              </p>
              <Button
                onClick={async () => {
                  setLoadingExplanation(true);
                  try {
                    await axios.post(
                      `http://localhost:8000/batches/${batchId}/generate-explanation`,
                      {},
                      {
                        headers: {
                          Authorization: `Bearer ${token}`,
                        },
                      }
                    );
                    // Refresh the explanation data
                    const response = await axios.get(
                      `http://localhost:8000/batches/${batchId}/explanation`,
                      {
                        headers: {
                          Authorization: `Bearer ${token}`,
                        },
                      }
                    );
                    setBatchExplanation(response.data);
                  } catch {
                    toast({
                      title: "Error",
                      description:
                        "Failed to generate explanation. Please try again.",
                      variant: "destructive",
                    });
                  } finally {
                    setLoadingExplanation(false);
                  }
                }}
              >
                Generate Analysis
              </Button>
            </div>
          )}
        </TabsContent>
      </Tabs>

      {showDistributionEditor && (
        <DistributionEditor
          personaId={personaId}
          personaName={personaName}
          batchId={batchId}
          token={token}
          currentDistribution={currentDistribution}
          onClose={() => setShowDistributionEditor(false)}
          onDistributionUpdated={() => {
            setShowDistributionEditor(false);
            setIsRegenerating(true);
            // Add a small delay to show the regeneration is happening
            setTimeout(() => {
              onTransactionUpdated?.();
              setIsRegenerating(false);
            }, 1000);
          }}
        />
      )}

      {isRegenerating && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 flex flex-col items-center gap-4">
            <Loader2 className="h-8 w-8 animate-spin text-[#261436]" />
            <p className="text-lg font-medium text-[#261436]">Regenerating Transactions...</p>
            <p className="text-sm text-gray-600">Please wait while we update the transaction distribution</p>
          </div>
        </div>
      )}
    </div>
  );
}
