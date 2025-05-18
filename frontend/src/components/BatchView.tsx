import { useState, useMemo } from 'react';
import { Transaction } from "@/types";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BatchAnalytics } from "./BatchAnalytics";
import { TransactionList } from "./TransactionList";
import { Button } from "@/components/ui/button";
import { Download, Settings } from "lucide-react";
import { DistributionEditor } from "./DistributionEditor";

interface BatchViewProps {
  transactions: Transaction[];
  batchName: string;
  createdAt: string;
  months: number;
  onExport: (format: 'json' | 'csv' | 'excel') => void;
  onTransactionUpdated?: () => void;
  personaId: number;
  personaName: string;
  batchId: number;
  token: string;
}

export function BatchView({ 
  transactions, 
  batchName, 
  createdAt, 
  months, 
  onExport,
  onTransactionUpdated,
  personaId,
  personaName,
  batchId,
  token
}: BatchViewProps) {
  const [showDistributionEditor, setShowDistributionEditor] = useState(false);

  // Calculate current distribution from transactions
  const currentDistribution = useMemo(() => {
    const categoryCount: Record<string, number> = {};
    const totalTransactions = transactions.length;

    // Count transactions per category
    transactions.forEach((tx) => {
      categoryCount[tx.category] = (categoryCount[tx.category] || 0) + 1;
    });

    // Convert counts to percentages
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
            onClick={() => onExport('json')} 
            variant="outline"
            className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
          >
            <Download className="mr-2 h-4 w-4" />
            Download as JSON
          </Button>
          <Button 
            onClick={() => onExport('csv')} 
            variant="outline"
            className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
          >
            <Download className="mr-2 h-4 w-4" />
            Download as CSV
          </Button>
          <Button 
            onClick={() => onExport('excel')} 
            variant="outline"
            className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
          >
            <Download className="mr-2 h-4 w-4" />
            Download as Excel
          </Button>
        </div>
        <Button
          onClick={() => setShowDistributionEditor(true)}
          variant="outline"
          className="bg-white text-[#261436] border-[#261436] hover:bg-[#F1E6EA]"
        >
          <Settings className="mr-2 h-4 w-4" />
          Edit Distribution
        </Button>
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
        </TabsList>
        <TabsContent value="transactions" className="mt-0">
          <TransactionList
            transactions={transactions}
            onTransactionUpdated={onTransactionUpdated}
            token={token}
          />
        </TabsContent>
        <TabsContent value="analytics" className="mt-4 bg-white rounded-md p-4 text-[#261436]">
          <BatchAnalytics 
            transactions={transactions}
          />
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
            onTransactionUpdated?.();
          }}
        />
      )}
    </div>
  );
} 