import { Transaction, TransactionExplanationData } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Pencil, Trash2, FilterIcon, X, ChevronDown, ChevronUp, Info } from "lucide-react";
import { useState, useMemo } from "react";
import { useToast } from "@/components/ui/use-toast";
import axios, { AxiosError } from "axios";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "./ui/badge";
import { TransactionExplanation } from "./TransactionExplanation";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

interface TransactionListProps {
  transactions: Transaction[];
  onTransactionUpdated?: () => void;
  token: string;
}

export function TransactionList({
  transactions,
  onTransactionUpdated,
  token,
}: TransactionListProps) {
  const { toast } = useToast();
  const [isDeleting, setIsDeleting] = useState<string | null>(null);
  const [editingTransaction, setEditingTransaction] =
    useState<Transaction | null>(null);
  const [editedValues, setEditedValues] = useState({
    amount: "",
    description: "",
    category: "",
    creditorName: "",
    useForTraining: true,
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  
  const [searchMerchant, setSearchMerchant] = useState("");
  const [searchCategory, setSearchCategory] = useState("");
  const [searchDate, setSearchDate] = useState("");
  const [transactionType, setTransactionType] = useState("all");
  const [showFilters, setShowFilters] = useState(false);
  const [selectedTransactionId, setSelectedTransactionId] = useState<string | null>(null);
  const [explanation, setExplanation] = useState<TransactionExplanationData | null>(null);
  const [loadingExplanation, setLoadingExplanation] = useState(false);
  const [showExplanationDialog, setShowExplanationDialog] = useState(false);

  const categories = [
    "Shopping",
    "Groceries",
    "Transport",
    "Dining",
    "Utilities",
    "Subscriptions",
    "ATM Withdrawals",
    "Salary",
    "Refunds",
    "Crypto",
    "Gambling",
  ];

  const filteredTransactions = useMemo(() => {
    return transactions.filter(transaction => {
      if (searchMerchant && !transaction.creditorName.toLowerCase().includes(searchMerchant.toLowerCase())) {
        return false;
      }
      
      if (searchCategory && searchCategory !== 'all' && transaction.category !== searchCategory) {
        return false;
      }
      
      if (searchDate) {
        const txDate = new Date(transaction.bookingDateTime).toISOString().split('T')[0];
        if (txDate !== searchDate) {
          return false;
        }
      }

      if (transactionType !== 'all') {
        const amount = parseFloat(transaction.transactionAmount.amount);
        if (transactionType === 'income' && amount <= 0) {
          return false;
        }
        if (transactionType === 'expense' && amount >= 0) {
          return false;
        }
      }
      
      return true;
    });
  }, [transactions, searchMerchant, searchCategory, searchDate, transactionType]);

  const handleEdit = (transaction: Transaction) => {
    setEditingTransaction(transaction);
    setEditedValues({
      amount: transaction.transactionAmount.amount,
      description: transaction.remittanceInformationUnstructured,
      category: transaction.category || "",
      creditorName: transaction.creditorName,
      useForTraining: true,
    });
  };

  const handleSaveEdit = async () => {
    if (!editingTransaction) return;

    setIsSubmitting(true);
    try {
      await axios.patch(
        `http://localhost:8000/transactions/${editingTransaction.transactionId}`,
        {
          transactionAmount: {
            amount: editedValues.amount,
            currency: editingTransaction.transactionAmount.currency,
          },
          remittanceInformationUnstructured: editedValues.description,
          category: editedValues.category,
          creditorName: editedValues.creditorName,
          useForTraining: editedValues.useForTraining,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      toast({
        title: "Success",
        description: "Transaction updated successfully",
      });

      onTransactionUpdated?.();
      setEditingTransaction(null);
    } catch (err) {
      const errorMessage = err instanceof AxiosError 
        ? err.response?.data?.detail || "Failed to update transaction"
        : "Failed to update transaction";

      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleDelete = async (transactionId: string) => {
    if (!confirm("Are you sure you want to delete this transaction?")) {
      return;
    }

    setIsDeleting(transactionId);
    try {
      await axios.delete(
        `http://localhost:8000/transactions/${transactionId}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      toast({
        title: "Success",
        description: "Transaction deleted successfully",
      });
      onTransactionUpdated?.();
    } catch (err) {
      const errorMessage = err instanceof AxiosError 
        ? err.response?.data?.detail || "Failed to delete transaction"
        : "Failed to delete transaction";

      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setIsDeleting(null);
    }
  };

  const fetchExplanation = async (transactionId: string) => {
    setLoadingExplanation(true);
    try {
      const response = await axios.get<TransactionExplanationData>(
        `http://localhost:8000/transactions/${transactionId}/explanation`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      setExplanation(response.data);
      setSelectedTransactionId(transactionId);
      setShowExplanationDialog(true);
    } catch (err) {
      const errorMessage = err instanceof AxiosError 
        ? err.response?.data?.detail || "Failed to load transaction explanation"
        : "Failed to load transaction explanation";

      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoadingExplanation(false);
    }
  };

  return (
    <>
      {(searchMerchant || searchCategory !== "" || searchDate || transactionType !== "all") && (
        <div className="flex items-center justify-between mb-4 bg-white/50 rounded-lg p-2">
          <div className="flex gap-2 flex-wrap">
            {searchMerchant && (
              <Badge variant="outline" className="bg-[#F1E6EA] text-[#261436] border-[#261436]/20">
                Merchant: {searchMerchant}
                <X 
                  className="h-3 w-3 ml-1 cursor-pointer" 
                  onClick={() => setSearchMerchant("")}
                />
              </Badge>
            )}
            {searchCategory && searchCategory !== "all" && (
              <Badge variant="outline" className="bg-[#F1E6EA] text-[#261436] border-[#261436]/20">
                Category: {searchCategory}
                <X 
                  className="h-3 w-3 ml-1 cursor-pointer" 
                  onClick={() => setSearchCategory("")}
                />
              </Badge>
            )}
            {searchDate && (
              <Badge variant="outline" className="bg-[#F1E6EA] text-[#261436] border-[#261436]/20">
                Date: {new Date(searchDate).toLocaleDateString('en-GB')}
                <X 
                  className="h-3 w-3 ml-1 cursor-pointer" 
                  onClick={() => setSearchDate("")}
                />
              </Badge>
            )}
            {transactionType !== "all" && (
              <Badge variant="outline" className="bg-[#F1E6EA] text-[#261436] border-[#261436]/20">
                Type: {transactionType === 'income' ? 'Income' : 'Expenses'}
                <X 
                  className="h-3 w-3 ml-1 cursor-pointer" 
                  onClick={() => setTransactionType("all")}
                />
              </Badge>
            )}
          </div>
          <Button
            variant="secondary"
            size="sm"
            className="text-[#261436] hover:bg-[#F1E6EA]"
            onClick={() => {
              setSearchMerchant("");
              setSearchCategory("");
              setSearchDate("");
              setTransactionType("all");
            }}
          >
            Clear all
          </Button>
        </div>
      )}

      <div className="flex items-center gap-4 mb-4">
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
      </div>

      {showFilters && (
        <Card className="bg-white mb-4">
          <CardContent className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
              <div>
                <Label htmlFor="merchant-filter" className="text-[#261436] text-sm">Filter by Merchant</Label>
                <Input
                  id="merchant-filter"
                  placeholder="Enter merchant name"
                  value={searchMerchant}
                  onChange={(e) => setSearchMerchant(e.target.value)}
                  className="mt-1 text-[#261436] placeholder:text-[#261436]/50 bg-white"
                />
              </div>
              
              <div>
                <Label htmlFor="category-filter" className="text-[#261436] text-sm">Filter by Category</Label>
                <Select value={searchCategory || undefined} onValueChange={setSearchCategory}>
                  <SelectTrigger id="category-filter" className="mt-1 text-[#261436] bg-white">
                    <SelectValue placeholder="All Categories" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-[#261436]/20 shadow-md">
                    <SelectItem value="all" className="text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer">All Categories</SelectItem>
                    {categories.map((category) => (
                      <SelectItem 
                        key={category} 
                        value={category} 
                        className="text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer"
                      >
                        {category}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="date-filter" className="text-[#261436] text-sm">Filter by Date</Label>
                <Input
                  id="date-filter"
                  type="date"
                  value={searchDate}
                  onChange={(e) => setSearchDate(e.target.value)}
                  className="mt-1 text-[#261436] bg-white"
                />
              </div>

              <div>
                <Label htmlFor="type-filter" className="text-[#261436] text-sm">Filter by Transaction Type</Label>
                <Select value={transactionType} onValueChange={setTransactionType}>
                  <SelectTrigger id="type-filter" className="mt-1 text-[#261436] bg-white">
                    <SelectValue placeholder="All Transactions" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border border-[#261436]/20 shadow-md">
                    <SelectItem value="all" className="text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer">All Transactions</SelectItem>
                    <SelectItem value="income" className="text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer">Income</SelectItem>
                    <SelectItem value="expense" className="text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer">Expenses</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      <Card className="bg-white">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b bg-[#F1E6EA]">
                  <th className="text-left p-3 text-[#261436] font-semibold">
                    Date
                  </th>
                  <th className="text-left p-3 text-[#261436] font-semibold">
                    Amount
                  </th>
                  <th className="text-left p-3 text-[#261436] font-semibold">
                    Category
                  </th>
                  <th className="text-left p-3 text-[#261436] font-semibold">
                    Description
                  </th>
                  <th className="text-left p-3 text-[#261436] font-semibold">
                    To
                  </th>
                  <th className="text-right p-3 text-[#261436] font-semibold">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredTransactions.map((transaction) => (
                  <tr
                    key={transaction.transactionId}
                    className="border-b hover:bg-[#F1E6EA]/50"
                  >
                    <td className="p-3 text-[#261436] font-medium">
                      {(() => {
                        const date = new Date(transaction.bookingDateTime);
                        if (isNaN(date.getTime())) {
                          return "—";
                        }
                        return date.toLocaleDateString('en-GB');
                      })()}
                    </td>
                    <td className="p-3 text-[#261436] font-medium">
                      {parseFloat(transaction.transactionAmount.amount).toFixed(
                        2
                      )}{" "}
                      {transaction.transactionAmount.currency}
                    </td>
                    <td className="p-3 text-[#261436] font-medium">
                      {transaction.category}
                    </td>
                    <td className="p-3 text-[#261436] font-medium">
                      {transaction.remittanceInformationUnstructured}
                    </td>
                    <td className="p-3 text-[#261436] font-medium">
                      {transaction.creditorName}
                    </td>
                    <td className="p-3 text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="secondary"
                          size="icon"
                          onClick={() => fetchExplanation(transaction.transactionId)}
                          disabled={loadingExplanation}
                          className="bg-[#F1E6EA] hover:bg-[#F1E6EA]/80 text-[#261436]"
                        >
                          <Info className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="secondary"
                          size="icon"
                          onClick={() => handleEdit(transaction)}
                          className="bg-[#F1E6EA] hover:bg-[#F1E6EA]/80 text-[#261436]"
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="secondary"
                          size="icon"
                          onClick={() => handleDelete(transaction.transactionId)}
                          disabled={isDeleting === transaction.transactionId}
                          className="bg-[#F1E6EA] hover:bg-[#F1E6EA]/80 text-[#261436]"
                        >
                          {isDeleting === transaction.transactionId ? (
                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-[#261436] border-t-transparent" />
                          ) : (
                            <Trash2 className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      <Dialog
        open={!!editingTransaction}
        onOpenChange={() => setEditingTransaction(null)}
      >
        <DialogContent className="sm:max-w-[425px] bg-white p-6">
          <DialogHeader className="mb-4">
            <DialogTitle className="text-xl font-semibold text-[#261436]">
              Edit Transaction
            </DialogTitle>
            <DialogDescription className="text-[#261436]/70">
              Make changes to the transaction details below.
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-6 py-4">
            <div className="grid gap-2">
              <Label
                htmlFor="amount"
                className="text-sm font-medium text-[#261436]"
              >
                Amount
              </Label>
              <Input
                id="amount"
                type="number"
                step="0.01"
                value={editedValues.amount}
                onChange={(e) =>
                  setEditedValues((prev) => ({
                    ...prev,
                    amount: e.target.value,
                  }))
                }
                className="w-full px-3 py-2 bg-white border border-[#261436]/20 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
              />
            </div>
            <div className="grid gap-2">
              <Label
                htmlFor="category"
                className="text-sm font-medium text-[#261436]"
              >
                Category
              </Label>
              <Select
                value={editedValues.category || undefined}
                onValueChange={(value) => {
                  console.log("Selected category:", value);
                  setEditedValues((prev) => {
                    const newValues = { ...prev, category: value };
                    console.log("New edited values:", newValues);
                    return newValues;
                  });
                }}
              >
                <SelectTrigger
                  id="category"
                  className="w-full px-3 py-2 bg-white border border-[#261436]/20 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
                >
                  <SelectValue placeholder="Select a category" className="text-[#261436]" />
                </SelectTrigger>
                <SelectContent 
                  className="bg-white border border-[#261436]/20 rounded-md shadow-md"
                  position="popper"
                  sideOffset={5}
                >
                  {categories.map((category) => (
                    <SelectItem
                      key={category}
                      value={category}
                      className="px-3 py-2 text-[#261436] bg-white hover:bg-[#F1E6EA] cursor-pointer focus:bg-[#F1E6EA] focus:text-[#261436] outline-none"
                    >
                      {category}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label
                htmlFor="description"
                className="text-sm font-medium text-[#261436]"
              >
                Description
              </Label>
              <Input
                id="description"
                value={editedValues.description}
                onChange={(e) =>
                  setEditedValues((prev) => ({
                    ...prev,
                    description: e.target.value,
                  }))
                }
                className="w-full px-3 py-2 bg-white border border-[#261436]/20 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
              />
            </div>
            <div className="grid gap-2">
              <Label
                htmlFor="creditorName"
                className="text-sm font-medium text-[#261436]"
              >
                To
              </Label>
              <Input
                id="creditorName"
                value={editedValues.creditorName}
                onChange={(e) =>
                  setEditedValues((prev) => ({
                    ...prev,
                    creditorName: e.target.value,
                  }))
                }
                className="w-full px-3 py-2 bg-white border border-[#261436]/20 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
              />
            </div>
            {/* <div className="flex items-center gap-2">
              <input
                type="checkbox"
                id="useForTraining"
                checked={editedValues.useForTraining}
                onChange={(e) =>
                  setEditedValues((prev) => ({
                    ...prev,
                    useForTraining: e.target.checked,
                  }))
                }
                className="h-4 w-4 rounded border-[#261436]/20 text-[#261436] focus:ring-[#261436]"
              />
              <Label
                htmlFor="useForTraining"
                className="text-sm font-medium text-[#261436] cursor-pointer flex items-center gap-2"
              >
                Use this update to improve future transaction generations
                <TooltipProvider>
                  <Tooltip>
                    <TooltipTrigger>
                      <Info className="h-4 w-4 text-[#261436]/70" />
                    </TooltipTrigger>
                    <TooltipContent>
                      <p className="max-w-xs">
                        When enabled, this transaction update will be used to train the model,
                        helping it generate more accurate transactions in future batches.
                        The model will learn from your edits to improve transaction amounts,
                        descriptions, and categorizations.
                      </p>
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              </Label>
            </div> */}
          </div>
          <DialogFooter className="gap-2 mt-6">
            <Button
              variant="outline"
              onClick={() => setEditingTransaction(null)}
              className="px-4 py-2 text-[#261436] border border-[#261436]/20 hover:bg-[#F1E6EA]"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSaveEdit}
              disabled={isSubmitting}
              className="px-4 py-2 bg-[#261436] text-white hover:bg-[#261436]/80 focus:ring-2 focus:ring-[#261436] focus:ring-offset-2"
            >
              {isSubmitting ? (
                <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
              ) : (
                "Save changes"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={showExplanationDialog} onOpenChange={setShowExplanationDialog}>
        <DialogContent className="sm:max-w-[600px] bg-white">
          <DialogHeader>
            <DialogTitle>Transaction Explanation</DialogTitle>
            <DialogDescription>
              Understanding the patterns and factors behind this transaction
            </DialogDescription>
          </DialogHeader>
          {explanation && (
            <TransactionExplanation
              transactionId={selectedTransactionId!}
              feature_importance={explanation.feature_importance}
              applied_patterns={explanation.applied_patterns}
              explanation_text={explanation.explanation_text}
              confidence_score={explanation.confidence_score}
              meta_info={explanation.meta_info}
            />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
