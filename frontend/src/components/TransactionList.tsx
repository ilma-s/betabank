import { Transaction } from "@/types";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Pencil, Trash2 } from "lucide-react";
import { useState } from "react";
import { useToast } from "@/components/ui/use-toast";
import axios from "axios";
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
  });
  const [isSubmitting, setIsSubmitting] = useState(false);

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

  const handleEdit = (transaction: Transaction) => {
    setEditingTransaction(transaction);
    setEditedValues({
      amount: transaction.transactionAmount.amount,
      description: transaction.remittanceInformationUnstructured,
      category: transaction.category || "",
      creditorName: transaction.creditorName,
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
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to update transaction",
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
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete transaction",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(null);
    }
  };

  return (
    <>
      <Card className="bg-white">
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b bg-[#F1E6EA]">
                  <th className="text-left p-3 text-[#261436] font-medium">
                    Date
                  </th>
                  <th className="text-left p-3 text-[#261436] font-medium">
                    Amount
                  </th>
                  <th className="text-left p-3 text-[#261436] font-medium">
                    Category
                  </th>
                  <th className="text-left p-3 text-[#261436] font-medium">
                    Description
                  </th>
                  <th className="text-left p-3 text-[#261436] font-medium">
                    To
                  </th>
                  <th className="text-right p-3 text-[#261436] font-medium">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((transaction) => (
                  <tr
                    key={transaction.transactionId}
                    className="border-b hover:bg-gray-50"
                  >
                    <td className="p-3 text-[#261436]">
                      {new Date(
                        transaction.bookingDateTime
                      ).toLocaleDateString('en-GB')}
                    </td>
                    <td className="p-3 text-[#261436]">
                      {parseFloat(transaction.transactionAmount.amount).toFixed(
                        2
                      )}{" "}
                      {transaction.transactionAmount.currency}
                    </td>
                    <td className="p-3 text-[#261436]">
                      {transaction.category}
                    </td>
                    <td className="p-3 text-[#261436]">
                      {transaction.remittanceInformationUnstructured}
                    </td>
                    <td className="p-3 text-[#261436]">
                      {transaction.creditorName}
                    </td>
                    <td className="p-3 text-right">
                      <div className="flex justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-gray-600 hover:text-[#261436]"
                          onClick={() => handleEdit(transaction)}
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 text-red-600 hover:text-red-800"
                          onClick={() =>
                            handleDelete(transaction.transactionId)
                          }
                          disabled={isDeleting === transaction.transactionId}
                        >
                          {isDeleting === transaction.transactionId ? (
                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-red-600 border-t-transparent" />
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
            <DialogDescription className="text-gray-500">
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
                className="w-full px-3 py-2 bg-white border border-gray-300 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
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
                value={editedValues.category}
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
                  className="w-full px-3 py-2 bg-white border border-gray-300 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
                >
                  <SelectValue placeholder="Select a category" />
                </SelectTrigger>
                <SelectContent 
                  className="bg-white border border-gray-300 rounded-md shadow-lg"
                  position="popper"
                  sideOffset={5}
                >
                  {categories.map((category) => (
                    <SelectItem
                      key={category}
                      value={category}
                      className="px-3 py-2 cursor-pointer text-[#261436] hover:bg-gray-50 focus:bg-gray-50 focus:text-[#261436] outline-none"
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
                className="w-full px-3 py-2 bg-white border border-gray-300 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
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
                className="w-full px-3 py-2 bg-white border border-gray-300 rounded-md text-[#261436] focus:outline-none focus:ring-2 focus:ring-[#261436] focus:border-transparent"
              />
            </div>
          </div>
          <DialogFooter className="gap-2 mt-6">
            <Button
              variant="outline"
              onClick={() => setEditingTransaction(null)}
              className="px-4 py-2 text-[#261436] border border-gray-300 hover:bg-gray-50"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSaveEdit}
              disabled={isSubmitting}
              className="px-4 py-2 bg-[#261436] text-white hover:bg-[#372052] focus:ring-2 focus:ring-[#261436] focus:ring-offset-2"
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
    </>
  );
}
