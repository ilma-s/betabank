"use client";

import { useState, useEffect } from "react";
import Login from "@/components/Login";
import { useAuth } from "@/context/AuthContext";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import axios from "axios";
import { Parser } from "json2csv";
import * as XLSX from "xlsx";
import { Transaction, Persona, TransactionBatch } from "@/types";
import { TransactionCharts } from "@/components/TransactionCharts";
import { Trash2, Pencil } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";

function Main() {
  const { logout, token } = useAuth();
  const { toast } = useToast();

  // State for generator tab
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedPersona, setSelectedPersona] = useState<string>("");
  const [batchName, setBatchName] = useState<string>("");
  const [selectedMonths, setSelectedMonths] = useState<string>("3"); // Default to 3 months
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [currentBatchId, setCurrentBatchId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // State for history tab
  const [batches, setBatches] = useState<TransactionBatch[]>([]);
  const [selectedBatch, setSelectedBatch] = useState<TransactionBatch | null>(
    null
  );
  const [batchTransactions, setBatchTransactions] = useState<Transaction[]>([]);
  const [loadingBatch, setLoadingBatch] = useState(false);
  const [isDeleting, setIsDeleting] = useState<number | null>(null);
  const [editingBatchId, setEditingBatchId] = useState<number | null>(null);
  const [editingBatchName, setEditingBatchName] = useState<string>("");

  // API base URL
  const API_BASE_URL = "http://localhost:8000";

  // Initial data load
  useEffect(() => {
    if (token) {
      const loadInitialData = async () => {
        console.log(
          "Starting initial data load with token:",
          token?.substring(0, 10) + "..."
        );
        setInitializing(true);
        try {
          // Make sure headers are set before making API calls
          console.log(
            "Explicitly setting Authorization header for initial load"
          );
          axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;

          // Ensure all personas exist first
          await axios.post(`${API_BASE_URL}/ensure-personas`);

          // Load data in parallel
          console.log("Making parallel API calls for initial data");
          const [personasResponse, batchesResponse] = await Promise.all([
            axios.get(`${API_BASE_URL}/personas`),
            axios.get(`${API_BASE_URL}/batches`),
          ]);

          console.log("Initial data loaded successfully");
          setPersonas(personasResponse.data.personas);
          setBatches(batchesResponse.data.batches);
          setError(null);
        } catch (error: any) {
          console.error("Failed to load initial data:", error);
          if (error.response && error.response.status === 401) {
            console.log("Got 401 during initial load, user needs to re-login");
            setError("You are not authorized. Please log in again.");
            // We don't need to call logout() here as the interceptor will handle it
          } else {
            setError("Failed to load initial data. Please try again.");
          }
        } finally {
          setInitializing(false);
        }
      };

      loadInitialData();
    }
  }, [token]);

  // The rest of the fetchPersonas and fetchBatches functions will be used for refreshing data
  const fetchPersonas = async () => {
    try {
      console.log("Refreshing personas data");
      const response = await axios.get(`${API_BASE_URL}/personas`);
      setPersonas(response.data.personas);
    } catch (error: any) {
      console.error("Failed to fetch personas:", error);
      if (error.response && error.response.status !== 401) {
        setError("Failed to load personas. Please try again later.");
      }
    }
  };

  const fetchBatches = async () => {
    try {
      console.log("Refreshing batches data");
      const response = await axios.get(`${API_BASE_URL}/batches`);
      setBatches(response.data.batches);
    } catch (error: any) {
      console.error("Failed to fetch batches:", error);
      if (error.response && error.response.status !== 401) {
        setError("Failed to load transaction batches. Please try again later.");
      }
    }
  };

  const generateTransactions = async () => {
    if (!selectedPersona) return;

    console.log("Generating transactions for persona:", selectedPersona);
    setLoading(true);
    try {
      const response = await axios.get(
        `${API_BASE_URL}/generate/${selectedPersona}`,
        {
          params: {
            batch_name: batchName || undefined,
            months: parseInt(selectedMonths),
          },
        }
      );
      console.log("Successfully generated transactions");
      setTransactions(response.data.transactions);
      setCurrentBatchId(response.data.batch_id);

      // Refresh batches list
      fetchBatches();
    } catch (error: any) {
      console.error("Failed to generate transactions:", error);
      if (error.response && error.response.status !== 401) {
        setError("Failed to generate transactions. Please try again later.");
      }
    } finally {
      setLoading(false);
    }
  };

  const viewBatchDetails = async (batch: TransactionBatch) => {
    console.log("Fetching details for batch:", batch.id);
    setSelectedBatch(batch);
    setLoadingBatch(true);

    try {
      const response = await axios.get(`${API_BASE_URL}/batches/${batch.id}`);
      console.log("Successfully loaded batch transactions");
      setBatchTransactions(response.data.transactions);
    } catch (error: any) {
      console.error("Failed to fetch batch transactions:", error);
      if (error.response && error.response.status !== 401) {
        setError("Failed to load batch details. Please try again later.");
      }
    } finally {
      setLoadingBatch(false);
    }
  };

  const exportToJson = (txData: Transaction[]) => {
    const dataStr =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(txData));
    const downloadAnchorNode = document.createElement("a");
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "transactions.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const exportToCsv = (txData: Transaction[]) => {
    const parser = new Parser();
    const csv = parser.parse(txData);
    const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csv);
    const downloadAnchorNode = document.createElement("a");
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "transactions.csv");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const exportToExcel = (txData: Transaction[]) => {
    const worksheet = XLSX.utils.json_to_sheet(txData);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Transactions");
    XLSX.writeFile(workbook, "transactions.xlsx");
  };

  const handleDelete = async (e: React.MouseEvent, batchId: number) => {
    e.stopPropagation(); // Prevent batch click when clicking delete

    if (isDeleting) return; // Prevent multiple deletes

    if (!confirm("Are you sure you want to delete this batch?")) {
      return;
    }

    setIsDeleting(batchId);

    try {
      const response = await axios.delete(`${API_BASE_URL}/batches/${batchId}`);

      if (response.status !== 200) {
        throw new Error("Failed to delete batch");
      }

      toast({
        title: "Success",
        description: "Batch deleted successfully",
      });

      // Refresh batches list
      fetchBatches();

      // Clear selected batch if it was deleted
      if (selectedBatch?.id === batchId) {
        setSelectedBatch(null);
        setBatchTransactions([]);
      }
    } catch (error) {
      console.error("Failed to delete batch:", error);
      toast({
        title: "Error",
        description: "Failed to delete batch",
        variant: "destructive",
      });
    } finally {
      setIsDeleting(null);
    }
  };

  const handleBatchNameUpdate = async (batchId: number, newName: string) => {
    // Don't update if name is empty or only whitespace
    if (!newName.trim()) {
      setEditingBatchId(null);
      setEditingBatchName("");
      return;
    }

    try {
      const response = await axios.patch(
        `${API_BASE_URL}/batches/${batchId}?name=${encodeURIComponent(
          newName.trim()
        )}`,
        {}, // Empty body since we're using query params
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (response.status === 200) {
        toast({
          title: "Success",
          description: "Batch name updated successfully",
        });

        // Update both the batches list and the selected batch
        fetchBatches();

        // Update the selected batch if it's the one being renamed
        if (selectedBatch && selectedBatch.id === batchId) {
          setSelectedBatch({
            ...selectedBatch,
            name: newName.trim(),
          });
        }
      }
    } catch (error) {
      console.error("Failed to update batch name:", error);
      toast({
        title: "Error",
        description: "Failed to update batch name",
        variant: "destructive",
      });
    } finally {
      setEditingBatchId(null);
      setEditingBatchName("");
    }
  };

  if (initializing)
    return (
      <div className="bg-[#261436] min-h-screen flex items-center justify-center text-white">
        Initializing application...
      </div>
    );
  if (loading)
    return (
      <div className="bg-[#261436] min-h-screen flex items-center justify-center text-white">
        Loading transactions...
      </div>
    );
  if (error)
    return (
      <div className="bg-[#261436] min-h-screen flex items-center justify-center">
        <Card className="bg-[#F1E6EA] p-6 max-w-md">
          <CardTitle className="text-[#261436] mb-4">Error</CardTitle>
          <CardContent>
            <p className="text-red-500">{error}</p>
            <Button
              onClick={() => {
                setError(null);
                // If we were initializing, try to reload data
                if (initializing && token) {
                  fetchPersonas();
                  fetchBatches();
                }
              }}
              className="mt-4 bg-[#261436] text-white"
            >
              Try Again
            </Button>
          </CardContent>
        </Card>
      </div>
    );

  return (
    <div className="bg-[#261436] min-h-screen">
      <div className="container mx-auto py-10">
        <div className="flex justify-end mb-4">
          <Button onClick={logout} className="bg-[#F1E6EA] text-[#261436]">
            Logout
          </Button>
        </div>

        <Tabs defaultValue="generator" className="w-full">
          <TabsList className="mb-4 bg-[#F1E6EA] p-2 rounded-lg flex gap-2">
            <TabsTrigger
              value="generator"
              className="data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:bg-[#F1E6EA] data-[state=inactive]:text-[#261436] px-6 py-2 rounded-md transition-all flex-1"
            >
              Generator
            </TabsTrigger>
            <TabsTrigger
              value="history"
              className="data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:bg-[#F1E6EA] data-[state=inactive]:text-[#261436] px-6 py-2 rounded-md transition-all flex-1"
            >
              Transaction History
            </TabsTrigger>
          </TabsList>

          {/* Generator Tab */}
          <TabsContent value="generator">
            <Card className="bg-[#F1E6EA]">
              <CardHeader>
                <CardTitle className="text-[#261436]">
                  Transaction Generator
                </CardTitle>
                <CardDescription className="text-[#261436]">
                  Select a persona and time period to generate sample
                  transactions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label
                    htmlFor="batchName"
                    className="text-[#261436] block mb-1"
                  >
                    Batch Name (Optional)
                  </label>
                  <input
                    id="batchName"
                    value={batchName}
                    onChange={(e) => setBatchName(e.target.value)}
                    placeholder="Enter a name for this batch"
                    className="text-[#261436] w-full p-2 rounded border border-gray-300"
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <label
                      htmlFor="persona"
                      className="text-[#261436] block mb-1"
                    >
                      Select Persona
                    </label>
                    <Select
                      onValueChange={setSelectedPersona}
                      value={selectedPersona}
                    >
                      <SelectTrigger className="w-full bg-white text-[#261436] border border-gray-300 p-2 rounded">
                        {selectedPersona
                          ? personas.find((p) => p.id === selectedPersona)
                              ?.name || "Select a persona"
                          : "Select a persona"}
                      </SelectTrigger>
                      <SelectContent className="bg-white text-[#261436] border border-gray-300 mt-1 rounded shadow-lg w-full">
                        {personas.map((persona) => (
                          <SelectItem
                            key={persona.id}
                            value={persona.id}
                            className="px-4 py-2 hover:bg-gray-100"
                          >
                            {persona.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label
                      htmlFor="months"
                      className="text-[#261436] block mb-1"
                    >
                      Time Period
                    </label>
                    <Select
                      onValueChange={setSelectedMonths}
                      value={selectedMonths}
                    >
                      <SelectTrigger className="w-full bg-white text-[#261436] border border-gray-300 p-2 rounded">
                        {selectedMonths} months
                      </SelectTrigger>
                      <SelectContent className="bg-white text-[#261436] border border-gray-300 mt-1 rounded shadow-lg w-full">
                        <SelectItem
                          value="3"
                          className="px-4 py-2 hover:bg-gray-100"
                        >
                          3 months
                        </SelectItem>
                        <SelectItem
                          value="6"
                          className="px-4 py-2 hover:bg-gray-100"
                        >
                          6 months
                        </SelectItem>
                        <SelectItem
                          value="12"
                          className="px-4 py-2 hover:bg-gray-100"
                        >
                          12 months
                        </SelectItem>
                        <SelectItem
                          value="24"
                          className="px-4 py-2 hover:bg-gray-100"
                        >
                          24 months
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <Button
                  onClick={generateTransactions}
                  disabled={!selectedPersona || loading}
                  className="w-full bg-[#261436] text-white"
                >
                  {loading ? (
                    <div className="flex items-center space-x-2">
                      <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                      <span>Generating...</span>
                    </div>
                  ) : (
                    "Generate Transactions"
                  )}
                </Button>

                {transactions.length > 0 && (
                  <>
                    <div className="flex justify-between items-center">
                      <h3 className="text-[#261436] font-semibold">
                        Generated Transactions
                      </h3>
                      <div className="flex space-x-2">
                        <Button
                          onClick={() => exportToJson(transactions)}
                          className="bg-[#261436] text-white"
                        >
                          Export to JSON
                        </Button>
                        <Button
                          onClick={() => exportToCsv(transactions)}
                          className="bg-[#261436] text-white"
                        >
                          Export to CSV
                        </Button>
                        <Button
                          onClick={() => exportToExcel(transactions)}
                          className="bg-[#261436] text-white"
                        >
                          Export to Excel
                        </Button>
                      </div>
                    </div>
                    <div className="rounded-md border text-[#261436]">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Date</TableHead>
                            <TableHead>Amount</TableHead>
                            <TableHead>Description</TableHead>
                            <TableHead>To</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {transactions.map((tx) => (
                            <TableRow key={tx.transactionId}>
                              <TableCell>
                                {new Date(
                                  tx.bookingDateTime
                                ).toLocaleDateString()}
                              </TableCell>
                              <TableCell>
                                {tx.transactionAmount.amount}{" "}
                                {tx.transactionAmount.currency}
                              </TableCell>
                              <TableCell>
                                {tx.remittanceInformationUnstructured}
                              </TableCell>
                              <TableCell>{tx.creditorName}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          {/* History Tab */}
          <TabsContent value="history">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Batches List */}
              <Card className="bg-[#F1E6EA] col-span-1">
                <CardHeader className="pb-2">
                  <CardTitle className="text-[#261436]">
                    Transaction Batches
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {batches.length === 0 ? (
                    <p className="text-[#261436]">
                      No transaction batches found.
                    </p>
                  ) : (
                    <div className="space-y-3 min-h-[200px] max-h-[600px] overflow-y-auto pr-2">
                      {batches.map((batch) => (
                        <div
                          key={batch.id}
                          className={`p-3 rounded border cursor-pointer group ${
                            selectedBatch?.id === batch.id
                              ? "bg-[#261436] text-white"
                              : "bg-white text-[#261436] hover:bg-gray-100"
                          }`}
                          onClick={() => viewBatchDetails(batch)}
                        >
                          <div className="flex flex-col h-full">
                            <div className="flex justify-between items-start">
                              <div>
                                <h3 className="font-medium">{batch.name}</h3>
                                <p className="text-sm opacity-80">
                                  {batch.persona_name}
                                </p>
                              </div>
                              <button
                                onClick={(e) => handleDelete(e, batch.id)}
                                disabled={isDeleting === batch.id}
                                className={`p-1.5 rounded-full ${
                                  isDeleting === batch.id ? "opacity-50" : ""
                                } hover:bg-red-100 ${
                                  selectedBatch?.id === batch.id
                                    ? "hover:bg-red-900 text-white hover:text-red-100"
                                    : "text-red-600 hover:text-red-800"
                                }`}
                                title="Delete batch"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                            <div className="flex justify-between text-xs mt-2 opacity-80">
                              <span>
                                {new Date(
                                  batch.created_at
                                ).toLocaleDateString()}
                              </span>
                              <span>
                                {batch.transaction_count} transactions
                              </span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Batch Details */}
              <Card className="bg-[#F1E6EA] col-span-1 md:col-span-2">
                <CardHeader className="pb-2">
                  <CardTitle className="text-[#261436] flex items-center">
                    {selectedBatch && (
                      <>
                        Batch:{' '}
                        {editingBatchId === selectedBatch.id ? (
                          <input
                            type="text"
                            value={editingBatchName}
                            onChange={(e) => setEditingBatchName(e.target.value)}
                            onKeyDown={(e) => {
                              e.stopPropagation();
                              if (e.key === 'Enter') {
                                e.preventDefault();
                                handleBatchNameUpdate(selectedBatch.id, editingBatchName);
                              } else if (e.key === 'Escape') {
                                e.preventDefault();
                                setEditingBatchId(null);
                                setEditingBatchName("");
                              }
                            }}
                            onBlur={() => {
                              if (editingBatchName.trim() !== "") {
                                handleBatchNameUpdate(selectedBatch.id, editingBatchName);
                              } else {
                                setEditingBatchId(null);
                                setEditingBatchName("");
                              }
                            }}
                            onClick={(e) => e.stopPropagation()}
                            className="bg-white text-[#261436] px-2 py-1 rounded focus:outline-none focus:ring-2 focus:ring-[#261436] ml-1 min-w-[200px]"
                            autoFocus
                          />
                        ) : (
                          <span
                            className="cursor-pointer hover:opacity-80 ml-1"
                            onDoubleClick={(e) => {
                              e.stopPropagation();
                              setEditingBatchId(selectedBatch.id);
                              setEditingBatchName(selectedBatch.name);
                            }}
                          >
                            {selectedBatch.name}
                          </span>
                        )}
                        <button
                          className="ml-2 p-1.5 rounded-full hover:bg-gray-100 opacity-50 hover:opacity-100"
                          title="Edit batch name"
                          onClick={(e) => {
                            e.stopPropagation();
                            if (editingBatchId !== selectedBatch.id) {
                              setEditingBatchId(selectedBatch.id);
                              setEditingBatchName(selectedBatch.name);
                            }
                          }}
                        >
                          <Pencil className="w-4 h-4" />
                        </button>
                      </>
                    )}
                  </CardTitle>
                  {selectedBatch && (
                    <CardDescription className="text-[#261436]">
                      {selectedBatch.persona_name} • {new Date(selectedBatch.created_at).toLocaleString()} • {selectedBatch.months} months of generated data
                    </CardDescription>
                  )}
                </CardHeader>
                <CardContent>
                  {!selectedBatch ? (
                    <p className="text-[#261436]">Select a batch from the list to view transactions.</p>
                  ) : loadingBatch ? (
                    <div className="flex justify-center p-8">
                      <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#261436] border-t-transparent" />
                    </div>
                  ) : batchTransactions.length === 0 ? (
                    <p className="text-[#261436]">No transactions found in this batch.</p>
                  ) : (
                    <>
                      <div className="flex justify-end space-x-2 mb-4">
                        <Button onClick={() => exportToJson(batchTransactions)} className="bg-[#261436] text-white">
                          Export to JSON
                        </Button>
                        <Button onClick={() => exportToCsv(batchTransactions)} className="bg-[#261436] text-white">
                          Export to CSV
                        </Button>
                        <Button onClick={() => exportToExcel(batchTransactions)} className="bg-[#261436] text-white">
                          Export to Excel
                        </Button>
                      </div>

                      <Tabs defaultValue="transactions" className="w-full">
                        <TabsList className="mb-4 bg-white p-1 rounded-lg flex gap-2">
                          <TabsTrigger 
                            value="transactions" 
                            className="data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:bg-white data-[state=inactive]:text-[#261436] px-6 py-2 rounded-md transition-all flex-1"
                          >
                            Transactions
                          </TabsTrigger>
                          <TabsTrigger 
                            value="analytics" 
                            className="data-[state=active]:bg-[#261436] data-[state=active]:text-white data-[state=inactive]:bg-white data-[state=inactive]:text-[#261436] px-6 py-2 rounded-md transition-all flex-1"
                          >
                            Analytics
                          </TabsTrigger>
                        </TabsList>

                        <TabsContent value="transactions">
                          <div className="rounded-md border text-[#261436]">
                            <Table>
                              <TableHeader>
                                <TableRow>
                                  <TableHead>Date</TableHead>
                                  <TableHead>Amount</TableHead>
                                  <TableHead>Description</TableHead>
                                  <TableHead>To</TableHead>
                                  <TableHead className="w-[100px] text-right">Actions</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {batchTransactions.map((tx) => (
                                  <TableRow key={tx.transactionId}>
                                    <TableCell>
                                      {new Date(tx.bookingDateTime).toLocaleDateString()}
                                    </TableCell>
                                    <TableCell>
                                      {tx.transactionAmount.amount}{" "}
                                      {tx.transactionAmount.currency}
                                    </TableCell>
                                    <TableCell>
                                      {tx.remittanceInformationUnstructured}
                                    </TableCell>
                                    <TableCell>{tx.creditorName}</TableCell>
                                    <TableCell className="text-right">
                                      <div className="flex items-center gap-2 justify-end">
                                        <button
                                          onClick={async (e) => {
                                            e.stopPropagation();
                                            if (!confirm('Are you sure you want to delete this transaction?')) {
                                              return;
                                            }
                                            try {
                                              const response = await axios.delete(
                                                `${API_BASE_URL}/transactions/${tx.transactionId}`,
                                                {
                                                  headers: {
                                                    'Authorization': `Bearer ${token}`
                                                  }
                                                }
                                              );
                                              
                                              if (response.status === 200) {
                                                toast({
                                                  title: "Success",
                                                  description: "Transaction deleted successfully",
                                                });
                                                // Refresh the transactions list
                                                if (selectedBatch) {
                                                  viewBatchDetails(selectedBatch);
                                                }
                                              }
                                            } catch (error) {
                                              console.error("Failed to delete transaction:", error);
                                              toast({
                                                title: "Error",
                                                description: "Failed to delete transaction",
                                                variant: "destructive",
                                              });
                                            }
                                          }}
                                          className="p-1.5 rounded-full hover:bg-red-100 text-red-600 hover:text-red-800"
                                          title="Delete transaction"
                                        >
                                          <Trash2 className="w-4 h-4" />
                                        </button>
                                        <button
                                          className="p-1.5 rounded-full hover:bg-gray-100"
                                          title="Edit transaction"
                                        >
                                          <Pencil className="w-4 h-4" />
                                        </button>
                                      </div>
                                    </TableCell>
                                  </TableRow>
                                ))}
                              </TableBody>
                            </Table>
                          </div>
                        </TabsContent>

                        <TabsContent value="analytics">
                          <TransactionCharts 
                            transactions={batchTransactions} 
                            personaType={selectedBatch.persona_name}
                          />
                        </TabsContent>
                      </Tabs>
                    </>
                  )}
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default function Page() {
  const { isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  return <Main />;
}
