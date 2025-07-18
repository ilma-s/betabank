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
import axios, { AxiosError } from "axios";
import { Parser } from "json2csv";
import * as XLSX from "xlsx";
import { Transaction, Persona, TransactionBatch } from "@/types";
import { DistributionEditor } from "@/components/DistributionEditor";
import { Pencil } from "lucide-react";
import { useToast } from "@/components/ui/use-toast";
import Link from "next/link";
import { BatchView } from "@/components/BatchView";
import { BatchList } from "@/components/BatchList";

interface ApiError {
  detail: string;
}

function Main() {
  const { logout, token } = useAuth();
  const { toast } = useToast();

  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedPersona, setSelectedPersona] = useState<number | null>(null);
  const [batchName, setBatchName] = useState<string>("");
  const [selectedMonths, setSelectedMonths] = useState<string>("3");
  const [transactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState("generator");
  const [editingDistribution, setEditingDistribution] = useState<{
    personaId: number;
    personaName: string;
    batchId?: number;
    currentDistribution?: Record<string, number>;
    isNewPersona?: boolean;
  } | null>(null);

  const [batches, setBatches] = useState<TransactionBatch[]>([]);
  const [selectedBatch, setSelectedBatch] = useState<TransactionBatch | null>(
    null
  );
  const [batchTransactions, setBatchTransactions] = useState<Transaction[]>([]);
  const [loadingBatch, setLoadingBatch] = useState(false);
  const [editingBatchId, setEditingBatchId] = useState<number | null>(null);
  const [editingBatchName, setEditingBatchName] = useState<string>("");

  const API_BASE_URL = "http://localhost:8000";

  useEffect(() => {
    if (token) {
      const loadInitialData = async () => {
        console.log(
          "Starting initial data load with token:",
          token?.substring(0, 10) + "..."
        );
        setInitializing(true);
        try {
          axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;

          try {
            await axios.get(`${API_BASE_URL}/personas`);
          } catch (error) {
            const axiosError = error as AxiosError<ApiError>;
            if (axiosError.response?.status === 401) {
              console.log("Token invalid, logging out user");
              logout();
              setError("Your session has expired. Please log in again.");
              return;
            }
            throw error;
          }

          await axios.post(`${API_BASE_URL}/ensure-personas`);

          console.log("Making parallel API calls for initial data");
          const [personasResponse, batchesResponse] = await Promise.all([
            axios.get(`${API_BASE_URL}/personas`),
            axios.get(`${API_BASE_URL}/batches`),
          ]);

          console.log("Initial data loaded successfully");
          setPersonas(personasResponse.data);
          setBatches(batchesResponse.data);
          setError(null);
        } catch (error) {
          console.error("Failed to load initial data:", error);
          const axiosError = error as AxiosError<ApiError>;
          if (axiosError.response?.status === 401) {
            console.log("Got 401 during initial load, user needs to re-login");
            logout();
            setError("Your session has expired. Please log in again.");
          } else {
            setError("Failed to load initial data. Please try again.");
          }
        } finally {
          setInitializing(false);
        }
      };

      loadInitialData();
    }
  }, [token, logout]);

  const fetchPersonas = async () => {
    try {
      console.log("Refreshing personas data");
      const response = await axios.get(`${API_BASE_URL}/personas`);
      setPersonas(response.data);
    } catch (error) {
      console.error("Failed to fetch personas:", error);
      const axiosError = error as AxiosError<ApiError>;
      if (axiosError.response && axiosError.response.status !== 401) {
        setError("Failed to load personas. Please try again later.");
      }
    }
  };

  const fetchBatches = async () => {
    try {
      console.log("Refreshing batches data");
      const response = await axios.get(`${API_BASE_URL}/batches`);
      setBatches(response.data);
    } catch (error) {
      console.error("Failed to fetch batches:", error);
      const axiosError = error as AxiosError<ApiError>;
      if (axiosError.response && axiosError.response.status !== 401) {
        setError("Failed to load transaction batches. Please try again later.");
      }
    }
  };

  const generateTransactions = async () => {
    if (!selectedPersona) return;

    const months = parseInt(selectedMonths);
    // Always use a multiple of 100 for batch_size
    const batch_size = Math.max(1, months) * 100;

    console.log("Generating transactions for persona:", selectedPersona);
    setLoading(true);
    try {
      const response = await axios.post(
        `${API_BASE_URL}/generate/${selectedPersona}`,
        {
          batch_name: batchName || undefined,
          batch_size,
        }
      );
      console.log("Successfully generated transactions");

      setBatchName("");
      setSelectedPersona(null);

      setSelectedBatch(response.data);
      setBatchTransactions(response.data.transactions);

      await fetchBatches();

      setActiveTab("history");

      toast({
        title: "Success",
        description: "Transactions generated successfully",
      });
    } catch (error) {
      console.error("Failed to generate transactions:", error);
      const axiosError = error as AxiosError<ApiError>;
      if (axiosError.response && axiosError.response.status !== 401) {
        setError("Failed to generate transactions. Please try again later.");
      }
    } finally {
      setLoading(false);
    }
  };

  const viewBatchDetails = async (batchId: number) => {
    console.log("Fetching details for batch:", batchId);
    const batch = batches.find((b) => b.id === batchId);
    if (!batch) return;

    setSelectedBatch(batch);
    setLoadingBatch(true);

    try {
      const response = await axios.get(`${API_BASE_URL}/batches/${batchId}`);
      console.log("Successfully loaded batch transactions");
      setBatchTransactions(response.data.transactions);
    } catch (error) {
      console.error("Failed to load batch details:", error);
      const axiosError = error as AxiosError<ApiError>;
      const errorMessage =
        axiosError.response?.data?.detail ||
        (error instanceof Error
          ? error.message
          : "Failed to load batch details");
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      });
    } finally {
      setLoadingBatch(false);
    }
  };

  const handleBatchExport = async (format: "json" | "csv" | "excel") => {
    if (!selectedBatch || !batchTransactions.length) return;

    try {
      switch (format) {
        case "json":
          const jsonData = JSON.stringify(batchTransactions, null, 2);
          const jsonBlob = new Blob([jsonData], { type: "application/json" });
          const jsonUrl = URL.createObjectURL(jsonBlob);
          const jsonLink = document.createElement("a");
          jsonLink.href = jsonUrl;
          jsonLink.download = `batch-${
            selectedBatch.name
          }-${new Date().toISOString()}.json`;
          document.body.appendChild(jsonLink);
          jsonLink.click();
          document.body.removeChild(jsonLink);
          URL.revokeObjectURL(jsonUrl);
          break;

        case "csv":
          const parser = new Parser();
          const csv = parser.parse(batchTransactions);
          const csvBlob = new Blob([csv], { type: "text/csv" });
          const csvUrl = URL.createObjectURL(csvBlob);
          const csvLink = document.createElement("a");
          csvLink.href = csvUrl;
          csvLink.download = `batch-${
            selectedBatch.name
          }-${new Date().toISOString()}.csv`;
          document.body.appendChild(csvLink);
          csvLink.click();
          document.body.removeChild(csvLink);
          URL.revokeObjectURL(csvUrl);
          break;

        case "excel":
          const ws = XLSX.utils.json_to_sheet(batchTransactions);
          const wb = XLSX.utils.book_new();
          XLSX.utils.book_append_sheet(wb, ws, "Transactions");
          XLSX.writeFile(
            wb,
            `batch-${selectedBatch.name}-${new Date().toISOString()}.xlsx`
          );
          break;
      }

      toast({
        title: "Success",
        description: `Batch exported to ${format.toUpperCase()} successfully`,
      });
    } catch (error) {
      console.error(`Failed to export batch to ${format}:`, error);
      toast({
        title: "Error",
        description: `Failed to export batch to ${format}. Please try again.`,
        variant: "destructive",
      });
    }
  };

  const handleBatchNameUpdate = async (batchId: number, newName: string) => {
    if (!newName.trim()) {
      setEditingBatchId(null);
      setEditingBatchName("");
      return;
    }

    try {
      const response = await axios.patch(
        `${API_BASE_URL}/batches/${batchId}`,
        {
          name: newName.trim()
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json"
          },
        }
      );

      if (response.status === 200) {
        toast({
          title: "Success",
          description: "Batch name updated successfully",
        });

        fetchBatches();

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

  const handleDistributionUpdate = async () => {
    if (selectedBatch) {
      await viewBatchDetails(selectedBatch.id);
    }
  };

  const refreshBatchAndList = async () => {
    await fetchBatches();

    if (selectedBatch) {
      const response = await axios.get(
        `${API_BASE_URL}/batches/${selectedBatch.id}`
      );
      setBatchTransactions(response.data.transactions);

      setSelectedBatch((prev) =>
        prev
          ? {
              ...prev,
              transaction_count: response.data.transactions.length,
            }
          : null
      );
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
    <div className="bg-[#261436] min-h-screen w-full">
      <div className="w-3/4 mx-auto min-h-screen">
        <div className="container mx-auto py-10">
          <div className="flex justify-end mb-4">
            <Button onClick={logout} className="bg-[#F1E6EA] text-[#261436]">
              Logout
            </Button>
          </div>

          <Tabs
            value={activeTab}
            onValueChange={setActiveTab}
            className="w-full"
          >
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

            <TabsContent value="generator">
              <div className="space-y-6">
                <Card className="bg-[#F1E6EA]">
                  <CardHeader>
                    <CardTitle className="text-[#261436]">
                      Generate Transactions
                    </CardTitle>
                    <CardDescription className="text-[#261436]">
                      Select a persona and generate synthetic transactions
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
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

                    <div className="flex gap-4">
                      <div className="w-1/2 space-y-2">
                        <label
                          htmlFor="persona"
                          className="text-[#261436] block mb-1"
                        >
                          Select Persona
                        </label>
                        <div className="flex items-center gap-2">
                          <Select
                            value={selectedPersona?.toString() || ""}
                            onValueChange={(value) =>
                              setSelectedPersona(parseInt(value))
                            }
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
                                  value={persona.id.toString()}
                                  className="px-4 py-2 hover:bg-gray-100"
                                >
                                  {persona.name}
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>

                      <div className="w-1/2 space-y-2">
                        <label
                          htmlFor="months"
                          className="text-[#261436] block mb-1"
                        >
                          Time Period
                        </label>
                        <Select
                          value={selectedMonths}
                          onValueChange={setSelectedMonths}
                        >
                          <SelectTrigger className="w-full bg-white text-[#261436] border border-gray-300 p-2 rounded">
                            {selectedMonths} months
                          </SelectTrigger>
                          <SelectContent className="bg-white text-[#261436] border border-gray-300 mt-1 rounded shadow-lg w-full">
                            <SelectItem value="3" className="px-4 py-2 hover:bg-gray-100">3 months</SelectItem>
                            <SelectItem value="6" className="px-4 py-2 hover:bg-gray-100">6 months</SelectItem>
                            <SelectItem value="12" className="px-4 py-2 hover:bg-gray-100">12 months</SelectItem>
                            <SelectItem value="24" className="px-4 py-2 hover:bg-gray-100">24 months</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>

                    <Button
                      onClick={generateTransactions}
                      disabled={!selectedPersona || loading}
                      className="w-full bg-[#261436] text-white mt-6"
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
                      <div className="mt-8 space-y-4">
                        <div className="flex justify-between items-center">
                          <h3 className="text-[#261436] font-semibold">
                            Generated Transactions
                          </h3>
                          <div className="flex space-x-2">
                            <Button
                              onClick={() => handleBatchExport("json")}
                              className="bg-[#261436] text-white"
                            >
                              Export to JSON
                            </Button>
                            <Button
                              onClick={() => handleBatchExport("csv")}
                              className="bg-[#261436] text-white"
                            >
                              Export to CSV
                            </Button>
                            <Button
                              onClick={() => handleBatchExport("excel")}
                              className="bg-[#261436] text-white"
                            >
                              Export to Excel
                            </Button>
                          </div>
                        </div>
                        <div className="rounded-md border text-[#261436] bg-white">
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
                                    ).toLocaleDateString("en-GB")}
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
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="history">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-[calc(100vh-12rem)]">
                <Card className="bg-[#F1E6EA] col-span-1 h-full overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-[#261436]">
                      Transaction Batches
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="h-[calc(100%-4rem)] overflow-hidden">
                    {batches.length === 0 ? (
                      <p className="text-[#261436]">
                        No transaction batches found.
                      </p>
                    ) : (
                      <BatchList
                        batches={batches}
                        onBatchClick={viewBatchDetails}
                        onBatchDeleted={fetchBatches}
                        token={token!}
                        selectedBatchId={selectedBatch?.id}
                      />
                    )}
                  </CardContent>
                </Card>

                <Card className="bg-[#F1E6EA] col-span-1 md:col-span-2 h-full overflow-hidden">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-[#261436] flex items-center">
                      {selectedBatch && (
                        <>
                          Batch:{" "}
                          {editingBatchId === selectedBatch.id ? (
                            <input
                              type="text"
                              value={editingBatchName}
                              onChange={(e) =>
                                setEditingBatchName(e.target.value)
                              }
                              onKeyDown={(e) => {
                                e.stopPropagation();
                                if (e.key === "Enter") {
                                  e.preventDefault();
                                  handleBatchNameUpdate(
                                    selectedBatch.id,
                                    editingBatchName
                                  );
                                } else if (e.key === "Escape") {
                                  e.preventDefault();
                                  setEditingBatchId(null);
                                  setEditingBatchName("");
                                }
                              }}
                              onBlur={() => {
                                if (editingBatchName.trim() !== "") {
                                  handleBatchNameUpdate(
                                    selectedBatch.id,
                                    editingBatchName
                                  );
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
                        {selectedBatch.persona_name} •{" "}
                        {(() => {
                          try {
                            const date = new Date(selectedBatch.created_at);
                            if (isNaN(date.getTime())) {
                              return "Invalid Date";
                            }
                            return date.toLocaleDateString("en-GB", {
                              day: "2-digit",
                              month: "2-digit",
                              year: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            });
                          } catch (error) {
                            console.error("Error formatting date:", error);
                            return "Invalid Date";
                          }
                        })()}{" "}
                        • {selectedBatch.months} months of generated data
                      </CardDescription>
                    )}
                  </CardHeader>
                  <CardContent className="h-[calc(100%-4rem)] overflow-y-auto">
                    {!selectedBatch ? (
                      <p className="text-[#261436]">
                        Select a batch from the list to view transactions.
                      </p>
                    ) : loadingBatch ? (
                      <div className="flex justify-center p-8">
                        <div className="h-8 w-8 animate-spin rounded-full border-4 border-[#261436] border-t-transparent" />
                      </div>
                    ) : batchTransactions.length === 0 ? (
                      <p className="text-[#261436]">
                        No transactions found in this batch.
                      </p>
                    ) : (
                      <div className="space-y-4">
                        <BatchView
                          transactions={batchTransactions}
                          batchName={selectedBatch.name}
                          createdAt={selectedBatch.created_at}
                          months={selectedBatch.months}
                          onExport={handleBatchExport}
                          onTransactionUpdated={refreshBatchAndList}
                          personaId={selectedBatch.persona_id}
                          personaName={selectedBatch.persona_name}
                          batchId={selectedBatch.id}
                          token={token!}
                        />
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>

          {editingDistribution !== null && (
            <DistributionEditor
              personaId={editingDistribution.personaId}
              personaName={editingDistribution.personaName}
              initialDistribution={
                personas.find((p) => p.id === editingDistribution.personaId)
                  ?.config_json?.custom_distribution
              }
              onClose={async () => {
                if (
                  editingDistribution.batchId &&
                  selectedBatch?.id === editingDistribution.batchId
                ) {
                  await viewBatchDetails(selectedBatch.id);

                  setSelectedBatch((prevBatch) => ({
                    ...prevBatch!,
                    updated_at: new Date().toISOString(),
                  }));
                }
                await fetchBatches();
                setEditingDistribution(null);
              }}
              token={token!}
              batchId={editingDistribution.batchId}
              currentDistribution={editingDistribution.currentDistribution}
              onDistributionUpdated={handleDistributionUpdate}
              isNewPersona={editingDistribution.isNewPersona}
            />
          )}
        </div>
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
