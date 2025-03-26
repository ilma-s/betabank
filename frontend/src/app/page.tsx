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
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
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

function Main() {
  const { logout, token } = useAuth();
  
  // State for generator tab
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedPersona, setSelectedPersona] = useState<string>("");
  const [batchName, setBatchName] = useState<string>("");
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [currentBatchId, setCurrentBatchId] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // State for history tab
  const [batches, setBatches] = useState<TransactionBatch[]>([]);
  const [selectedBatch, setSelectedBatch] = useState<TransactionBatch | null>(null);
  const [batchTransactions, setBatchTransactions] = useState<Transaction[]>([]);
  const [loadingBatch, setLoadingBatch] = useState(false);

  // Mock data for admin user
  const mockPersonas: Persona[] = [
    { id: "1", name: "Crypto Enthusiast", description: "A tech-savvy individual who primarily invests in cryptocurrencies" },
    { id: "2", name: "Shopping Addict", description: "Someone who frequently shops online and at retail stores" },
    { id: "3", name: "Gambling Addict", description: "A person with a regular gambling habit" },
    { id: "4", name: "Money Mule", description: "Individual involved in moving illegally acquired money" }
  ];
  
  // Generate 30 random transactions
  const generateMockTransactions = (count: number = 30): Transaction[] => {
    const merchants = [
      "Amazon EU", "Netflix", "Local Supermarket", "Uber", "Spotify",
      "Apple Store", "Zara", "H&M", "IKEA", "Target", "Walmart",
      "Best Buy", "Costco", "McDonald's", "Starbucks", "Shell Gas",
      "Booking.com", "Airbnb", "Delta Airlines", "Coinbase", "Binance",
      "Steam", "PlayStation Store", "Nike", "Adidas", "Deliveroo",
      "Subway", "KFC", "Burger King", "DoorDash"
    ];
    
    const categories = [
      "Shopping", "Entertainment", "Groceries", "Transportation", "Music",
      "Technology", "Fashion", "Home", "Food", "Gaming", "Travel",
      "Cryptocurrency", "Utilities", "Healthcare", "Education", "Gifts"
    ];
    
    const descriptions = [
      "Online Purchase", "Monthly Subscription", "Grocery Shopping", 
      "Ride Service", "Premium Subscription", "App Purchase", "Clothing",
      "Home Decor", "Food Delivery", "Digital Game", "Vacation Booking",
      "Crypto Investment", "Bill Payment", "Medical Services", "Course Fee",
      "Birthday Gift"
    ];
    
    const transactions: Transaction[] = [];
    
    // Create a date 30 days ago as a starting point
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - 30);
    
    for (let i = 0; i < count; i++) {
      // Generate random date between 30 days ago and now
      const txDate = new Date(
        startDate.getTime() + Math.random() * (new Date().getTime() - startDate.getTime())
      );
      
      // Generate random amount between 1 and 500
      const amount = (Math.random() * 499 + 1).toFixed(2);
      
      // Pick random values from our arrays
      const merchantIndex = Math.floor(Math.random() * merchants.length);
      const categoryIndex = Math.floor(Math.random() * categories.length);
      const descriptionIndex = Math.floor(Math.random() * descriptions.length);
      
      transactions.push({
        transactionId: `tx-${i + 1}`,
        bookingDateTime: txDate.toISOString(),
        valueDateTime: txDate.toISOString(),
        transactionAmount: { amount, currency: "EUR" },
        remittanceInformationUnstructured: descriptions[descriptionIndex],
        creditorName: merchants[merchantIndex],
        creditorAccount: { iban: `DE${Math.floor(Math.random() * 1000000000)}` },
        debtorName: "John Doe",
        debtorAccount: { iban: "DE987654321" },
        category: categories[categoryIndex]
      });
    }
    
    // Sort by date, newest first
    return transactions.sort((a, b) => 
      new Date(b.bookingDateTime).getTime() - new Date(a.bookingDateTime).getTime()
    );
  };
  
  const mockTransactions: Transaction[] = generateMockTransactions();
  
  const mockBatches: TransactionBatch[] = [
    {
      id: 1,
      name: "Sample Batch 1",
      persona_id: 1,
      persona_name: "Crypto Enthusiast",
      created_at: new Date().toISOString(),
      transaction_count: 30
    },
    {
      id: 2,
      name: "Sample Batch 2",
      persona_id: 2,
      persona_name: "Shopping Addict",
      created_at: new Date().toISOString(),
      transaction_count: 30
    }
  ];

  const isAdminUser = token === 'admin-demo-token-12345';

  useEffect(() => {
    if (token) {
      fetchPersonas();
      fetchBatches();
    }
  }, [token]);

  const fetchPersonas = async () => {
    // If admin user, return mock data
    if (isAdminUser) {
      setPersonas(mockPersonas);
      return;
    }
    
    // Otherwise, use the API
    try {
      const response = await axios.get("http://localhost:8000/personas", {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      setPersonas(response.data.personas);
    } catch (error) {
      console.error("Failed to fetch personas:", error);
    }
  };

  const fetchBatches = async () => {
    // If admin user, return mock data
    if (isAdminUser) {
      setBatches(mockBatches);
      return;
    }
    
    // Otherwise, use the API
    try {
      const response = await axios.get("http://localhost:8000/batches", {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      setBatches(response.data.batches);
    } catch (error) {
      console.error("Failed to fetch batches:", error);
    }
  };

  const generateTransactions = async () => {
    if (!selectedPersona) return;

    setLoading(true);
    try {
      // If admin user, return mock data
      if (isAdminUser) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Generate fresh transactions for this batch
        const freshTransactions = generateMockTransactions();
        setTransactions(freshTransactions);
        setCurrentBatchId(Math.floor(Math.random() * 1000));
        
        // Create a new mock batch
        const newBatch: TransactionBatch = {
          id: Math.floor(Math.random() * 1000),
          name: batchName || `Batch ${new Date().toLocaleString()}`,
          persona_id: parseInt(selectedPersona),
          persona_name: mockPersonas.find(p => p.id === selectedPersona)?.name || "Unknown",
          created_at: new Date().toISOString(),
          transaction_count: freshTransactions.length
        };
        
        // Add to batches
        setBatches([newBatch, ...batches]);
        return;
      }
      
      // Otherwise, use the API
      const response = await axios.get(
        `http://localhost:8000/generate/${selectedPersona}`,
        { 
          params: { batch_name: batchName || undefined },
          headers: {
            Authorization: `Bearer ${token}`
          }
        }
      );
      setTransactions(response.data.transactions);
      setCurrentBatchId(response.data.batch_id);
      
      // Refresh batches list
      fetchBatches();
    } catch (error) {
      console.error("Failed to generate transactions:", error);
    } finally {
      setLoading(false);
    }
  };

  const viewBatchDetails = async (batch: TransactionBatch) => {
    setSelectedBatch(batch);
    setLoadingBatch(true);
    
    try {
      // If admin user, return mock data
      if (isAdminUser) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Generate unique but consistent transactions for this batch using the batch ID as seed
        const batchTransactions = generateMockTransactions(batch.transaction_count);
        setBatchTransactions(batchTransactions);
        return;
      }
      
      // Otherwise, use the API
      const response = await axios.get(`http://localhost:8000/batches/${batch.id}`, {
        headers: {
          Authorization: `Bearer ${token}`
        }
      });
      setBatchTransactions(response.data.transactions);
    } catch (error) {
      console.error("Failed to fetch batch transactions:", error);
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

  if (loading) return <div className="bg-[#261436] min-h-screen flex items-center justify-center text-white">Loading transactions...</div>;
  if (error) return <div className="bg-[#261436] min-h-screen flex items-center justify-center text-white">Error loading transactions: {error}</div>;

  return (
    <div className="bg-[#261436] min-h-screen">
      <div className="container mx-auto py-10">
        <div className="flex justify-end mb-4">
          <Button 
            onClick={logout}
            className="bg-[#F1E6EA] text-[#261436]"
          >
            Logout
          </Button>
        </div>
        
        <Tabs defaultValue="generator" className="w-full">
          <TabsList className="mb-4 bg-[#F1E6EA] text-[#261436]">
            <TabsTrigger value="generator">Generator</TabsTrigger>
            <TabsTrigger value="history">Transaction History</TabsTrigger>
          </TabsList>
          
          {/* Generator Tab */}
          <TabsContent value="generator">
            <Card className="bg-[#F1E6EA]">
              <CardHeader>
                <CardTitle className="text-[#261436]">
                  Transaction Generator
                </CardTitle>
                <CardDescription className="text-[#261436]">
                  Select a persona to generate sample transactions
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label htmlFor="batchName" className="text-[#261436] block mb-1">Batch Name (Optional)</label>
                  <input 
                    id="batchName"
                    value={batchName}
                    onChange={(e) => setBatchName(e.target.value)}
                    placeholder="Enter a name for this batch"
                    className="text-[#261436] w-full p-2 rounded border border-gray-300"
                  />
                </div>
                
                <div className="space-y-2">
                  <label htmlFor="persona" className="text-[#261436] block mb-1">Select Persona</label>
                  <Select
                    onValueChange={setSelectedPersona}
                    value={selectedPersona}
                  >
                    <SelectTrigger className="w-full bg-white text-[#261436] border border-gray-300 p-2 rounded">
                      {selectedPersona ? 
                        personas.find(p => p.id === selectedPersona)?.name || "Select a persona" 
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
                      <h3 className="text-[#261436] font-semibold">Generated Transactions</h3>
                      <div className="flex space-x-2">
                        <Button onClick={() => exportToJson(transactions)} className="bg-[#261436] text-white">
                          Export to JSON
                        </Button>
                        <Button onClick={() => exportToCsv(transactions)} className="bg-[#261436] text-white">
                          Export to CSV
                        </Button>
                        <Button onClick={() => exportToExcel(transactions)} className="bg-[#261436] text-white">
                          Export to Excel
                        </Button>
                      </div>
                    </div>
                    <div className="rounded-md border">
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
                <CardHeader>
                  <CardTitle className="text-[#261436]">Transaction Batches</CardTitle>
                </CardHeader>
                <CardContent>
                  {batches.length === 0 ? (
                    <p className="text-[#261436]">No transaction batches found.</p>
                  ) : (
                    <div className="space-y-3">
                      {batches.map((batch) => (
                        <div 
                          key={batch.id} 
                          className={`p-3 rounded border cursor-pointer ${
                            selectedBatch?.id === batch.id 
                              ? 'bg-[#261436] text-white' 
                              : 'bg-white text-[#261436] hover:bg-gray-100'
                          }`}
                          onClick={() => viewBatchDetails(batch)}
                        >
                          <h3 className="font-medium">{batch.name}</h3>
                          <p className="text-sm opacity-80">{batch.persona_name}</p>
                          <div className="flex justify-between text-xs mt-1">
                            <span>{new Date(batch.created_at).toLocaleDateString()}</span>
                            <span>{batch.transaction_count} transactions</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
              
              {/* Batch Details */}
              <Card className="bg-[#F1E6EA] col-span-1 md:col-span-2">
                <CardHeader>
                  <CardTitle className="text-[#261436]">
                    {selectedBatch ? `Batch: ${selectedBatch.name}` : 'Select a Batch'}
                  </CardTitle>
                  {selectedBatch && (
                    <CardDescription className="text-[#261436]">
                      {selectedBatch.persona_name} • {new Date(selectedBatch.created_at).toLocaleString()}
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
                      <div className="rounded-md border">
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
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
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
