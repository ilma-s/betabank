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
import { Transaction, Persona } from "@/types";

function Main() {
  const { logout } = useAuth();
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [selectedPersona, setSelectedPersona] = useState<string>("");
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPersonas();
  }, []);

  const fetchPersonas = async () => {
    try {
      const response = await axios.get("http://localhost:8000/personas");
      setPersonas(response.data.personas);
    } catch (error) {
      console.error("Failed to fetch personas:", error);
    }
  };

  const generateTransactions = async () => {
    if (!selectedPersona) return;

    setLoading(true);
    try {
      const response = await axios.get(
        `http://localhost:8000/generate/${selectedPersona}`
      );
      setTransactions(response.data.transactions);
    } catch (error) {
      console.error("Failed to generate transactions:", error);
    } finally {
      setLoading(false);
    }
  };

  const exportToJson = () => {
    const dataStr =
      "data:text/json;charset=utf-8," +
      encodeURIComponent(JSON.stringify(transactions));
    const downloadAnchorNode = document.createElement("a");
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "transactions.json");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const exportToCsv = () => {
    const parser = new Parser();
    const csv = parser.parse(transactions);
    const dataStr = "data:text/csv;charset=utf-8," + encodeURIComponent(csv);
    const downloadAnchorNode = document.createElement("a");
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", "transactions.csv");
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
  };

  const exportToExcel = () => {
    const worksheet = XLSX.utils.json_to_sheet(transactions);
    const workbook = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(workbook, worksheet, "Transactions");
    XLSX.writeFile(workbook, "transactions.xlsx");
  };

  if (loading) return <div>Loading transactions...</div>;
  if (error) return <div>Error loading transactions: {error}</div>;

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
                <div className="flex space-x-2">
                  <Button onClick={exportToJson} className="bg-[#261436] text-white">
                    Export to JSON
                  </Button>
                  <Button onClick={exportToCsv} className="bg-[#261436] text-white">
                    Export to CSV
                  </Button>
                  <Button onClick={exportToExcel} className="bg-[#261436] text-white">
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
