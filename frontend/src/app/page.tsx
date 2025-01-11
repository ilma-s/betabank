"use client";

import { useState, useEffect } from "react";
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
} from "@/@/components/ui/select";
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

export default function Home() {
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
    <div className="bg-[#261436] min-h-screen flex items-center justify-center">
      <div className="container mx-auto p-10">
        <Card className="bg-[#F1E6EA] shadow-lg relative">
          <CardHeader>
            <CardTitle className="text-[#261436] text-lg font-bold">
              Transaction Generator
            </CardTitle>
            <CardDescription className="text-[#261436]">
              Select a persona to generate sample transactions
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex items-center justify-between space-x-4 mb-6">
              <div className="relative">
                <Select
                  onValueChange={setSelectedPersona}
                  value={selectedPersona}
                >
                  <SelectTrigger className="w-[180px] text-[#261436] border border-gray-300 rounded">
                    <SelectValue
                      placeholder="Select a persona"
                      className="text-[#261436]"
                    />
                  </SelectTrigger>
                  <SelectContent className="absolute z-10 bg-white shadow-lg rounded-md w-[180px]">
                    {personas.map((persona) => (
                      <SelectItem
                        key={persona.id}
                        value={persona.id}
                        className="text-[#261436] px-4 py-2 hover:bg-gray-100"
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
                className="bg-[#261436] text-white px-4 py-2 rounded"
              >
                {loading ? (
                  <div className="flex items-center space-x-2">
                    <div className="h-4 w-4 animate-spin rounded-full border-2 border-primary border-t-transparent"></div>
                    <span>Generating...</span>
                  </div>
                ) : (
                  "Generate Transactions"
                )}
              </Button>
            </div>

            {transactions.length > 0 && (
              <div>
                <div className="flex justify-end space-x-2 mb-4">
                  <Button
                    onClick={exportToJson}
                    className="bg-gray-200 text-black px-3 py-2 rounded"
                  >
                    Export to JSON
                  </Button>
                  <Button
                    onClick={exportToCsv}
                    className="bg-gray-200 text-black px-3 py-2 rounded"
                  >
                    Export to CSV
                  </Button>
                  <Button
                    onClick={exportToExcel}
                    className="bg-gray-200 text-black px-3 py-2 rounded"
                  >
                    Export to Excel
                  </Button>
                </div>
                <div className="overflow-x-auto">
                  <Table className="min-w-full text-left text-[#261436] border border-gray-300">
                    <TableHeader className="bg-gray-100">
                      <TableRow>
                        <TableHead className="px-4 py-2">Date</TableHead>
                        <TableHead className="px-4 py-2">Amount</TableHead>
                        <TableHead className="px-4 py-2">Description</TableHead>
                        <TableHead className="px-4 py-2">To</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {transactions.map((tx) => (
                        <TableRow
                          key={tx.transactionId}
                          className="hover:bg-gray-50"
                        >
                          <TableCell className="px-4 py-2">
                            {new Date(tx.bookingDateTime).toLocaleDateString()}
                          </TableCell>
                          <TableCell className="px-4 py-2">
                            {tx.transactionAmount.amount}{" "}
                            {tx.transactionAmount.currency}
                          </TableCell>
                          <TableCell className="px-4 py-2">
                            {tx.remittanceInformationUnstructured}
                          </TableCell>
                          <TableCell className="px-4 py-2">
                            {tx.creditorName}
                          </TableCell>
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
    </div>
  );
}
