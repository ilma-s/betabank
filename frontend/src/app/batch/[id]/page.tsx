"use client";

import { useAuth } from "@/context/AuthContext";
import { useRouter } from "next/navigation";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import Link from "next/link";
import { useEffect, useState } from "react";
import axios from "axios";

interface Transaction {
  transactionId: string;
  bookingDateTime: string;
  valueDateTime: string;
  transactionAmount: {
    amount: string;
    currency: string;
  };
  creditorName: string;
  creditorAccount: {
    iban: string;
  };
  debtorName: string;
  debtorAccount: {
    iban: string;
  };
  remittanceInformationUnstructured: string;
  category: string;
}

interface BatchDetails {
  id: number;
  name: string;
  persona_id: number;
  persona_name: string;
  created_at: string;
  transaction_count: number;
  preview: any;
  transactions: Transaction[];
  months: number;
}

export default function BatchPage({ params }: { params: { id: string } }) {
  const { isAuthenticated, token } = useAuth();
  const router = useRouter();
  const [batch, setBatch] = useState<BatchDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!isAuthenticated) {
      router.push("/");
      return;
    }

    const fetchBatch = async () => {
      try {
        const response = await axios.get(`http://localhost:8000/batches/${params.id}`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setBatch(response.data);
      } catch (error) {
        setError("Failed to load batch details");
        console.error("Error fetching batch:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchBatch();
  }, [isAuthenticated, token, params.id]);

  if (!isAuthenticated) {
    return null;
  }

  if (loading) {
    return (
      <div className="bg-[#261436] min-h-screen flex items-center justify-center">
        <div className="text-white">Loading...</div>
      </div>
    );
  }

  if (error || !batch) {
    return (
      <div className="bg-[#261436] min-h-screen flex items-center justify-center">
        <Card className="bg-[#F1E6EA] p-6">
          <CardTitle className="text-[#261436] mb-4">Error</CardTitle>
          <CardContent>
            <p className="text-red-500">{error || "Batch not found"}</p>
            <Button
              onClick={() => router.push("/history")}
              className="mt-4 bg-[#261436] text-white"
            >
              Return to History
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="bg-[#261436] min-h-screen">
      <div className="container mx-auto py-10">
        <div className="flex justify-between items-center mb-6">
          <Link
            href="/history"
            className="text-white hover:text-gray-200 transition-colors"
          >
            ← Back to History
          </Link>
        </div>

        <Card className="bg-[#F1E6EA]">
          <CardHeader>
            <CardTitle className="text-[#261436]">{batch.name}</CardTitle>
            <CardDescription className="text-[#261436]">
              {batch.persona_name} • {new Date(batch.created_at).toLocaleString()} •{" "}
              {batch.months} months of generated data
            </CardDescription>
          </CardHeader>
          <CardContent>
            {/* Transaction list will go here */}
            <div className="space-y-4">
              {batch.transactions.map((transaction) => (
                <div
                  key={transaction.transactionId}
                  className="p-4 bg-white rounded-lg border border-gray-200"
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium text-[#261436]">
                        {transaction.creditorName}
                      </p>
                      <p className="text-sm text-[#261436]/70">
                        {transaction.remittanceInformationUnstructured}
                      </p>
                      <p className="text-sm text-[#261436]/70">
                        {new Date(transaction.bookingDateTime).toLocaleDateString('en-GB', {
                          day: '2-digit',
                          month: '2-digit',
                          year: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit'
                        })}
                      </p>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-[#261436]">
                        {transaction.transactionAmount.amount}{" "}
                        {transaction.transactionAmount.currency}
                      </p>
                      <p className="text-sm text-[#261436]/70">
                        {transaction.category}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 