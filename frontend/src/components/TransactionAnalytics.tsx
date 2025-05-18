import { useMemo, useState } from 'react';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  ComposedChart,
  Area
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, ZoomIn, Filter } from "lucide-react";
import { Transaction } from "@/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

interface TransactionAnalyticsProps {
  transactions: Transaction[];
}

export function TransactionAnalytics({ transactions }: TransactionAnalyticsProps) {
  const [timeRange, setTimeRange] = useState('all');
  const [categoryFilter, setCategoryFilter] = useState('all');

  // Prepare data for different visualizations
  const analytics = useMemo(() => {
    // Filter transactions based on time range
    const now = new Date();
    const filteredTransactions = transactions.filter(tx => {
      const txDate = new Date(tx.bookingDateTime);
      switch (timeRange) {
        case '7d':
          return (now.getTime() - txDate.getTime()) <= 7 * 24 * 60 * 60 * 1000;
        case '30d':
          return (now.getTime() - txDate.getTime()) <= 30 * 24 * 60 * 60 * 1000;
        case '90d':
          return (now.getTime() - txDate.getTime()) <= 90 * 24 * 60 * 60 * 1000;
        default:
          return true;
      }
    });

    // Sort transactions by date
    const sortedTransactions = [...filteredTransactions].sort(
      (a, b) => new Date(a.bookingDateTime).getTime() - new Date(b.bookingDateTime).getTime()
    );

    // Daily spending trend
    const dailySpending = sortedTransactions.reduce((acc: Record<string, number>, tx) => {
      const date = new Date(tx.bookingDateTime).toLocaleDateString();
      acc[date] = (acc[date] || 0) + parseFloat(tx.transactionAmount.amount);
      return acc;
    }, {});

    // Category distribution
    const categoryDistribution = sortedTransactions.reduce((acc: Record<string, number>, tx) => {
      acc[tx.category] = (acc[tx.category] || 0) + parseFloat(tx.transactionAmount.amount);
      return acc;
    }, {});

    // Monthly totals with trend
    const monthlyData = sortedTransactions.reduce((acc: Record<string, any>, tx) => {
      const month = new Date(tx.bookingDateTime).toLocaleDateString('en-US', { year: 'numeric', month: 'short' });
      if (!acc[month]) {
        acc[month] = {
          total: 0,
          count: 0,
          average: 0,
          transactions: []
        };
      }
      acc[month].total += parseFloat(tx.transactionAmount.amount);
      acc[month].count += 1;
      acc[month].transactions.push(tx);
      acc[month].average = acc[month].total / acc[month].count;
      return acc;
    }, {});

    // Day of week analysis
    const dowAnalysis = sortedTransactions.reduce((acc: Record<string, number>, tx) => {
      const dow = new Date(tx.bookingDateTime).toLocaleDateString('en-US', { weekday: 'long' });
      acc[dow] = (acc[dow] || 0) + parseFloat(tx.transactionAmount.amount);
      return acc;
    }, {});

    // Time of day analysis
    const todAnalysis = sortedTransactions.reduce((acc: Record<string, number>, tx) => {
      const hour = new Date(tx.bookingDateTime).getHours();
      const timeSlot = `${hour.toString().padStart(2, '0')}:00`;
      acc[timeSlot] = (acc[timeSlot] || 0) + parseFloat(tx.transactionAmount.amount);
      return acc;
    }, {});

    // Transaction size distribution
    const sizeDistribution = sortedTransactions.reduce((acc: Record<string, number>, tx) => {
      const amount = parseFloat(tx.transactionAmount.amount);
      let range = '0-100';
      if (amount > 100 && amount <= 500) range = '101-500';
      else if (amount > 500 && amount <= 1000) range = '501-1000';
      else if (amount > 1000 && amount <= 5000) range = '1001-5000';
      else if (amount > 5000) range = '5000+';
      acc[range] = (acc[range] || 0) + 1;
      return acc;
    }, {});

    // Calculate trends and patterns
    const trends = {
      totalVolume: sortedTransactions.length,
      totalValue: sortedTransactions.reduce((sum, tx) => sum + parseFloat(tx.transactionAmount.amount), 0),
      averageValue: sortedTransactions.length > 0 
        ? sortedTransactions.reduce((sum, tx) => sum + parseFloat(tx.transactionAmount.amount), 0) / sortedTransactions.length 
        : 0,
      categoryGrowth: Object.entries(monthlyData).reduce((acc: Record<string, any>, [month, data]: [string, any]) => {
        Object.entries(data.transactions.reduce((catAcc: Record<string, number>, tx: Transaction) => {
          catAcc[tx.category] = (catAcc[tx.category] || 0) + parseFloat(tx.transactionAmount.amount);
          return catAcc;
        }, {})).forEach(([category, amount]) => {
          if (!acc[category]) acc[category] = [];
          acc[category].push({ month, amount });
        });
        return acc;
      }, {})
    };

    return {
      dailySpending: Object.entries(dailySpending).map(([date, amount]) => ({ date, amount })),
      categoryDistribution: Object.entries(categoryDistribution).map(([name, value]) => ({ name, value })),
      monthlyTotals: Object.entries(monthlyData).map(([month, data]: [string, any]) => ({ 
        month, 
        total: data.total,
        average: data.average,
        count: data.count
      })),
      dowAnalysis: Object.entries(dowAnalysis).map(([day, amount]) => ({ day, amount })),
      todAnalysis: Object.entries(todAnalysis).map(([hour, amount]) => ({ hour, amount })),
      sizeDistribution: Object.entries(sizeDistribution).map(([range, count]) => ({ range, count })),
      trends
    };
  }, [transactions, timeRange, categoryFilter]);

  // Colors for charts
  const COLORS = [
    '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d',
    '#a4de6c', '#d0ed57', '#ffc658', '#ff7300', '#8dd1e1', '#a4de6c'
  ];

  const exportAnalytics = () => {
    const data = {
      summary: {
        totalTransactions: analytics.trends.totalVolume,
        totalValue: analytics.trends.totalValue,
        averageTransactionValue: analytics.trends.averageValue,
        timeRange,
        exportDate: new Date().toISOString()
      },
      analytics: {
        dailySpending: analytics.dailySpending,
        categoryDistribution: analytics.categoryDistribution,
        monthlyTotals: analytics.monthlyTotals,
        dayOfWeekAnalysis: analytics.dowAnalysis,
        timeOfDayAnalysis: analytics.todAnalysis,
        transactionSizeDistribution: analytics.sizeDistribution,
        categoryGrowthTrends: analytics.trends.categoryGrowth
      },
      rawTransactions: transactions
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transaction-analytics-${new Date().toISOString()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-[#261436]">Transaction Analytics</h2>
        <div className="flex gap-4">
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger className="w-[180px]">
              <SelectValue placeholder="Time Range" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Time</SelectItem>
              <SelectItem value="7d">Last 7 Days</SelectItem>
              <SelectItem value="30d">Last 30 Days</SelectItem>
              <SelectItem value="90d">Last 90 Days</SelectItem>
            </SelectContent>
          </Select>
          <Button onClick={exportAnalytics} className="bg-[#261436] text-white">
            <Download className="mr-2 h-4 w-4" />
            Export Analytics
          </Button>
        </div>
      </div>

      {/* Summary Cards
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-[#261436]">Total Transactions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{analytics.trends.totalVolume}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-[#261436]">Total Value</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">${analytics.trends.totalValue.toFixed(2)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-[#261436]">Average Transaction</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">${analytics.trends.averageValue.toFixed(2)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-[#261436]">Active Categories</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{analytics.categoryDistribution.length}</p>
          </CardContent>
        </Card>
      </div> */}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Spending Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">Spending Trend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={analytics.dailySpending}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="amount" stroke="#8884d8" />
                  <Area type="monotone" dataKey="amount" fill="#8884d8" fillOpacity={0.3} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Category Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">Category Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={analytics.categoryDistribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={100}
                    label
                  >
                    {analytics.categoryDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Day of Week Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">Day of Week Patterns</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.dowAnalysis}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="amount" fill="#82ca9d" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Time of Day Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">Time of Day Patterns</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analytics.todAnalysis}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="hour" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="amount" stroke="#8884d8" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Transaction Size Distribution */}
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">Transaction Size Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.sizeDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Monthly Trend Analysis */}
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">Monthly Trend Analysis</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={analytics.monthlyTotals}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Bar yAxisId="left" dataKey="total" fill="#8884d8" />
                  <Line yAxisId="right" type="monotone" dataKey="average" stroke="#82ca9d" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
} 