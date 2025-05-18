import { useMemo, useState } from "react";
import {
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
  ComposedChart,
  Scatter,
  ScatterChart,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Download, Settings } from "lucide-react";
import { Transaction } from "@/types";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "src/components/ui/carousel";
import { DistributionEditor } from "./DistributionEditor";

interface BatchAnalyticsProps {
  transactions: Transaction[];
}

export function BatchAnalytics({ 
  transactions
}: BatchAnalyticsProps) {
  const analytics = useMemo(() => {
    // Sort transactions by date
    const sortedTransactions = [...transactions].sort(
      (a, b) =>
        new Date(a.bookingDateTime).getTime() -
        new Date(b.bookingDateTime).getTime()
    );

    // Separate income and expense transactions
    const { income, expenses } = sortedTransactions.reduce(
      (acc, tx) => {
        const amount = parseFloat(tx.transactionAmount.amount);
        if (amount >= 0) {
          acc.income.push(tx);
        } else {
          acc.expenses.push(tx);
        }
        return acc;
      },
      { income: [] as Transaction[], expenses: [] as Transaction[] }
    );

    // Calculate amounts for metrics
    const amounts = sortedTransactions.map((tx) =>
      parseFloat(tx.transactionAmount.amount)
    );
    const totalIncome = income.reduce((sum, tx) => sum + parseFloat(tx.transactionAmount.amount), 0);
    const totalExpenses = Math.abs(expenses.reduce((sum, tx) => sum + parseFloat(tx.transactionAmount.amount), 0));
    const netAmount = totalIncome - totalExpenses;

    // Category distribution with separate income/expense tracking
    const categoryDistribution = sortedTransactions.reduce(
      (acc: Record<string, { count: number; amount: number; isIncome: boolean }>, tx) => {
        const amount = parseFloat(tx.transactionAmount.amount);
        if (!acc[tx.category]) {
          acc[tx.category] = { count: 0, amount: 0, isIncome: amount >= 0 };
        }
        acc[tx.category].count += 1;
        acc[tx.category].amount += amount;
        return acc;
      },
      {}
    );

    // Sort categories by absolute amount in descending order
    const sortedCategories = Object.entries(categoryDistribution)
      .map(([category, data]) => ({
        name: category,
        value: Math.abs(data.amount),
        actualAmount: data.amount,
        count: data.count,
        isIncome: data.isIncome,
        percentage: ((data.count / sortedTransactions.length) * 100).toFixed(1),
        amountPercentage: ((Math.abs(data.amount) / (data.isIncome ? totalIncome : totalExpenses)) * 100).toFixed(1)
      }))
      .sort((a, b) => Math.abs(b.actualAmount) - Math.abs(a.actualAmount));

    // Daily transaction volume and amounts with income/expense split
    const dailyMetrics = sortedTransactions.reduce(
      (acc: Record<string, any>, tx) => {
        const date = new Date(tx.bookingDateTime).toLocaleDateString();
        const amount = parseFloat(tx.transactionAmount.amount);
        if (!acc[date]) {
          acc[date] = { 
            date, 
            count: 0, 
            income: 0, 
            expenses: 0,
            net: 0 
          };
        }
        acc[date].count += 1;
        if (amount >= 0) {
          acc[date].income += amount;
        } else {
          acc[date].expenses += Math.abs(amount);
        }
        acc[date].net = acc[date].income - acc[date].expenses;
        return acc;
      },
      {}
    );

    // Transaction size analysis with income/expense flag
    const transactionSizes = sortedTransactions.map((tx) => {
      const amount = parseFloat(tx.transactionAmount.amount);
      return {
        amount: Math.abs(amount),
        actualAmount: amount,
        category: tx.category,
        date: new Date(tx.bookingDateTime).toLocaleDateString(),
        isIncome: amount >= 0
      };
    });

    // Hour of day distribution with income/expense split
    const hourlyDistribution = sortedTransactions.reduce(
      (acc: Record<number, { income: number; expenses: number }>, tx) => {
        const hour = new Date(tx.bookingDateTime).getHours();
        const amount = parseFloat(tx.transactionAmount.amount);
        if (!acc[hour]) {
          acc[hour] = { income: 0, expenses: 0 };
        }
        if (amount >= 0) {
          acc[hour].income += 1;
        } else {
          acc[hour].expenses += 1;
        }
        return acc;
      },
      {}
    );

    return {
      categoryDistribution: sortedCategories,
      dailyMetrics: Object.values(dailyMetrics),
      transactionSizes,
      hourlyDistribution: Array.from({ length: 24 }, (_, hour) => ({
        hour: `${hour.toString().padStart(2, "0")}:00`,
        income: hourlyDistribution[hour]?.income || 0,
        expenses: hourlyDistribution[hour]?.expenses || 0,
      })),
      totalIncome,
      totalExpenses,
      netAmount,
      incomeCount: income.length,
      expenseCount: expenses.length
    };
  }, [transactions]);

  // Calculate key metrics
  const metrics = useMemo(() => {
    return {
      totalTransactions: transactions.length,
      totalIncome: analytics.totalIncome,
      totalExpenses: analytics.totalExpenses,
      netAmount: analytics.netAmount,
      incomeCount: analytics.incomeCount,
      expenseCount: analytics.expenseCount,
      numCategories: new Set(transactions.map((tx) => tx.category)).size,
      avgIncomeSize: analytics.incomeCount > 0 ? analytics.totalIncome / analytics.incomeCount : 0,
      avgExpenseSize: analytics.expenseCount > 0 ? analytics.totalExpenses / analytics.expenseCount : 0,
    };
  }, [transactions, analytics]);

  const COLORS = [
    "#FF8042",
    "#0088FE",
    "#00C49F",
    "#FFBB28",
    "#8884d8",
    "#82ca9d",
    "#ffc658",
    "#8dd1e1",
    "#a4de6c",
    "#d0ed57",
  ];

  return (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-3 md:grid-cols-3 gap-2">
        <Card className="p-1">
          <CardHeader className="p-1">
            <CardTitle className="text-xs font-medium text-[#261436]">
              Total Transactions
            </CardTitle>
          </CardHeader>
          <CardContent className="p-1">
            <p className="text-lg font-bold text-[#261436]">
              {metrics.totalTransactions}
              <span className="text-xs font-normal ml-2">
                ({metrics.incomeCount} income, {metrics.expenseCount} expenses)
              </span>
            </p>
          </CardContent>
        </Card>
        <Card className="p-1">
          <CardHeader className="p-1">
            <CardTitle className="text-xs font-medium text-green-600">
              Total Income
            </CardTitle>
          </CardHeader>
          <CardContent className="p-1">
            <p className="text-lg font-bold text-green-600">
              +{metrics.totalIncome.toFixed(2)} EUR
            </p>
          </CardContent>
        </Card>
        <Card className="p-1">
          <CardHeader className="p-1">
            <CardTitle className="text-xs font-medium text-red-600">
              Total Expenses
            </CardTitle>
          </CardHeader>
          <CardContent className="p-1">
            <p className="text-lg font-bold text-red-600">
              -{metrics.totalExpenses.toFixed(2)} EUR
            </p>
          </CardContent>
        </Card>
        <Card className="p-1">
          <CardHeader className="p-1">
            <CardTitle className="text-xs font-medium text-[#261436]">
              Net Amount
            </CardTitle>
          </CardHeader>
          <CardContent className="p-1">
            <p className={`text-lg font-bold ${metrics.netAmount >= 0 ? 'text-green-600' : 'text-red-600'}`}>
              {metrics.netAmount >= 0 ? '+' : ''}{metrics.netAmount.toFixed(2)} EUR
            </p>
          </CardContent>
        </Card>
        <Card className="p-1">
          <CardHeader className="p-1">
            <CardTitle className="text-xs font-medium text-[#261436]">
              Categories
            </CardTitle>
          </CardHeader>
          <CardContent className="p-1">
            <p className="text-lg font-bold text-[#261436]">
              {metrics.numCategories}
            </p>
          </CardContent>
        </Card>
        <Card className="p-1">
          <CardHeader className="p-1">
            <CardTitle className="text-xs font-medium text-[#261436]">
              Average Transaction
            </CardTitle>
          </CardHeader>
          <CardContent className="p-1 space-y-1">
            <p className="text-sm font-medium text-green-600">
              Income: +{metrics.avgIncomeSize.toFixed(2)} EUR
            </p>
            <p className="text-sm font-medium text-red-600">
              Expense: -{metrics.avgExpenseSize.toFixed(2)} EUR
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Charts Carousel */}
      <div className="w-full">
        <Carousel className="w-full">
          <CarouselContent>
            {/* Category Distribution */}
            <CarouselItem>
              <Card className="p-4">
                <CardHeader className="p-4 pb-2">
                  <CardTitle className="text-lg text-[#261436]">
                    Category Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="h-[400px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={analytics.categoryDistribution}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            outerRadius={150}
                            isAnimationActive={false}
                            label={false}
                            labelLine={false}
                          >
                            {analytics.categoryDistribution.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={COLORS[index % COLORS.length]}
                              />
                            ))}
                          </Pie>
                          <Tooltip
                            formatter={(value: number, name: string, props: any) => {
                              const entry = props.payload;
                              return [`${entry.actualAmount >= 0 ? '+' : ''}${entry.actualAmount.toFixed(2)} EUR`, name];
                            }}
                          />
                          <Legend />
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="flex flex-col h-[400px]">
                      <h3 className="font-semibold mb-2 p-4">Category Distribution</h3>
                      <div className="overflow-y-auto flex-1 pr-2">
                        <div className="grid gap-3 p-4 pt-0">
                          {analytics.categoryDistribution.map((category, index) => (
                            <div key={category.name} className="space-y-2 pb-2 border-b last:border-b-0">
                              <div className="flex items-center gap-2">
                                <div 
                                  className="w-3 h-3 rounded-full flex-shrink-0" 
                                  style={{ 
                                    backgroundColor: COLORS[index % COLORS.length]
                                  }}
                                />
                                <span className="flex-1 font-medium">{category.name}</span>
                              </div>
                              <div className="grid gap-1 text-sm pl-5">
                                <div className="text-gray-600">
                                  {category.count} transactions ({category.percentage}% of total count)
                                </div>
                                <div className={category.isIncome ? 'text-green-600' : 'text-red-600'}>
                                  {category.isIncome ? '+' : '-'}{Math.abs(category.actualAmount).toFixed(2)} EUR
                                  <span className="text-gray-600 ml-1">
                                    ({category.amountPercentage}% of {category.isIncome ? 'income' : 'expenses'})
                                  </span>
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </CarouselItem>

            {/* Daily Transaction Volume */}
            <CarouselItem>
              <Card className="p-4">
                <CardHeader className="p-4 pb-2">
                  <CardTitle className="text-lg text-[#261436]">
                    Daily Transaction Volume
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={analytics.dailyMetrics}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" />
                        <YAxis yAxisId="left" />
                        <YAxis yAxisId="right" orientation="right" />
                        <Tooltip />
                        <Legend />
                        <Bar
                          yAxisId="left"
                          dataKey="income"
                          fill="#22c55e"
                          name="Income"
                          stackId="a"
                        />
                        <Bar
                          yAxisId="left"
                          dataKey="expenses"
                          fill="#ef4444"
                          name="Expenses"
                          stackId="a"
                        />
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="net"
                          stroke="#8884d8"
                          name="Net Amount"
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </CarouselItem>

            {/* Transaction Size Distribution */}
            <CarouselItem>
              <Card className="p-4">
                <CardHeader className="p-4 pb-2">
                  <CardTitle className="text-lg text-[#261436]">
                    Transaction Size Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="date" name="Date" />
                        <YAxis dataKey="amount" name="Amount" />
                        <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                        <Legend />
                        {analytics.categoryDistribution.map(
                          (category, index) => (
                            <Scatter
                              key={category.name}
                              name={category.name}
                              data={analytics.transactionSizes.filter(
                                (t) => t.category === category.name
                              )}
                              fill={COLORS[index % COLORS.length]}
                            />
                          )
                        )}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </CarouselItem>

            {/* Hourly Distribution */}
            <CarouselItem>
              <Card className="p-4">
                <CardHeader className="p-4 pb-2">
                  <CardTitle className="text-lg text-[#261436]">
                    Transaction Time Distribution
                  </CardTitle>
                </CardHeader>
                <CardContent className="p-0">
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={analytics.hourlyDistribution}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="hour" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="income"
                          fill="#22c55e"
                          name="Income"
                          stackId="a"
                        />
                        <Bar
                          dataKey="expenses"
                          fill="#ef4444"
                          name="Expenses"
                          stackId="a"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </CarouselItem>
          </CarouselContent>
          <CarouselPrevious className="bg-[#261436] text-white hover:bg-[#372851]" />
          <CarouselNext className="bg-[#261436] text-white hover:bg-[#372851]" />
        </Carousel>
      </div>
    </div>
  );
}
