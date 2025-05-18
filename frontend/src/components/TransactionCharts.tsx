import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
import { useMemo, useCallback } from "react";
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "./ui/chart";
import { Transaction } from "@/types";

interface CategoryData {
  count: number;
  amount: number;
}

interface ChartConfig {
  [key: string]: {
    label: string;
    color: string;
  };
}

interface TransactionChartsProps {
  transactions: Transaction[];
  personaType: string;
}

export function TransactionCharts({
  transactions,
  personaType,
}: TransactionChartsProps) {
  const categoryDistribution = useMemo(
    () =>
      transactions.reduce((acc: Record<string, CategoryData>, tx) => {
        const category = tx.category;
        if (!acc[category]) {
          acc[category] = { count: 0, amount: 0 };
        }
        acc[category].count++;
        acc[category].amount += parseFloat(tx.transactionAmount.amount);
        return acc;
      }, {}),
    [transactions]
  );

  const getCriticalCategory = useCallback(() => {
    switch (personaType.toLowerCase()) {
      case "gambling addict":
        return "Gambling";
      case "shopping addict":
        return "Shopping";
      case "crypto enthusiast":
        return "Crypto";
      case "money mule":
        return "Transfer";
      default:
        return null;
    }
  }, [personaType]);

  const criticalCategory = getCriticalCategory();
  const criticalData = useMemo(
    () =>
      criticalCategory
        ? transactions
            .filter((tx) => tx.category === criticalCategory)
            .map((tx) => ({
              date: new Date(tx.bookingDateTime).toLocaleDateString(),
              amount: parseFloat(tx.transactionAmount.amount),
            }))
            .sort(
              (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
            )
        : [],
    [transactions, criticalCategory]
  );

  const chartConfig = useMemo<ChartConfig>(
    () => ({
      value: {
        label: "Transactions",
        color: "hsl(var(--chart-1))",
      },
      Crypto: {
        label: "Crypto",
        color: "hsl(var(--chart-1))",
      },
      Shopping: {
        label: "Shopping",
        color: "hsl(var(--chart-2))",
      },
      Transport: {
        label: "Transport",
        color: "hsl(var(--chart-3))",
      },
      Utilities: {
        label: "Utilities",
        color: "hsl(var(--chart-4))",
      },
      Dining: {
        label: "Dining",
        color: "hsl(var(--chart-5))",
      },
      Subscriptions: {
        label: "Subscriptions",
        color: "hsl(var(--chart-6))",
      },
      ATM: {
        label: "ATM Withdrawals",
        color: "hsl(var(--chart-7))",
      },
      Refunds: {
        label: "Refunds",
        color: "hsl(var(--chart-8))",
      },
      Groceries: {
        label: "Groceries",
        color: "hsl(var(--chart-9))",
      },
      Salary: {
        label: "Salary",
        color: "hsl(var(--chart-10))",
      },
    }),
    []
  );

  const pieData = useMemo(
    () =>
      Object.entries(categoryDistribution).map(
        ([category, data]: [string, CategoryData]) => ({
          name: category,
          value: data.count,
          amount: data.amount,
          fill:
            chartConfig[category as keyof typeof chartConfig]?.color ||
            "hsl(var(--chart-1))",
        })
      ),
    [categoryDistribution, chartConfig]
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <Card className="flex flex-col">
        <CardHeader className="items-center pb-0">
          <CardTitle>Category Distribution</CardTitle>
          <CardDescription>Transaction Categories</CardDescription>
        </CardHeader>
        <CardContent className="flex-1 pb-0">
          <ChartContainer
            config={chartConfig}
            className="mx-auto aspect-square max-h-[250px] [&_.recharts-text]:fill-background"
          >
            <PieChart>
              <ChartTooltip
                content={<ChartTooltipContent nameKey="value" hideLabel />}
              />
              <Pie
                data={pieData}
                dataKey="value"
                cx="50%"
                cy="50%"
                outerRadius={100}
                isAnimationActive={false}
                label={undefined}
                labelLine={false}
                nameKey={undefined}
              >
                {pieData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={`hsl(var(--chart-${(index % 10) + 1}))`}
                  />
                ))}
              </Pie>
            </PieChart>
          </ChartContainer>
        </CardContent>
      </Card>

      {criticalCategory && (
        <Card>
          <CardHeader>
            <CardTitle className="text-[#261436]">
              {criticalCategory} Transactions Over Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            {criticalData.length > 0 ? (
              <div className="h-[400px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={criticalData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="date"
                      angle={-45}
                      textAnchor="end"
                      height={60}
                      interval={Math.ceil(criticalData.length / 10)}
                    />
                    <YAxis />
                    <Tooltip
                      formatter={(value: number) => [
                        `${value.toFixed(2)} EUR`,
                        "Amount",
                      ]}
                    />
                    <Bar dataKey="amount" fill="#8884d8" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-[#261436] opacity-70">
                No {criticalCategory.toLowerCase()} transactions in this batch.
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
