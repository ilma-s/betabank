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
  Legend,
  LabelList,
} from "recharts";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import { TrendingUp } from "lucide-react";
import { useMemo, useCallback } from "react";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "./ui/chart";

const COLORS = {
  Crypto: "hsl(var(--chart-1))",
  Shopping: "hsl(var(--chart-2))",
  Transport: "hsl(var(--chart-3))",
  Utilities: "hsl(var(--chart-4))",
  Dining: "hsl(var(--chart-5))",
  Subscriptions: "hsl(var(--chart-6))",
  ATM: "hsl(var(--chart-7))",
  Refunds: "hsl(var(--chart-8))",
  Groceries: "hsl(var(--chart-9))",
  Salary: "hsl(var(--chart-10))",
};

interface TransactionChartsProps {
  transactions: any[];
  personaType: string;
}

export function TransactionCharts({
  transactions,
  personaType,
}: TransactionChartsProps) {
  // Calculate category distribution
  const categoryDistribution = useMemo(
    () =>
      transactions.reduce((acc: any, tx) => {
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

  // Get critical category based on persona
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

  const chartConfig = {
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
  } as const;

  // Convert to array format for charts and calculate percentages
  const totalTransactions = transactions.length;
  const pieData = useMemo(
    () =>
      Object.entries(categoryDistribution).map(
        ([category, data]: [string, any]) => ({
          name: category,
          value: data.count,
          amount: data.amount,
          fill:
            chartConfig[category as keyof typeof chartConfig]?.color ||
            "hsl(var(--chart-1))",
        })
      ),
    [categoryDistribution]
  );

  const totalAmount = useMemo(() => {
    return transactions.reduce(
      (acc, tx) => acc + parseFloat(tx.transactionAmount.amount),
      0
    );
  }, [transactions]);

  const averageAmount = useMemo(() => {
    return totalAmount / totalTransactions;
  }, [totalAmount, totalTransactions]);

  const maxAmount = useMemo(() => {
    return Math.max(
      ...transactions.map((tx) => parseFloat(tx.transactionAmount.amount))
    );
  }, [transactions]);

  const minAmount = useMemo(() => {
    return Math.min(
      ...transactions.map((tx) => parseFloat(tx.transactionAmount.amount))
    );
  }, [transactions]);

  const numCategories = useMemo(() => {
    return Object.keys(categoryDistribution).length;
  }, [categoryDistribution]);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Pie Chart */}
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
                label={undefined} // <- safest to explicitly unset
                labelLine={false} // <- hides lines
                nameKey={undefined} // <- disables name-based labeling
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

      {/* Bar Chart for Critical Category */}
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
