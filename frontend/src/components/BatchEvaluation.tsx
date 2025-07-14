import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";

interface EvaluationMetrics {
  inception_score: number;
  fid_score: number;
  diversity_score: number;
  realism_score: number;
  overall_score: number;
}

interface BatchEvaluationProps {
  batch_id: number;
  persona_name: string;
  transaction_count: number;
  evaluation_metrics: EvaluationMetrics;
}

export function BatchEvaluation({
  batch_id,
  persona_name,
  transaction_count,
  evaluation_metrics,
}: BatchEvaluationProps) {
  const getQualityColor = (score: number, metric: string) => {
    if (metric === 'fid_score') {
      // Lower FID is better
      if (score < 50) return "text-green-600";
      if (score < 100) return "text-yellow-600";
      return "text-red-600";
    } else {
      // Higher scores are better for other metrics
      if (score > 0.7) return "text-green-600";
      if (score > 0.4) return "text-yellow-600";
      return "text-red-600";
    }
  };

  const getQualityText = (score: number, metric: string) => {
    if (metric === 'fid_score') {
      if (score < 50) return "Excellent";
      if (score < 100) return "Good";
      return "Needs Improvement";
    } else {
      if (score > 0.7) return "Excellent";
      if (score > 0.4) return "Good";
      return "Needs Improvement";
    }
  };

  const formatScore = (score: number, metric: string) => {
    if (metric === 'fid_score') {
      return score.toFixed(2);
    } else {
      return (score * 100).toFixed(1) + '%';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2 mb-4">
        <h2 className="text-2xl font-bold text-[#261436]">Batch Evaluation Metrics</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Inception Score */}
        <Card className="border-2 border-gray-100">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg">
                Inception Score
              </CardTitle>
              <span className="text-sm font-medium">{getQualityText(evaluation_metrics.inception_score, 'inception_score')}</span>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Quality Score</span>
                <span className={`font-bold ${getQualityColor(evaluation_metrics.inception_score, 'inception_score')}`}>
                  {formatScore(evaluation_metrics.inception_score, 'inception_score')}
                </span>
              </div>
              <Progress 
                value={evaluation_metrics.inception_score * 100} 
                className="h-2"
              />
              <p className="text-xs text-gray-500">
                Measures the quality and diversity of generated transactions. Higher scores indicate better generation.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Fréchet Inception Distance */}
        <Card className="border-2 border-gray-100">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <TrendingUp className="h-5 w-5 text-purple-600" />
                Fréchet Inception Distance
              </CardTitle>
              {getQualityBadge(evaluation_metrics.fid_score, 'fid_score')}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Distance Score</span>
                <span className={`font-bold ${getQualityColor(evaluation_metrics.fid_score, 'fid_score')}`}>
                  {formatScore(evaluation_metrics.fid_score, 'fid_score')}
                </span>
              </div>
              <Progress 
                value={Math.max(0, 100 - evaluation_metrics.fid_score)} 
                className="h-2"
              />
              <p className="text-xs text-gray-500">
                Measures similarity to real transaction patterns. Lower scores indicate better realism.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Diversity Score */}
        <Card className="border-2 border-gray-100">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <Award className="h-5 w-5 text-green-600" />
                Diversity Score
              </CardTitle>
              {getQualityBadge(evaluation_metrics.diversity_score, 'diversity_score')}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Variety Score</span>
                <span className={`font-bold ${getQualityColor(evaluation_metrics.diversity_score, 'diversity_score')}`}>
                  {formatScore(evaluation_metrics.diversity_score, 'diversity_score')}
                </span>
              </div>
              <Progress 
                value={evaluation_metrics.diversity_score * 100} 
                className="h-2"
              />
              <p className="text-xs text-gray-500">
                Measures the variety in categories, merchants, amounts, and timing of transactions.
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Realism Score */}
        <Card className="border-2 border-gray-100">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-lg flex items-center gap-2">
                <Trophy className="h-5 w-5 text-orange-600" />
                Realism Score
              </CardTitle>
              {getQualityBadge(evaluation_metrics.realism_score, 'realism_score')}
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">Realism Score</span>
                <span className={`font-bold ${getQualityColor(evaluation_metrics.realism_score, 'realism_score')}`}>
                  {formatScore(evaluation_metrics.realism_score, 'realism_score')}
                </span>
              </div>
              <Progress 
                value={evaluation_metrics.realism_score * 100} 
                className="h-2"
              />
              <p className="text-xs text-gray-500">
                Measures how realistic the transaction patterns are compared to real-world behavior.
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Overall Score */}
      <Card className="border-2 border-blue-200 bg-blue-50">
        <CardHeader>
          <CardTitle className="text-xl flex items-center gap-2 text-blue-800">
            <Trophy className="h-6 w-6" />
            Overall Quality Score
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-lg font-semibold text-blue-800">Combined Quality</span>
              <span className={`text-2xl font-bold ${getQualityColor(evaluation_metrics.overall_score, 'overall_score')}`}>
                {formatScore(evaluation_metrics.overall_score, 'overall_score')}
              </span>
            </div>
            <Progress 
              value={evaluation_metrics.overall_score * 100} 
              className="h-3 bg-blue-200"
            />
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Batch ID:</span>
                <span className="ml-2 font-medium">{batch_id}</span>
              </div>
              <div>
                <span className="text-gray-600">Persona:</span>
                <span className="ml-2 font-medium">{persona_name}</span>
              </div>
              <div>
                <span className="text-gray-600">Transactions:</span>
                <span className="ml-2 font-medium">{transaction_count}</span>
              </div>
              <div>
                <span className="text-gray-600">Evaluation Date:</span>
                <span className="ml-2 font-medium">{new Date().toLocaleDateString()}</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quality Summary */}
      <Card className="border-2 border-gray-100">
        <CardHeader>
          <CardTitle className="text-lg">Quality Assessment Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="font-medium">Inception Score Quality</span>
              {getQualityBadge(evaluation_metrics.inception_score, 'inception_score')}
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="font-medium">FID Quality</span>
              {getQualityBadge(evaluation_metrics.fid_score, 'fid_score')}
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="font-medium">Diversity Quality</span>
              {getQualityBadge(evaluation_metrics.diversity_score, 'diversity_score')}
            </div>
            <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
              <span className="font-medium">Realism Quality</span>
              {getQualityBadge(evaluation_metrics.realism_score, 'realism_score')}
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
} 