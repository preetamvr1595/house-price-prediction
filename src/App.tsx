import React, { useState } from "react";
import "./App.css";

interface ComparisonRow {
  model: string;
  prediction: string;
  performance: string;
}

interface ApiResponse {
  best_model: string;
  comparison: ComparisonRow[];
}

export default function App() {
  const [form, setForm] = useState({
    size: "",
    bedrooms: "",
    age: "",
    location: "",
  });

  const [result, setResult] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          size: Number(form.size),
          bedrooms: Number(form.bedrooms),
          age: Number(form.age),
          location: Number(form.location),
        }),
      });

      if (!res.ok) throw new Error("Backend error");

      const data: ApiResponse = await res.json();
      setResult(data);
    } catch {
      setError("Backend not running or CORS issue.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <div className="glass-card">
        <header className="header">
          <div className="logo-badge">AI</div>
          <h1>House Price Prediction</h1>
          <p>Estimate property value using advanced machine learning models</p>
        </header>

        <form className="prediction-form" onSubmit={handleSubmit}>
          <div className="input-group">
            <div className="field">
              <label>Square Footage</label>
              <input
                name="size"
                type="number"
                placeholder="e.g. 2500"
                value={form.size}
                onChange={handleChange}
                required
              />
            </div>
            <div className="field">
              <label>Bedrooms</label>
              <input
                name="bedrooms"
                type="number"
                placeholder="e.g. 3"
                value={form.bedrooms}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          <div className="input-group">
            <div className="field">
              <label>Property Age (Years)</label>
              <input
                name="age"
                type="number"
                placeholder="e.g. 10"
                value={form.age}
                onChange={handleChange}
                required
              />
            </div>
            <div className="field">
              <label>Location Rank (1-10)</label>
              <input
                name="location"
                type="number"
                placeholder="e.g. 7"
                value={form.location}
                onChange={handleChange}
                required
              />
            </div>
          </div>

          <button className="submit-btn" type="submit" disabled={loading}>
            {loading ? (
              <span className="loader-container">
                <span className="spinner"></span>
                Processing...
              </span>
            ) : (
              "Generate Prediction"
            )}
          </button>
        </form>

        {error && (
          <div className="error-toast">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        {result && (
          <div className="results-container">
            <div className="best-model-banner">
              <h3>Best Model Recommendation</h3>
              <div className="badge">{result.best_model}</div>
            </div>

            <div className="comparison-grid">
              {result.comparison.map((r, i) => (
                <div key={i} className="model-card">
                  <div className="model-name">{r.model}</div>
                  <div className="prediction-value">{r.prediction}</div>
                  <div className="performance-metric">{r.performance}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
