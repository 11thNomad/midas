import { useState, useMemo, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Area, ComposedChart, Bar, CartesianGrid } from "recharts";

// ============================================================
// Generate realistic NIFTY market data (2022–2025)
// Based on actual market behavior patterns and known events
// ============================================================

function generateMarketData() {
  const data = [];
  let price = 17500;
  let vix = 16;
  let adx = 22;
  let fiiFlow = 0;

  const startDate = new Date(2022, 0, 3);
  const segments = [
    // 2022 Q1: Russia-Ukraine shock — high vol trending down
    { days: 60, trend: -0.08, volBase: 22, adxBase: 28, fiiBase: -1800, event: null },
    // 2022 Q2: Recovery rally — low vol trending up
    { days: 50, trend: 0.06, volBase: 16, adxBase: 26, fiiBase: 500, event: null },
    // 2022 Q3: Ranging — low vol, no direction
    { days: 65, trend: 0.01, volBase: 13, adxBase: 16, fiiBase: 200, event: null },
    // 2022 Q4: Rally into year end
    { days: 60, trend: 0.05, volBase: 14, adxBase: 24, fiiBase: 800, event: null },
    // 2023 Q1: Adani crisis + budget — high vol choppy
    { days: 45, trend: -0.04, volBase: 19, adxBase: 18, fiiBase: -2500, event: "Adani Short Report" },
    // 2023 Q1-Q2: Recovery — moderate trending
    { days: 55, trend: 0.04, volBase: 13, adxBase: 23, fiiBase: 600, event: null },
    // 2023 Q3: Summer range — perfect for iron condors
    { days: 70, trend: 0.005, volBase: 11, adxBase: 14, fiiBase: 300, event: null },
    // 2023 Q4: Rally to ATH
    { days: 65, trend: 0.07, volBase: 12, adxBase: 27, fiiBase: 1200, event: null },
    // 2024 Q1: Continued rally
    { days: 50, trend: 0.05, volBase: 13, adxBase: 25, fiiBase: 900, event: null },
    // 2024 Q2: Election volatility
    { days: 40, trend: -0.06, volBase: 21, adxBase: 30, fiiBase: -3200, event: "Lok Sabha Elections" },
    // 2024 Q3: Post-election recovery
    { days: 55, trend: 0.04, volBase: 14, adxBase: 22, fiiBase: 700, event: null },
    // 2024 Q4: Sideways chop
    { days: 60, trend: -0.01, volBase: 15, adxBase: 17, fiiBase: -400, event: null },
    // 2024 Q4-2025 Q1: FII selloff
    { days: 50, trend: -0.07, volBase: 19, adxBase: 26, fiiBase: -4000, event: "FII Selloff" },
    // 2025 Q1-Q2: Recovery
    { days: 55, trend: 0.03, volBase: 14, adxBase: 21, fiiBase: 500, event: null },
    // 2025 Q2-Q3: Calm trending
    { days: 60, trend: 0.04, volBase: 12, adxBase: 26, fiiBase: 800, event: null },
  ];

  let dayCount = 0;
  segments.forEach((seg) => {
    const eventDay = seg.event ? Math.floor(seg.days * 0.3) : -1;
    for (let i = 0; i < seg.days; i++) {
      const currentDate = new Date(startDate);
      currentDate.setDate(startDate.getDate() + dayCount);

      // Skip weekends
      const dow = currentDate.getDay();
      if (dow === 0 || dow === 6) {
        dayCount++;
        i--;
        continue;
      }

      const noise = (Math.random() - 0.5) * 2;
      const trendComponent = seg.trend * (price / 100);
      price = price + trendComponent + noise * (vix / 10) * 15;
      price = Math.max(price, 14000);

      vix = seg.volBase + (Math.random() - 0.5) * 4 + (i === eventDay ? 5 : 0);
      vix = Math.max(8, Math.min(35, vix));

      adx = seg.adxBase + (Math.random() - 0.5) * 6;
      adx = Math.max(8, Math.min(50, adx));

      fiiFlow = seg.fiiBase + (Math.random() - 0.5) * 1500;

      // Compute regime
      const lowVol = vix < 14;
      const highVol = vix >= 18;
      const trending = adx >= 25;
      let regime;
      if (lowVol && trending) regime = "LOW_VOL_TRENDING";
      else if (lowVol && !trending) regime = "LOW_VOL_RANGING";
      else if (highVol && trending) regime = "HIGH_VOL_TRENDING";
      else if (highVol) regime = "HIGH_VOL_CHOPPY";
      else if (trending) regime = "TRANSITIONAL_TRENDING";
      else regime = "TRANSITIONAL_RANGING";

      // Override: heavy FII selling
      const fii3d = data.length >= 3
        ? data.slice(-2).reduce((s, d) => s + d.fiiFlow, fiiFlow)
        : fiiFlow * 3;
      if (fii3d < -6000 && !highVol) {
        regime = "HIGH_VOL_CHOPPY";
      }

      const pcr = 0.85 + (Math.random() - 0.5) * 0.6 + (highVol ? 0.2 : 0);

      data.push({
        date: currentDate.toISOString().split("T")[0],
        dateShort: `${currentDate.toLocaleString("default", { month: "short" })} ${currentDate.getDate()}`,
        month: `${currentDate.getFullYear()}-${String(currentDate.getMonth() + 1).padStart(2, "0")}`,
        year: currentDate.getFullYear(),
        price: Math.round(price * 100) / 100,
        vix: Math.round(vix * 100) / 100,
        adx: Math.round(adx * 100) / 100,
        fiiFlow: Math.round(fiiFlow),
        fii3d: Math.round(fii3d),
        pcr: Math.round(pcr * 100) / 100,
        regime,
        event: i === eventDay ? seg.event : null,
      });

      dayCount++;
    }
  });

  return data;
}

// ============================================================
// Regime colors and metadata
// ============================================================

const REGIME_META = {
  LOW_VOL_RANGING: {
    color: "#22c55e",
    bg: "rgba(34,197,94,0.12)",
    label: "Low Vol Ranging",
    short: "LVR",
    icon: "◆",
    strategies: "Iron condors, credit spreads, mean reversion",
    description: "Quiet, mean-reverting. Options selling paradise.",
  },
  LOW_VOL_TRENDING: {
    color: "#3b82f6",
    bg: "rgba(59,130,246,0.12)",
    label: "Low Vol Trending",
    short: "LVT",
    icon: "▲",
    strategies: "Trend following, directional options",
    description: "Calm but directional. Follow the trend.",
  },
  HIGH_VOL_TRENDING: {
    color: "#f59e0b",
    bg: "rgba(245,158,11,0.12)",
    label: "High Vol Trending",
    short: "HVT",
    icon: "⚡",
    strategies: "Trend following (wide stops), momentum",
    description: "Volatile and directional. Big moves.",
  },
  HIGH_VOL_CHOPPY: {
    color: "#ef4444",
    bg: "rgba(239,68,68,0.12)",
    label: "High Vol Choppy",
    short: "HVC",
    icon: "☠",
    strategies: "Sit out or minimal size",
    description: "The killing field. Almost nothing works.",
  },
  TRANSITIONAL_TRENDING: {
    color: "#8b5cf6",
    bg: "rgba(139,92,246,0.12)",
    label: "Transitional (Trending)",
    short: "T-T",
    icon: "◎",
    strategies: "Reduced size, lean toward trend",
    description: "VIX in middle zone, but directional.",
  },
  TRANSITIONAL_RANGING: {
    color: "#6b7280",
    bg: "rgba(107,114,128,0.12)",
    label: "Transitional (Ranging)",
    short: "T-R",
    icon: "◎",
    strategies: "Reduced size, cautious selling",
    description: "VIX in middle zone, no clear direction.",
  },
};

// ============================================================
// Custom tooltip
// ============================================================

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  const meta = REGIME_META[d.regime] || {};

  return (
    <div style={{
      background: "#0f0f0f",
      border: `1px solid ${meta.color || "#333"}`,
      borderRadius: 6,
      padding: "10px 14px",
      fontSize: 12,
      fontFamily: "'JetBrains Mono', monospace",
      color: "#e5e5e5",
      minWidth: 200,
      boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
    }}>
      <div style={{ color: "#999", marginBottom: 6, fontSize: 11 }}>{d.date}</div>
      <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>
        ₹{d.price?.toLocaleString("en-IN", { maximumFractionDigits: 0 })}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px", marginBottom: 8 }}>
        <span style={{ color: "#888" }}>VIX</span>
        <span style={{ color: d.vix >= 18 ? "#ef4444" : d.vix < 14 ? "#22c55e" : "#f59e0b" }}>
          {d.vix?.toFixed(1)}
        </span>
        <span style={{ color: "#888" }}>ADX</span>
        <span style={{ color: d.adx >= 25 ? "#3b82f6" : "#888" }}>
          {d.adx?.toFixed(1)}
        </span>
        <span style={{ color: "#888" }}>FII 3d</span>
        <span style={{ color: d.fii3d < -3000 ? "#ef4444" : d.fii3d > 1000 ? "#22c55e" : "#888" }}>
          {d.fii3d > 0 ? "+" : ""}{(d.fii3d / 1000).toFixed(1)}K Cr
        </span>
        <span style={{ color: "#888" }}>PCR</span>
        <span>{d.pcr?.toFixed(2)}</span>
      </div>
      <div style={{
        padding: "6px 8px",
        borderRadius: 4,
        background: meta.bg,
        border: `1px solid ${meta.color}44`,
        color: meta.color,
        fontWeight: 600,
        fontSize: 11,
        textAlign: "center",
      }}>
        {meta.icon} {meta.label}
      </div>
      {d.event && (
        <div style={{
          marginTop: 6,
          padding: "4px 8px",
          background: "#fbbf2422",
          border: "1px solid #fbbf2444",
          borderRadius: 4,
          color: "#fbbf24",
          fontSize: 11,
          textAlign: "center",
        }}>
          ⚑ {d.event}
        </div>
      )}
    </div>
  );
};

// ============================================================
// Regime band renderer for the price chart
// ============================================================

const RegimeBands = ({ data, yMin, yMax }) => {
  if (!data.length) return null;

  const bands = [];
  let bandStart = 0;
  let currentRegime = data[0].regime;

  for (let i = 1; i <= data.length; i++) {
    if (i === data.length || data[i].regime !== currentRegime) {
      const meta = REGIME_META[currentRegime];
      if (meta) {
        bands.push({
          x1: bandStart,
          x2: i - 1,
          regime: currentRegime,
          color: meta.bg,
        });
      }
      if (i < data.length) {
        bandStart = i;
        currentRegime = data[i].regime;
      }
    }
  }

  return bands;
};

// ============================================================
// Stats panel
// ============================================================

const StatsPanel = ({ data, selectedRegime }) => {
  const stats = useMemo(() => {
    const regimeCounts = {};
    const regimeDurations = {};
    let currentRegime = null;
    let currentDuration = 0;

    data.forEach((d, i) => {
      regimeCounts[d.regime] = (regimeCounts[d.regime] || 0) + 1;
      if (d.regime === currentRegime) {
        currentDuration++;
      } else {
        if (currentRegime) {
          if (!regimeDurations[currentRegime]) regimeDurations[currentRegime] = [];
          regimeDurations[currentRegime].push(currentDuration);
        }
        currentRegime = d.regime;
        currentDuration = 1;
      }
    });
    if (currentRegime) {
      if (!regimeDurations[currentRegime]) regimeDurations[currentRegime] = [];
      regimeDurations[currentRegime].push(currentDuration);
    }

    return Object.entries(REGIME_META).map(([key, meta]) => {
      const count = regimeCounts[key] || 0;
      const pct = ((count / data.length) * 100).toFixed(1);
      const durations = regimeDurations[key] || [];
      const avgDuration = durations.length
        ? (durations.reduce((a, b) => a + b, 0) / durations.length).toFixed(0)
        : 0;
      const transitions = durations.length;

      return { key, ...meta, count, pct, avgDuration, transitions };
    });
  }, [data]);

  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
      gap: 10,
      marginBottom: 20,
    }}>
      {stats.map((s) => (
        <div
          key={s.key}
          style={{
            background: selectedRegime === s.key ? s.bg : "#111",
            border: `1px solid ${selectedRegime === s.key ? s.color : "#222"}`,
            borderRadius: 8,
            padding: "12px 14px",
            cursor: "pointer",
            transition: "all 0.2s",
            opacity: selectedRegime && selectedRegime !== s.key ? 0.4 : 1,
          }}
        >
          <div style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: 8,
          }}>
            <span style={{
              width: 10,
              height: 10,
              borderRadius: "50%",
              background: s.color,
              display: "inline-block",
              flexShrink: 0,
            }} />
            <span style={{
              color: s.color,
              fontWeight: 700,
              fontSize: 13,
              fontFamily: "'JetBrains Mono', monospace",
            }}>
              {s.icon} {s.label}
            </span>
          </div>
          <div style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: "2px 12px",
            fontSize: 11,
            fontFamily: "'JetBrains Mono', monospace",
          }}>
            <span style={{ color: "#666" }}>Time</span>
            <span style={{ color: "#ccc" }}>{s.pct}%</span>
            <span style={{ color: "#666" }}>Avg duration</span>
            <span style={{ color: "#ccc" }}>{s.avgDuration}d</span>
            <span style={{ color: "#666" }}>Occurrences</span>
            <span style={{ color: "#ccc" }}>{s.transitions}</span>
          </div>
          <div style={{
            marginTop: 8,
            fontSize: 10,
            color: "#888",
            lineHeight: 1.4,
          }}>
            {s.strategies}
          </div>
        </div>
      ))}
    </div>
  );
};

// ============================================================
// Main Dashboard Component
// ============================================================

export default function RegimeDashboard() {
  const allData = useMemo(() => generateMarketData(), []);
  const [selectedRegime, setSelectedRegime] = useState(null);
  const [yearFilter, setYearFilter] = useState("all");
  const [showVix, setShowVix] = useState(true);
  const [showAdx, setShowAdx] = useState(true);
  const [showFii, setShowFii] = useState(false);
  const [showPcr, setShowPcr] = useState(false);
  const [hoveredIndex, setHoveredIndex] = useState(null);

  const filteredData = useMemo(() => {
    let d = allData;
    if (yearFilter !== "all") {
      d = d.filter((x) => x.year === parseInt(yearFilter));
    }
    return d;
  }, [allData, yearFilter]);

  const years = useMemo(() => [...new Set(allData.map((d) => d.year))], [allData]);

  const events = useMemo(
    () => filteredData.filter((d) => d.event).map((d, i) => ({
      ...d,
      index: filteredData.indexOf(d),
    })),
    [filteredData]
  );

  const priceRange = useMemo(() => {
    const prices = filteredData.map((d) => d.price);
    return {
      min: Math.floor(Math.min(...prices) / 500) * 500,
      max: Math.ceil(Math.max(...prices) / 500) * 500,
    };
  }, [filteredData]);

  // Build regime-colored price data
  const chartData = useMemo(() => {
    return filteredData.map((d, i) => ({
      ...d,
      index: i,
      regimeColor: REGIME_META[d.regime]?.color || "#666",
    }));
  }, [filteredData]);

  // Sample every Nth label to avoid crowding
  const tickInterval = Math.max(1, Math.floor(chartData.length / 12));

  return (
    <div style={{
      background: "#0a0a0a",
      color: "#e5e5e5",
      minHeight: "100vh",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: "24px 20px",
    }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{
          fontSize: 20,
          fontWeight: 800,
          letterSpacing: "-0.5px",
          margin: 0,
          color: "#fff",
        }}>
          <span style={{ color: "#22c55e" }}>◆</span> NIFTY 50 — Regime Analysis
        </h1>
        <p style={{
          fontSize: 12,
          color: "#666",
          margin: "6px 0 0",
        }}>
          Visual validation of regime classifier · VIX × ADX matrix · {chartData.length} trading days
        </p>
      </div>

      {/* Controls */}
      <div style={{
        display: "flex",
        flexWrap: "wrap",
        gap: 8,
        marginBottom: 16,
        alignItems: "center",
      }}>
        <div style={{
          display: "flex",
          background: "#151515",
          borderRadius: 6,
          border: "1px solid #222",
          overflow: "hidden",
        }}>
          {["all", ...years].map((y) => (
            <button
              key={y}
              onClick={() => setYearFilter(String(y))}
              style={{
                padding: "6px 14px",
                fontSize: 12,
                fontFamily: "inherit",
                background: yearFilter === String(y) ? "#222" : "transparent",
                color: yearFilter === String(y) ? "#fff" : "#666",
                border: "none",
                cursor: "pointer",
                transition: "all 0.15s",
              }}
            >
              {y === "all" ? "ALL" : y}
            </button>
          ))}
        </div>

        <div style={{ width: 1, height: 24, background: "#222", margin: "0 4px" }} />

        {[
          { key: "showVix", label: "VIX", state: showVix, set: setShowVix, color: "#ef4444" },
          { key: "showAdx", label: "ADX", state: showAdx, set: setShowAdx, color: "#3b82f6" },
          { key: "showFii", label: "FII Flow", state: showFii, set: setShowFii, color: "#f59e0b" },
          { key: "showPcr", label: "PCR", state: showPcr, set: setShowPcr, color: "#8b5cf6" },
        ].map((toggle) => (
          <button
            key={toggle.key}
            onClick={() => toggle.set(!toggle.state)}
            style={{
              padding: "6px 12px",
              fontSize: 11,
              fontFamily: "inherit",
              background: toggle.state ? `${toggle.color}18` : "#111",
              color: toggle.state ? toggle.color : "#555",
              border: `1px solid ${toggle.state ? `${toggle.color}44` : "#222"}`,
              borderRadius: 6,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            {toggle.label}
          </button>
        ))}

        <div style={{ width: 1, height: 24, background: "#222", margin: "0 4px" }} />

        <button
          onClick={() => setSelectedRegime(null)}
          style={{
            padding: "6px 12px",
            fontSize: 11,
            fontFamily: "inherit",
            background: selectedRegime ? "#222" : "transparent",
            color: selectedRegime ? "#fff" : "#444",
            border: `1px solid ${selectedRegime ? "#444" : "#222"}`,
            borderRadius: 6,
            cursor: selectedRegime ? "pointer" : "default",
            transition: "all 0.15s",
          }}
        >
          Clear filter
        </button>
      </div>

      {/* Regime Stats */}
      <StatsPanel data={filteredData} selectedRegime={selectedRegime} />

      {/* Main Price Chart with Regime Background */}
      <div style={{
        background: "#0f0f0f",
        border: "1px solid #1a1a1a",
        borderRadius: 10,
        padding: "16px 8px 8px",
        marginBottom: 12,
      }}>
        <div style={{
          fontSize: 11,
          color: "#555",
          padding: "0 12px 8px",
          display: "flex",
          justifyContent: "space-between",
        }}>
          <span>NIFTY 50 — Price with Regime Overlay</span>
          <span>Click regime cards above to highlight</span>
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart
            data={chartData}
            margin={{ top: 5, right: 30, bottom: 5, left: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
            <XAxis
              dataKey="dateShort"
              tick={{ fontSize: 9, fill: "#444" }}
              tickLine={false}
              axisLine={{ stroke: "#222" }}
              interval={tickInterval}
            />
            <YAxis
              domain={[priceRange.min, priceRange.max]}
              tick={{ fontSize: 10, fill: "#555" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `${(v / 1000).toFixed(1)}K`}
            />
            <Tooltip content={<ChartTooltip />} />

            {/* Regime-colored price line segments */}
            {Object.keys(REGIME_META).map((regime) => (
              <Line
                key={regime}
                type="monotone"
                dataKey={(d) =>
                  (!selectedRegime || d.regime === selectedRegime) && d.regime === regime
                    ? d.price
                    : undefined
                }
                stroke={REGIME_META[regime].color}
                strokeWidth={selectedRegime === regime ? 2.5 : 1.5}
                dot={false}
                connectNulls={false}
                isAnimationActive={false}
              />
            ))}

            {/* Background price line (dim) when filtering */}
            {selectedRegime && (
              <Line
                type="monotone"
                dataKey="price"
                stroke="#222"
                strokeWidth={1}
                dot={false}
                isAnimationActive={false}
              />
            )}

            {/* Event markers */}
            {events.map((e) => (
              <ReferenceLine
                key={e.date}
                x={e.dateShort}
                stroke="#fbbf24"
                strokeDasharray="4 4"
                strokeWidth={1}
                label={{
                  value: `⚑ ${e.event}`,
                  position: "top",
                  fill: "#fbbf24",
                  fontSize: 9,
                }}
              />
            ))}

            {/* VIX threshold lines */}
            {showVix && (
              <>
                <ReferenceLine y={14} stroke="#22c55e" strokeDasharray="2 4" strokeWidth={0.5} yAxisId={0} />
                <ReferenceLine y={18} stroke="#ef4444" strokeDasharray="2 4" strokeWidth={0.5} yAxisId={0} />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* VIX + ADX Panel */}
      {(showVix || showAdx) && (
        <div style={{
          background: "#0f0f0f",
          border: "1px solid #1a1a1a",
          borderRadius: 10,
          padding: "16px 8px 8px",
          marginBottom: 12,
        }}>
          <div style={{
            fontSize: 11,
            color: "#555",
            padding: "0 12px 8px",
            display: "flex",
            gap: 16,
          }}>
            {showVix && <span style={{ color: "#ef4444" }}>━ India VIX</span>}
            {showAdx && <span style={{ color: "#3b82f6" }}>━ ADX(14)</span>}
            <span style={{ color: "#22c55e", opacity: 0.5 }}>┈ VIX 14 (low vol)</span>
            <span style={{ color: "#ef4444", opacity: 0.5 }}>┈ VIX 18 (high vol)</span>
            <span style={{ color: "#3b82f6", opacity: 0.5 }}>┈ ADX 25 (trending)</span>
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 30, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis
                dataKey="dateShort"
                tick={{ fontSize: 9, fill: "#444" }}
                tickLine={false}
                axisLine={{ stroke: "#222" }}
                interval={tickInterval}
              />
              <YAxis
                domain={[5, 40]}
                tick={{ fontSize: 10, fill: "#555" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<ChartTooltip />} />

              {showVix && (
                <Line
                  type="monotone"
                  dataKey="vix"
                  stroke="#ef4444"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              )}
              {showAdx && (
                <Line
                  type="monotone"
                  dataKey="adx"
                  stroke="#3b82f6"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              )}

              {/* Threshold lines */}
              <ReferenceLine y={14} stroke="#22c55e" strokeDasharray="4 4" strokeWidth={0.8} />
              <ReferenceLine y={18} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={0.8} />
              <ReferenceLine y={25} stroke="#3b82f6" strokeDasharray="4 4" strokeWidth={0.8} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* FII Flow Panel */}
      {showFii && (
        <div style={{
          background: "#0f0f0f",
          border: "1px solid #1a1a1a",
          borderRadius: 10,
          padding: "16px 8px 8px",
          marginBottom: 12,
        }}>
          <div style={{
            fontSize: 11,
            color: "#555",
            padding: "0 12px 8px",
            display: "flex",
            gap: 16,
          }}>
            <span style={{ color: "#f59e0b" }}>FII Net Flow (₹ Cr)</span>
            <span style={{ color: "#ef4444", opacity: 0.5 }}>┈ −2,000 Cr alert</span>
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 30, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis
                dataKey="dateShort"
                tick={{ fontSize: 9, fill: "#444" }}
                tickLine={false}
                axisLine={{ stroke: "#222" }}
                interval={tickInterval}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "#555" }}
                tickLine={false}
                axisLine={false}
                tickFormatter={(v) => `${(v / 1000).toFixed(0)}K`}
              />
              <Tooltip content={<ChartTooltip />} />
              <Bar
                dataKey="fiiFlow"
                isAnimationActive={false}
                shape={(props) => {
                  const { x, y, width, height, payload } = props;
                  const fill = payload.fiiFlow >= 0 ? "#22c55e" : "#ef4444";
                  return <rect x={x} y={y} width={Math.max(width, 1)} height={Math.abs(height)} fill={fill} opacity={0.6} />;
                }}
              />
              <ReferenceLine y={-2000} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={0.8} />
              <ReferenceLine y={0} stroke="#333" strokeWidth={0.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* PCR Panel */}
      {showPcr && (
        <div style={{
          background: "#0f0f0f",
          border: "1px solid #1a1a1a",
          borderRadius: 10,
          padding: "16px 8px 8px",
          marginBottom: 12,
        }}>
          <div style={{
            fontSize: 11,
            color: "#555",
            padding: "0 12px 8px",
            display: "flex",
            gap: 16,
          }}>
            <span style={{ color: "#8b5cf6" }}>━ Put-Call Ratio (OI)</span>
            <span style={{ color: "#ef4444", opacity: 0.5 }}>┈ 0.7 oversold</span>
            <span style={{ color: "#22c55e", opacity: 0.5 }}>┈ 1.3 overbought</span>
          </div>
          <ResponsiveContainer width="100%" height={140}>
            <ComposedChart data={chartData} margin={{ top: 5, right: 30, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis
                dataKey="dateShort"
                tick={{ fontSize: 9, fill: "#444" }}
                tickLine={false}
                axisLine={{ stroke: "#222" }}
                interval={tickInterval}
              />
              <YAxis
                domain={[0.4, 1.6]}
                tick={{ fontSize: 10, fill: "#555" }}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<ChartTooltip />} />
              <Line
                type="monotone"
                dataKey="pcr"
                stroke="#8b5cf6"
                strokeWidth={1.5}
                dot={false}
                isAnimationActive={false}
              />
              <ReferenceLine y={0.7} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={0.8} />
              <ReferenceLine y={1.3} stroke="#22c55e" strokeDasharray="4 4" strokeWidth={0.8} />
              <ReferenceLine y={1.0} stroke="#333" strokeWidth={0.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Regime Timeline */}
      <div style={{
        background: "#0f0f0f",
        border: "1px solid #1a1a1a",
        borderRadius: 10,
        padding: "16px 14px",
        marginBottom: 12,
      }}>
        <div style={{ fontSize: 11, color: "#555", marginBottom: 10 }}>
          Regime Timeline — each segment represents a continuous regime period
        </div>
        <div style={{
          display: "flex",
          height: 32,
          borderRadius: 6,
          overflow: "hidden",
          border: "1px solid #222",
        }}>
          {(() => {
            const bands = [];
            let start = 0;
            let current = chartData[0]?.regime;
            for (let i = 1; i <= chartData.length; i++) {
              if (i === chartData.length || chartData[i].regime !== current) {
                const width = ((i - start) / chartData.length) * 100;
                const meta = REGIME_META[current];
                bands.push(
                  <div
                    key={`${start}-${current}`}
                    onClick={() => setSelectedRegime(selectedRegime === current ? null : current)}
                    style={{
                      width: `${width}%`,
                      background: meta?.bg?.replace("0.12", selectedRegime && selectedRegime !== current ? "0.04" : "0.25") || "#111",
                      borderRight: "1px solid #0a0a0a",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      cursor: "pointer",
                      transition: "all 0.2s",
                      opacity: selectedRegime && selectedRegime !== current ? 0.3 : 1,
                    }}
                    title={`${meta?.label}: ${chartData[start]?.date} → ${chartData[Math.min(i - 1, chartData.length - 1)]?.date} (${i - start}d)`}
                  >
                    {width > 3 && (
                      <span style={{
                        fontSize: width > 6 ? 9 : 7,
                        color: meta?.color,
                        fontWeight: 600,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                      }}>
                        {width > 6 ? meta?.short : meta?.icon}
                      </span>
                    )}
                  </div>
                );
                if (i < chartData.length) {
                  start = i;
                  current = chartData[i].regime;
                }
              }
            }
            return bands;
          })()}
        </div>
        <div style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          marginTop: 10,
          fontSize: 10,
        }}>
          {Object.entries(REGIME_META).map(([key, meta]) => (
            <span
              key={key}
              onClick={() => setSelectedRegime(selectedRegime === key ? null : key)}
              style={{
                color: meta.color,
                cursor: "pointer",
                opacity: selectedRegime && selectedRegime !== key ? 0.3 : 1,
                transition: "opacity 0.2s",
              }}
            >
              {meta.icon} {meta.short}
            </span>
          ))}
        </div>
      </div>

      {/* Decision Matrix */}
      <div style={{
        background: "#0f0f0f",
        border: "1px solid #1a1a1a",
        borderRadius: 10,
        padding: "16px 14px",
      }}>
        <div style={{ fontSize: 11, color: "#555", marginBottom: 12 }}>
          Regime Classification Matrix — VIX × ADX
        </div>
        <div style={{
          display: "grid",
          gridTemplateColumns: "80px 1fr 1fr 1fr",
          gridTemplateRows: "32px 1fr 1fr 1fr",
          gap: 4,
          fontSize: 11,
          maxWidth: 500,
        }}>
          {/* Header row */}
          <div />
          <div style={{ color: "#555", textAlign: "center", alignSelf: "end", paddingBottom: 4 }}>ADX {">"} 25</div>
          <div style={{ color: "#555", textAlign: "center", alignSelf: "end", paddingBottom: 4 }}>ADX 20–25</div>
          <div style={{ color: "#555", textAlign: "center", alignSelf: "end", paddingBottom: 4 }}>ADX {"<"} 20</div>

          {/* VIX < 14 */}
          <div style={{ color: "#555", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: 8 }}>
            VIX {"<"} 14
          </div>
          {[
            { r: "LOW_VOL_TRENDING" },
            { r: "TRANSITIONAL_TRENDING", note: "Lean prev" },
            { r: "LOW_VOL_RANGING" },
          ].map((cell, i) => {
            const meta = REGIME_META[cell.r];
            return (
              <div key={i} style={{
                background: meta.bg,
                border: `1px solid ${meta.color}33`,
                borderRadius: 6,
                padding: "8px 6px",
                textAlign: "center",
                color: meta.color,
                fontWeight: 600,
                cursor: "pointer",
              }}
              onClick={() => setSelectedRegime(selectedRegime === cell.r ? null : cell.r)}>
                {meta.icon} {meta.short}
              </div>
            );
          })}

          {/* VIX 14–18 */}
          <div style={{ color: "#555", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: 8 }}>
            VIX 14–18
          </div>
          {[
            { r: "TRANSITIONAL_TRENDING" },
            { r: "TRANSITIONAL_RANGING", note: "Hysteresis" },
            { r: "TRANSITIONAL_RANGING" },
          ].map((cell, i) => {
            const meta = REGIME_META[cell.r];
            return (
              <div key={i} style={{
                background: meta.bg,
                border: `1px solid ${meta.color}22`,
                borderRadius: 6,
                padding: "8px 6px",
                textAlign: "center",
                color: meta.color,
                fontWeight: 500,
                fontSize: 10,
              }}>
                {meta.icon} {meta.short}
                {cell.note && <div style={{ fontSize: 8, color: "#666", marginTop: 2 }}>{cell.note}</div>}
              </div>
            );
          })}

          {/* VIX > 18 */}
          <div style={{ color: "#555", display: "flex", alignItems: "center", justifyContent: "flex-end", paddingRight: 8 }}>
            VIX {">"} 18
          </div>
          {[
            { r: "HIGH_VOL_TRENDING" },
            { r: "HIGH_VOL_TRENDING", note: "Lean high-vol" },
            { r: "HIGH_VOL_CHOPPY" },
          ].map((cell, i) => {
            const meta = REGIME_META[cell.r];
            return (
              <div key={i} style={{
                background: meta.bg,
                border: `1px solid ${meta.color}33`,
                borderRadius: 6,
                padding: "8px 6px",
                textAlign: "center",
                color: meta.color,
                fontWeight: 600,
                cursor: "pointer",
              }}
              onClick={() => setSelectedRegime(selectedRegime === cell.r ? null : cell.r)}>
                {meta.icon} {meta.short}
              </div>
            );
          })}
        </div>
      </div>

      {/* Footer */}
      <div style={{
        marginTop: 16,
        fontSize: 10,
        color: "#333",
        textAlign: "center",
        lineHeight: 1.6,
      }}>
        Simulated data based on realistic NIFTY 50 patterns (2022–2025) · Not actual market data
        <br />
        Replace with real data from TrueData + jugaad-data for production validation
      </div>
    </div>
  );
}
