import React from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, LabelList, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'

export const COLORS = {
  actual: 'var(--series-actual)',
  predicted: 'var(--series-predicted)',
  neutral: 'var(--series-neutral)',
  warn: 'var(--series-warn)',
}

const axis = { stroke: 'var(--muted)', fontSize: 12, tickLine: false }
const tip = { contentStyle: { background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text)' } }

// Grouped bars: actual vs predicted for the same categories.
export function ActualVsPredictedBars({ data, xKey, actualKey, predictedKey, yFormat, height = 280, actualName = 'Actual', predictedName = 'Predicted' }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} tickFormatter={yFormat} width={56} />
        <Tooltip {...tip} formatter={(v) => (yFormat ? yFormat(v) : v)} />
        <Legend />
        <Bar dataKey={actualKey} name={actualName} fill={COLORS.actual} radius={[3, 3, 0, 0]} isAnimationActive={false} />
        <Bar dataKey={predictedKey} name={predictedName} fill={COLORS.predicted} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function SingleBars({ data, xKey, yKey, yFormat, height = 280, name, layout = 'horizontal', colorFor, baseline }) {
  const vertical = layout === 'vertical'
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout={layout} margin={{ top: 8, right: 16, left: vertical ? 40 : 0, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" horizontal={vertical ? false : true} vertical={vertical} />
        {vertical ? (
          <>
            <XAxis type="number" {...axis} tickFormatter={yFormat} />
            <YAxis type="category" dataKey={xKey} {...axis} width={110} />
          </>
        ) : (
          <>
            <XAxis dataKey={xKey} {...axis} />
            <YAxis {...axis} tickFormatter={yFormat} width={56} />
          </>
        )}
        <Tooltip {...tip} formatter={(v) => (yFormat ? yFormat(v) : v)} />
        {baseline != null && (vertical ? <ReferenceLine x={baseline} stroke={COLORS.warn} strokeDasharray="4 3" /> : <ReferenceLine y={baseline} stroke={COLORS.warn} strokeDasharray="4 3" />)}
        <Bar dataKey={yKey} name={name} fill={COLORS.predicted} radius={vertical ? [0, 3, 3, 0] : [3, 3, 0, 0]} isAnimationActive={false}>
          {colorFor && data.map((d, i) => <Cell key={i} fill={colorFor(d)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// Calibration: actual vs predicted lines across ordered bins.
export function CalibrationChart({ data, xKey, actualKey, predictedKey, yFormat, height = 280 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" vertical={false} />
        <XAxis dataKey={xKey} {...axis} />
        <YAxis {...axis} tickFormatter={yFormat} width={56} />
        <Tooltip {...tip} formatter={(v) => (yFormat ? yFormat(v) : v)} />
        <Legend />
        <Line type="linear" dataKey={actualKey} name="Actual" stroke={COLORS.actual} strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
        <Line type="linear" dataKey={predictedKey} name="Predicted" stroke={COLORS.predicted} strokeWidth={2} strokeDasharray="5 3" dot={{ r: 3 }} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  )
}

// Fixed axis (0 to `max`) so bars are read against the thresholds, not stretched to fill the width.
export function ThresholdBars({ data, xKey, yKey, thresholds, height = 280, yFormat, max = 0.3 }) {
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} layout="vertical" margin={{ top: 26, right: 24, left: 40, bottom: 4 }}>
        <CartesianGrid stroke="var(--border)" horizontal={false} />
        <XAxis type="number" {...axis} domain={[0, max]} ticks={[0, 0.1, 0.25, max].filter((v, i, a) => a.indexOf(v) === i)} tickFormatter={yFormat} allowDataOverflow />
        <YAxis type="category" dataKey={xKey} {...axis} width={130} />
        <Tooltip {...tip} formatter={(v) => (yFormat ? yFormat(v) : v)} />
        {thresholds.map((t) => (
          <ReferenceLine key={t.value} x={t.value} stroke={COLORS.warn} strokeDasharray="4 3" label={{ value: t.label, fill: 'var(--muted)', fontSize: 11, position: 'top' }} />
        ))}
        <Bar dataKey={yKey} fill={COLORS.predicted} radius={[0, 3, 3, 0]} isAnimationActive={false} minPointSize={2}>
          <LabelList dataKey={yKey} position="right" formatter={(v) => Number(v).toFixed(5)} fill="var(--muted)" fontSize={11} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
