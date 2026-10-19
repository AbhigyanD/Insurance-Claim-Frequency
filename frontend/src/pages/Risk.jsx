import React, { useState } from 'react'
import { SingleBars } from '../components/charts'
import { Badge, Callout, Card, Segmented, Table } from '../components/ui'
import { fmtInt, fmtMoney, fmtNum, fmtPct, isSmallSample } from '../format'

const DIMENSIONS = [
  { value: 'driver_age', label: 'Driver age' },
  { value: 'vehicle_age', label: 'Vehicle age' },
  { value: 'area', label: 'Area' },
  { value: 'region', label: 'Region' },
  { value: 'fuel', label: 'Fuel' },
  { value: 'brand', label: 'Brand' },
]

export default function Risk({ data }) {
  const [dim, setDim] = useState('driver_age')
  const rows = data.risk[dim]
  const dist = data.distribution

  return (
    <>
      <Callout kind="info">
        Expected cost differences below are <strong>associations</strong> in observational data. They show where the model predicts higher or
        lower losses, not what would happen if a policyholder changed. Groups with under 1,000 exposure-years are marked and should be read with care.
      </Callout>

      <Card title="Expected annual cost by category" subtitle="Average predicted annual claims cost per policy (€), validation split">
        <Segmented label="Risk dimension" options={DIMENSIONS} value={dim} onChange={setDim} />
        <SingleBars
          data={rows}
          xKey="level"
          yKey="avg_expected_annual_cost"
          name="Avg expected annual cost"
          yFormat={fmtMoney}
          layout={rows.length > 8 ? 'vertical' : 'horizontal'}
          height={rows.length > 8 ? 460 : 280}
          colorFor={(d) => (isSmallSample(d.exposure) ? 'var(--series-neutral)' : 'var(--series-predicted)')}
        />
        <Table
          columns={[
            { key: 'level', label: 'Level', align: 'left', render: (v, r) => (<>{v} {isSmallSample(r.exposure) && <Badge kind="warn">small sample</Badge>}</>) },
            { key: 'policies', label: 'Policies', render: fmtInt },
            { key: 'exposure', label: 'Exposure-yrs', render: fmtInt },
            { key: 'avg_expected_annual_cost', label: 'Expected €/yr', render: fmtMoney },
            { key: 'actual_pure_premium', label: 'Actual €/yr', render: fmtMoney },
            { key: 'predicted_frequency', label: 'Pred. freq.', render: (v) => fmtPct(v, 2) },
            { key: 'actual_frequency', label: 'Actual freq.', render: (v) => fmtPct(v, 2) },
            { key: 'cost_actual_to_expected', label: 'Cost A÷E', render: (v) => fmtNum(v, 2) },
          ]}
          rows={rows}
          rowClass={(r) => (isSmallSample(r.exposure) ? 'small' : '')}
        />
      </Card>

      <Card title="Distribution of predicted annual cost" subtitle={`Median ${fmtMoney(dist.p50)}, 90th percentile ${fmtMoney(dist.p90)}, 99th percentile ${fmtMoney(dist.p99)} per policy-year`}>
        <SingleBars data={dist.bins.map((b) => ({ ...b, label: Math.round(b.from) }))} xKey="label" yKey="policies" name="Policies" yFormat={fmtInt} height={260} />
        <p className="muted">{dist.note}</p>
      </Card>
    </>
  )
}
