import React from 'react'
import { ActualVsPredictedBars } from '../components/charts'
import { Callout, Card, Kpi, Table } from '../components/ui'
import { describeRatio, fmtInt, fmtMoney, fmtMoneyCompact, fmtNum, fmtPct, prettyModel } from '../format'

export default function Overview({ data }) {
  const { overview: o, segments } = data
  const freqPred = o.expected_claims / o.exposure
  const freqAct = o.actual_claims / o.exposure
  const gapPct = Math.abs(1 - o.actual_to_expected) * 100

  return (
    <>
      <Callout kind="info">
        Figures are from the <strong>validation split</strong> ({fmtInt(o.policies)} policies) used for model selection. The held-out test
        split is not shown. Expected cost is a pure-premium estimate of claims losses, not a customer price.
      </Callout>

      <div className="grid kpis">
        <Kpi label="Policies evaluated" value={fmtInt(o.policies)} sub={`${fmtInt(o.exposure)} exposure-years`} />
        <Kpi label="Actual claims cost" value={fmtMoneyCompact(o.actual_cost)} sub="historical outcome" />
        <Kpi label="Expected claims cost" value={fmtMoneyCompact(o.expected_cost)} sub="model prediction" />
        <Kpi
          tone={gapPct > 5 ? 'warn' : undefined}
          label="Actual ÷ expected cost"
          value={fmtNum(o.actual_to_expected, 3)}
          sub={`actual is ${describeRatio(o.actual_to_expected)} expected`}
        />
        <Kpi label="Claim frequency" value={fmtPct(freqAct, 2)} sub={`predicted ${fmtPct(freqPred, 2)} per exposure-year`} />
        <Kpi label="Frequency ranking (Gini)" value={fmtNum(o.frequency_metrics.gini, 3)} sub={prettyModel(o.selected_frequency_model)} />
      </div>

      {gapPct > 5 && (
        <Callout kind="warn">
          Total expected cost is {gapPct.toFixed(0)}% {o.actual_to_expected < 1 ? 'above' : 'below'} actual cost. Frequency is well calibrated, so
          the gap comes from severity: the average claim in the training data is inflated by a few very large claims, and severity could not be
          predicted reliably from the available features.
        </Callout>
      )}

      <div className="grid two">
        <Card title="Pure premium by predicted risk segment" subtitle="Actual vs expected claims cost per exposure-year (€)">
          <ActualVsPredictedBars data={segments} xKey="risk_segment" actualKey="actual_pure_premium" predictedKey="expected_pure_premium" yFormat={fmtMoney} actualName="Actual" predictedName="Expected" />
        </Card>
        <Card title="Model performance" subtitle="Selected models, validation split">
          <Table
            columns={[
              { key: 'k', label: 'Metric', align: 'left' },
              { key: 'v', label: 'Value' },
            ]}
            rows={[
              { k: `Frequency: ${prettyModel(o.selected_frequency_model)}`, v: '' },
              { k: 'Poisson deviance (lower is better)', v: fmtNum(o.frequency_metrics.poisson_deviance, 4) },
              { k: 'Actual ÷ predicted claims', v: fmtNum(o.frequency_metrics.actual_to_predicted, 3) },
              { k: `Severity: ${prettyModel(o.selected_severity_model)}`, v: '' },
              { k: 'Gamma deviance (lower is better)', v: fmtNum(o.severity_metrics.gamma_deviance, 3) },
              { k: 'MAE (€ per claim)', v: fmtMoney(o.severity_metrics.mae) },
            ]}
            rowClass={(r) => (r.v === '' ? 'selected' : '')}
          />
          <p className="muted" style={{ marginTop: 10 }}>
            Severity is only weakly predictable here: no severity model was reliably better than a constant average (see Model performance).
          </p>
        </Card>
      </div>

      <Card title="Segments" subtitle="Segments are quintiles of predicted annual cost, with cut points set on training data only.">
        <Table
          columns={[
            { key: 'risk_segment', label: 'Segment', align: 'left' },
            { key: 'policies', label: 'Policies', render: fmtInt },
            { key: 'expected_pure_premium', label: 'Expected €/yr', render: fmtMoney },
            { key: 'actual_pure_premium', label: 'Actual €/yr', render: fmtMoney },
            { key: 'frequency_actual_to_expected', label: 'Freq. actual ÷ expected', render: (v) => fmtNum(v, 3) },
            { key: 'cost_actual_to_expected', label: 'Cost actual ÷ expected', render: (v) => fmtNum(v, 3) },
          ]}
          rows={segments}
        />
      </Card>
    </>
  )
}
