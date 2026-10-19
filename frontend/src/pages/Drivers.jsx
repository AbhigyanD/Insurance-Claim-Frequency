import React from 'react'
import { SingleBars } from '../components/charts'
import { Callout, Card, Table } from '../components/ui'
import { fmtNum, fmtPct } from '../format'

const TERM_LABEL = (t) =>
  t.replace('DrivAge_band_', 'Driver age ').replace('BonusMalus_band_', 'Bonus-malus ').replace('VehGas_Regular', 'Fuel: regular (vs diesel)').replace('LogDensity', 'Log population density')

export default function Drivers({ data }) {
  const { shap, relativities } = data.explain
  const age = relativities.filter((r) => r.term.startsWith('DrivAge')).map((r) => ({ ...r, label: TERM_LABEL(r.term) }))
  const bm = relativities.filter((r) => r.term.startsWith('BonusMalus')).map((r) => ({ ...r, label: TERM_LABEL(r.term) }))

  return (
    <>
      <Callout kind="warn">
        These are statistical <strong>associations</strong>, not causal effects. Bonus-malus and driver age partly stand in for each other and for claims
        history, so a large importance does not mean that changing the factor would change a policyholder's risk.
      </Callout>

      <Card title="What drives predicted claim frequency?" subtitle="Mean absolute SHAP value (log claim-rate scale) for the LightGBM frequency model, 5,000 validation policies">
        <SingleBars data={[...shap].reverse()} xKey="feature" yKey="mean_abs_shap" name="Mean |SHAP|" layout="vertical" height={320} yFormat={(v) => fmtNum(v, 2)} />
        <p className="muted">Bonus-malus accounts for {fmtPct(shap[0].share_of_total, 0)} of total importance, followed by driver age. Area adds little beyond density and region.</p>
      </Card>

      <div className="grid two">
        <Card title="Driver age: claim-rate relativity" subtitle="Poisson GLM, relative to the youngest band (18–21). 1.0 = same rate.">
          <SingleBars data={age} xKey="label" yKey="relativity" name="Relativity" layout="vertical" height={300} yFormat={(v) => fmtNum(v, 2)} />
        </Card>
        <Card title="Bonus-malus: claim-rate relativity" subtitle="Poisson GLM, relative to the 100–125 band.">
          <SingleBars data={bm} xKey="label" yKey="relativity" name="Relativity" layout="vertical" height={300} yFormat={(v) => fmtNum(v, 2)} />
        </Card>
      </div>

      <Card title="Relativities with 95% intervals">
        <Table
          columns={[
            { key: 'label', label: 'Term', align: 'left' },
            { key: 'relativity', label: 'Relativity', render: (v) => fmtNum(v, 3) },
            { key: 'ci', label: '95% interval', render: (_, r) => `${fmtNum(r.ci_low, 3)} to ${fmtNum(r.ci_high, 3)}` },
          ]}
          rows={relativities.map((r) => ({ ...r, label: TERM_LABEL(r.term) }))}
        />
      </Card>
    </>
  )
}
