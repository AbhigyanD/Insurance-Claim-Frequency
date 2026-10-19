import React from 'react'
import { CalibrationChart } from '../components/charts'
import { Badge, Callout, Card, Table } from '../components/ui'
import { fmtMoney, fmtNum, fmtPct, prettyModel } from '../format'

const sigBadge = (sig, better) => (sig ? <Badge kind={better ? 'good' : 'bad'}>{better ? 'reliably better' : 'reliably worse'}</Badge> : <Badge kind="neutral">no reliable difference</Badge>)

export default function Models({ data }) {
  const { models: m, calibration: c, overview: o } = data
  const freqCal = c.frequency_lightgbm.map((r) => ({ ...r, bin: `D${r.bin}` }))
  const sevCal = c.severity_lightgbm.map((r) => ({ ...r, bin: `Q${r.bin}` }))

  return (
    <>
      <Callout kind="info">
        All results are on the validation split, which was used to choose models. They are not final test results. Highlighted rows are the
        models selected for the expected-cost predictions.
      </Callout>

      <Card title="Frequency models" subtitle="Claim counts with exposure handled as a log offset. Lower deviance is better; higher Gini means better risk ranking.">
        <Table
          columns={[
            { key: 'model', label: 'Model', align: 'left', render: prettyModel },
            { key: 'poisson_deviance', label: 'Poisson deviance', render: (v) => fmtNum(v, 4) },
            { key: 'mae_counts', label: 'MAE (counts)', render: (v) => fmtNum(v, 4) },
            { key: 'actual_to_predicted', label: 'Actual ÷ predicted', render: (v) => fmtNum(v, 3) },
            { key: 'gini', label: 'Gini', render: (v) => fmtNum(v, 3) },
          ]}
          rows={m.frequency}
          rowClass={(r) => (r.model === o.selected_frequency_model ? 'selected' : '')}
        />
        <p className="muted" style={{ marginTop: 10 }}>
          Pearson dispersion of the Poisson GLM is {fmtNum(m.dispersion.poisson_glm_pearson_dispersion_train, 2)} (above 1 suggests overdispersion). The
          Negative Binomial model (alpha ≈ {fmtNum(m.dispersion.nb_alpha, 2)}) gave almost the same predictions, so it adds little for point
          prediction.
        </p>
      </Card>

      <div className="grid two">
        <Card title="Frequency calibration" subtitle="Policies grouped into ten exposure-equal deciles by predicted risk (LightGBM)">
          <CalibrationChart data={freqCal} xKey="bin" actualKey="actual_freq" predictedKey="predicted_freq" yFormat={(v) => fmtPct(v, 1)} />
        </Card>
        <Card title="Severity calibration" subtitle="Claims grouped into quintiles by predicted severity (LightGBM)">
          <CalibrationChart data={sevCal} xKey="bin" actualKey="actual_mean" predictedKey="predicted_mean" yFormat={fmtMoney} />
        </Card>
      </div>

      <Card title="Severity models" subtitle="Claims with a positive amount only (15,872 train / 5,266 validation).">
        <Table
          columns={[
            { key: 'model', label: 'Model', align: 'left', render: prettyModel },
            { key: 'gamma_deviance', label: 'Gamma deviance', render: (v) => fmtNum(v, 3) },
            { key: 'mae', label: 'MAE', render: fmtMoney },
            { key: 'rmse', label: 'RMSE', render: fmtMoney },
            { key: 'actual_mean', label: 'Actual mean', render: fmtMoney },
            { key: 'predicted_mean', label: 'Predicted mean', render: fmtMoney },
          ]}
          rows={m.severity}
          rowClass={(r) => (r.model === o.selected_severity_model ? 'selected' : '')}
        />
        <h3 style={{ marginTop: 18 }}>Is any severity model better than a constant average?</h3>
        <p className="muted">Bootstrap of validation claims: difference in Gamma deviance versus the constant mean (negative is better).</p>
        <Table
          columns={[
            { key: 'model', label: 'Model', align: 'left', render: prettyModel },
            { key: 'diff', label: 'Difference', render: (v) => fmtNum(v, 3) },
            { key: 'ci', label: '95% interval', render: (_, r) => `${fmtNum(r.ci_low, 3)} to ${fmtNum(r.ci_high, 3)}` },
            { key: 'significant', label: 'Verdict', render: (v, r) => sigBadge(v, r.diff < 0) },
          ]}
          rows={m.severity_bootstrap}
        />
        <p className="muted" style={{ marginTop: 10 }}>
          No severity model is reliably different from a constant. Differences in expected cost across policies are therefore driven almost entirely by
          frequency, and severity relativities should be treated as unreliable.
        </p>
      </Card>

      <Card title="Frequency variants" subtitle="Difference in Poisson deviance versus the selected LightGBM (positive is worse)">
        <Table
          columns={[
            { key: 'model', label: 'Model', align: 'left', render: prettyModel },
            { key: 'gini', label: 'Gini', render: (v) => fmtNum(v, 3) },
            { key: 'dev_diff_vs_lightgbm', label: 'Δ deviance', render: (v) => (v == null ? '–' : fmtNum(v, 4)) },
            { key: 'significant', label: 'Verdict', render: (v, r) => (r.dev_diff_vs_lightgbm == null ? '' : sigBadge(v, r.dev_diff_vs_lightgbm < 0)) },
          ]}
          rows={m.frequency_variants}
        />
      </Card>

      <Card title="Pure premium: two-part vs direct Tweedie" subtitle="Exposure-weighted Tweedie deviance (p=1.5), lower is better">
          <Table
            columns={[
              { key: 'model', label: 'Model', align: 'left', render: prettyModel },
              { key: 'tweedie_dev_p1.5', label: 'Deviance', render: (v) => fmtNum(v, 2) },
              { key: 'cost_gini', label: 'Cost Gini', render: (v) => fmtNum(v, 3) },
              { key: 'actual_to_expected', label: 'Actual ÷ expected', render: (v) => fmtNum(v, 2) },
            ]}
            rows={m.pure_premium_variants}
            rowClass={(r) => (r.model.includes('selected') ? 'selected' : '')}
          />
      </Card>

      <Card title="Performance by feature group" subtitle="Actual ÷ expected. Frequency is stable; cost swings because it rests on few, heavy-tailed claims.">
        <Table
          columns={[
            { key: 'feature', label: 'Feature', align: 'left' },
            { key: 'level', label: 'Level', align: 'left' },
            { key: 'policies', label: 'Policies', render: (v) => v.toLocaleString('en-GB') },
            { key: 'frequency_actual_to_expected', label: 'Frequency A÷E', render: (v) => fmtNum(v, 3) },
            { key: 'cost_actual_to_expected', label: 'Cost A÷E', render: (v) => fmtNum(v, 3) },
          ]}
          rows={c.stability}
        />
      </Card>
    </>
  )
}
