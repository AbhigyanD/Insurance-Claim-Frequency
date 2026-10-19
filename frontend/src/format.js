// Number formatting helpers. Currency is euros: the source data is French motor insurance.
const nf = (opts) => new Intl.NumberFormat('en-GB', opts)

export const fmtInt = (v) => (v == null ? '–' : nf({ maximumFractionDigits: 0 }).format(v))
export const fmtMoney = (v) => (v == null ? '–' : '€' + nf({ maximumFractionDigits: 0 }).format(v))
// Compact euros (K / M / B). Done by hand rather than with Intl's compact notation, whose symbols
// ("M" vs "m") depend on the ICU data of the runtime and so differ between Node versions and browsers.
const COMPACT_UNITS = [[1e9, 'B'], [1e6, 'M'], [1e3, 'K']]
export function fmtMoneyCompact(v) {
  if (v == null || !Number.isFinite(v)) return '–'
  const abs = Math.abs(v)
  const sign = v < 0 ? '-' : ''
  let i = COMPACT_UNITS.findIndex(([unit]) => abs >= unit)
  if (i === -1) return `${sign}€${Math.round(abs)}`
  // if rounding would print 1000.0 of this unit (e.g. 999,999 -> 1000.0K), step up one unit instead
  if (i > 0 && Math.round((abs / COMPACT_UNITS[i][0]) * 10) / 10 >= 1000) i -= 1
  const [unit, suffix] = COMPACT_UNITS[i]
  return `${sign}€${(abs / unit).toFixed(1)}${suffix}`
}

export const fmtNum = (v, d = 3) => (v == null ? '–' : Number(v).toFixed(d))
export const fmtPct = (v, d = 1) => (v == null ? '–' : (v * 100).toFixed(d) + '%')

// Actual / expected ratio -> plain-language description of the gap.
export function describeRatio(ratio) {
  if (ratio == null) return ''
  const gap = Math.abs(ratio - 1)
  if (gap < 0.02) return 'in line with'
  const pct = (gap * 100).toFixed(0)
  return ratio < 1 ? `${pct}% below` : `${pct}% above`
}

// Small samples make actual/expected ratios unreliable; flag them rather than hide them.
export const isSmallSample = (exposure, min = 1000) => exposure != null && exposure < min

export const prettyModel = (name) =>
  ({
    constant_rate: 'Constant rate',
    poisson_glm: 'Poisson GLM',
    negbin_glm: 'Negative Binomial GLM',
    lightgbm_poisson: 'LightGBM (Poisson)',
    lightgbm_monotone_bonusmalus: 'LightGBM, monotone bonus-malus',
    glm_boosted_lightgbm: 'GLM-boosted LightGBM',
    constant_mean: 'Constant mean',
    gamma_glm: 'Gamma GLM',
    'gamma_glm_capped_p99.5': 'Gamma GLM, claims capped at p99.5',
    ridge_gamma_glm: 'Ridge Gamma GLM',
    lightgbm_gamma: 'LightGBM (Gamma)',
    'frequency_x_severity (selected)': 'Frequency × severity (selected)',
    'tweedie_lightgbm_p1.3': 'Tweedie LightGBM, p=1.3',
    'tweedie_lightgbm_p1.5': 'Tweedie LightGBM, p=1.5',
    'tweedie_lightgbm_p1.7': 'Tweedie LightGBM, p=1.7',
  })[name] || name
