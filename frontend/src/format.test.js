import { describe, expect, it } from 'vitest'
import { describeRatio, fmtInt, fmtMoney, fmtMoneyCompact, fmtNum, fmtPct, isSmallSample, prettyModel } from './format'

describe('format helpers', () => {
  it('formats numbers and handles missing values', () => {
    expect(fmtInt(135598)).toBe('135,598')
    expect(fmtMoney(2187.4)).toBe('€2,187')
    expect(fmtMoneyCompact(11283713)).toBe('€11.3M')
    expect(fmtNum(0.233839, 4)).toBe('0.2338')
    expect(fmtPct(0.0734, 2)).toBe('7.34%')
    for (const f of [fmtInt, fmtMoney, fmtMoneyCompact, fmtNum, fmtPct]) expect(f(null)).toBe('–')
  })

  it('describes actual/expected ratios in plain language', () => {
    expect(describeRatio(0.92)).toBe('8% below')
    expect(describeRatio(1.25)).toBe('25% above')
    expect(describeRatio(1.01)).toBe('in line with')
    expect(describeRatio(null)).toBe('')
  })

  it('flags small samples and names models', () => {
    expect(isSmallSample(999)).toBe(true)
    expect(isSmallSample(1000)).toBe(false)
    expect(prettyModel('lightgbm_poisson')).toBe('LightGBM (Poisson)')
    expect(prettyModel('unknown_model')).toBe('unknown_model')
  })
})
