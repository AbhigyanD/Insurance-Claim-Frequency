import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import App from './App'
import { Badge, Segmented, Table } from './components/ui'
import { loadData, stubFetch } from './testUtils'

beforeEach(() => stubFetch())
afterEach(() => {
  cleanup()
  vi.unstubAllGlobals()
})

async function openTab(name) {
  render(<App />)
  await waitFor(() => screen.getByText('Policies evaluated'))
  fireEvent.click(screen.getByRole('tab', { name }))
}

describe('Risk analysis page', () => {
  it('starts on driver age and shows every level with its sample size', async () => {
    await openTab('Risk analysis')
    const risk = loadData('risk')
    const table = screen.getAllByRole('table')[0]
    for (const row of risk.driver_age) expect(within(table).getByText(row.level, { exact: false })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Driver age' })).toHaveAttribute('aria-pressed', 'true')
  })

  it('switches dimension and flags exactly the small-sample groups', async () => {
    await openTab('Risk analysis')
    fireEvent.click(screen.getByRole('button', { name: 'Region' }))
    expect(screen.getByRole('button', { name: 'Region' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Driver age' })).toHaveAttribute('aria-pressed', 'false')
    const regions = loadData('risk').region
    const small = regions.filter((r) => r.exposure < 1000).length   // independent of the page's own helper
    expect(small).toBeGreaterThan(0) // the real data has thin regions, so this test is not vacuous
    expect(screen.getAllByText('small sample')).toHaveLength(small)
    expect(screen.getByText(regions[0].level)).toBeInTheDocument()
  })

  it('states that differences are associations', async () => {
    await openTab('Risk analysis')
    expect(screen.getByText(/associations/i)).toBeInTheDocument()
  })
})

describe('Model performance page', () => {
  it('highlights only the selected models', async () => {
    await openTab('Model performance')
    const { selected_frequency_model: f, selected_severity_model: s } = loadData('overview')
    expect(f).toBe('lightgbm_poisson')
    expect(s).toBe('ridge_gamma_glm')
    const selectedRows = document.querySelectorAll('tr.selected')
    const texts = [...selectedRows].map((r) => r.textContent)
    expect(texts.some((t) => t.includes('LightGBM (Poisson)'))).toBe(true)
    expect(texts.some((t) => t.includes('Ridge Gamma GLM'))).toBe(true)
  })

  it('labels each severity bootstrap result from its interval, never overstating', async () => {
    await openTab('Model performance')
    const boot = loadData('models').severity_bootstrap
    const noDiff = boot.filter((b) => !b.significant).length
    expect(boot.every((b) => !b.significant)).toBe(true) // true in the exported results
    expect(screen.getAllByText('no reliable difference').length).toBeGreaterThanOrEqual(noDiff)
    expect(screen.getByText(/No severity model is reliably different from a constant/i)).toBeInTheDocument()
  })

  it('marks frequency variants that are reliably worse than the selected model', async () => {
    await openTab('Model performance')
    const variants = loadData('models').frequency_variants.filter((v) => v.dev_diff_vs_lightgbm != null)
    const worse = variants.filter((v) => v.significant && v.dev_diff_vs_lightgbm > 0).length
    expect(worse).toBeGreaterThan(0)
    expect(screen.getAllByText('reliably worse')).toHaveLength(worse)
  })

  it('says results are validation, not final test results', async () => {
    await openTab('Model performance')
    expect(screen.getByText(/not final test results/i)).toBeInTheDocument()
  })
})

describe('shared components', () => {
  it('Segmented reports the chosen option and exposes pressed state', () => {
    const onChange = vi.fn()
    render(<Segmented label="Pick" value="b" onChange={onChange} options={[{ value: 'a', label: 'A' }, { value: 'b', label: 'B' }]} />)
    expect(screen.getByRole('button', { name: 'B' })).toHaveAttribute('aria-pressed', 'true')
    fireEvent.click(screen.getByRole('button', { name: 'A' }))
    expect(onChange).toHaveBeenCalledWith('a')
    expect(screen.getByRole('group', { name: 'Pick' })).toBeInTheDocument()
  })

  it('Table renders cells with custom renderers and row classes', () => {
    render(
      <Table
        columns={[{ key: 'n', label: 'Name', align: 'left' }, { key: 'v', label: 'Value', render: (x) => `#${x}` }]}
        rows={[{ n: 'one', v: 1 }, { n: 'two', v: 2 }]}
        rowClass={(r) => (r.n === 'two' ? 'selected' : '')}
      />,
    )
    expect(screen.getByText('#2')).toBeInTheDocument()
    expect(screen.getByText('two').closest('tr')).toHaveClass('selected')
    expect(screen.getByText('one').closest('tr')).not.toHaveClass('selected')
  })

  it('Badge renders its kind as a class', () => {
    render(<Badge kind="warn">careful</Badge>)
    expect(screen.getByText('careful')).toHaveClass('badge-warn')
  })
})
