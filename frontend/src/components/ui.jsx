import React from 'react'

export function Kpi({ label, value, sub, tone }) {
  return (
    <div className={`kpi${tone ? ` kpi-${tone}` : ''}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

export function Card({ title, subtitle, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {title && <h3>{title}</h3>}
      {subtitle && <p className="muted">{subtitle}</p>}
      {children}
    </section>
  )
}

export function Callout({ kind = 'info', children }) {
  return (
    <div className={`callout callout-${kind}`} role="note">
      {children}
    </div>
  )
}

export function Badge({ kind, children }) {
  return <span className={`badge badge-${kind}`}>{children}</span>
}

// columns: [{ key, label, render?, align? }]
export function Table({ columns, rows, rowClass }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.align === 'left' ? 'left' : ''}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={rowClass ? rowClass(r) : ''}>
              {columns.map((c) => (
                <td key={c.key} className={c.align === 'left' ? 'left' : ''}>
                  {c.render ? c.render(r[c.key], r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Segmented({ options, value, onChange, label }) {
  return (
    <div className="segmented" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" className={o.value === value ? 'active' : ''} aria-pressed={o.value === value} onClick={() => onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  )
}
