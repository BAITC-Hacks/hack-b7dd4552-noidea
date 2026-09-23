import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './style.css';
import {SourcesPanel} from './SourcesPanel';
import {ForecastPage} from './ForecastPage';
import {DeletedTurbines, TurbineDelete} from './TurbineActions';

type Point = {time: string; [key: string]: string | number | null};
type Dataset = {
  has_data: boolean; id: number; name: string; latitude: number; longitude: number; rows: number;
  start: string; end: string; coverage: number; missing_slots: number; gap_count: number;
  complete_hours: number; partial_hours: number; missing_hours: number; hours: number;
  dataset_kind: string; source_name: string; sha256: string;
  largest_gaps: {start: string; end: string; missing_slots: number}[];
};
type Weather = {
  id: string; turbine_id: number; run: string; retrieved_at: string; model: string;
  points?: Point[]; cached?: boolean; hours: number; grid_latitude: number; grid_longitude: number;
  missing_values: Record<string, number>; sha256: string;
};
const n = (v: number) => v.toLocaleString('ru-RU');
const dt = (s: string) => s.slice(0, 16).replace('T', ' ');
async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Ошибка запроса (${response.status}). Проверьте параметры.`);
  }
  return response.json();
}

function Chart({points, metric, unit, color = '#127e72', fixed}: {
  points: Point[]; metric: string; unit: string; color?: string; fixed?: [number, number];
}) {
  const [hover, setHover] = useState<number | null>(null);
  const values = points.map(p => p[metric]).filter((v): v is number => typeof v === 'number');
  if (!points.length || !values.length) return <div className="empty">Нет полных измерений за выбранный период.</div>;
  const low = fixed ? fixed[0] : Math.min(...values);
  const high = fixed ? fixed[1] : Math.max(...values);
  const pad = fixed ? 0 : Math.max((high - low) * .12, .5);
  const min = low - pad, max = high + pad;
  const x = (i: number) => 58 + i * 886 / Math.max(points.length - 1, 1);
  const y = (v: number) => 225 - (v - min) / Math.max(max - min, 1e-6) * 195;
  let pen = false;
  const path = points.map((p, i) => {
    if (typeof p[metric] !== 'number') { pen = false; return ''; }
    const command = pen ? 'L' : 'M'; pen = true;
    return `${command}${x(i).toFixed(2)},${y(p[metric] as number).toFixed(2)}`;
  }).join(' ');
  const selected = hover === null ? null : points[Math.min(hover, points.length - 1)];
  return <div className="chart">
    <div className="chart-readout">{selected ? `${dt(selected.time)} · ${typeof selected[metric] === 'number' ? (selected[metric] as number).toFixed(2) + ' ' + unit : 'неполный час / пропуск'}` : `${unit} · наведите на график для просмотра значений`}</div>
    <svg viewBox="0 0 980 270" role="img" aria-label={`График: ${metric}, ${unit}`}
      onMouseLeave={() => setHover(null)} onMouseMove={e => {
        const rect = e.currentTarget.getBoundingClientRect();
        const position = (e.clientX - rect.left) / rect.width * 980;
        setHover(Math.max(0, Math.min(points.length - 1, Math.round((position - 58) / 886 * (points.length - 1)))));
      }}>
      {[0, 1, 2, 3, 4].map(i => { const v = min + (max - min) * i / 4; return <g key={i}><line x1="58" x2="944" y1={y(v)} y2={y(v)} stroke="#e5eae8"/><text x="47" y={y(v) + 4} textAnchor="end">{v.toFixed(1)}</text></g>; })}
      <path d={path} fill="none" stroke={color} strokeWidth="2.4" strokeLinejoin="round"/>
      {points.length === 1 && typeof points[0][metric] === 'number' && <circle cx={x(0)} cy={y(points[0][metric] as number)} r="4" fill={color}/>}
      {[0, .25, .5, .75, 1].map(f => { const i = Math.round(f * (points.length - 1)); return <text key={f} x={x(i)} y="251" textAnchor={f === 0 ? 'start' : f === 1 ? 'end' : 'middle'}>{dt(points[i].time)}</text>; })}
      {hover !== null && <line x1={x(hover)} x2={x(hover)} y1="25" y2="225" stroke={color} strokeDasharray="4 4"/>}
    </svg>
  </div>;
}

function WeatherPanel({turbineId}: {turbineId: number}) {
  const [runDate, setRunDate] = useState('');
  const [hour, setHour] = useState('00');
  const [runs, setRuns] = useState<Weather[]>([]);
  const [weather, setWeather] = useState<Weather | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [variable, setVariable] = useState('wind_speed_100m');
  useEffect(() => {
    let active = true;
    api<{items: Weather[]}>(`/api/weather?turbine_id=${turbineId}`).then(data => {
      if (active) setRuns(data.items);
    }).catch(e => { if (active) setError(e.message); });
    return () => { active = false; };
  }, [turbineId]);
  async function load(id?: string) {
    setBusy(true); setError(''); setWeather(null);
    try {
      const value = await api<Weather>(id ? `/api/weather/${id}` : '/api/weather', id ? undefined : {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({turbine_id: turbineId, run: `${runDate}T${hour}:00:00+00:00`}),
      });
      setWeather(value);
      setRuns((await api<{items: Weather[]}>(`/api/weather?turbine_id=${turbineId}`)).items);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  }
  const variables: Record<string, [string, string]> = {
    wind_speed_100m: ['Ветер · 100 м', 'м/с'], wind_speed_10m: ['Ветер · 10 м', 'м/с'],
    temperature_2m: ['Температура · 2 м', '°C'], wind_direction_100m: ['Направление · 100 м', '°'],
  };
  return <section className="panel weather-panel">
    <div className="section-heading"><div><div className="eyebrow">Внешний источник</div><h2>Архив прогнозов погоды</h2></div><span className="tag">ECMWF IFS · UTC</span></div>
    <p className="muted">Отдельный выпуск прогноза на 72 часа. Загрузка по координатам выбранной турбины.</p>
    <form className="filters" onSubmit={e => { e.preventDefault(); void load(); }}>
      <label>Дата выпуска<input type="date" value={runDate} min="2024-03-01" max={new Date().toISOString().slice(0,10)} onChange={e => setRunDate(e.target.value)} required/></label>
      <label>Час, UTC<select value={hour} onChange={e => setHour(e.target.value)}>{['00', '06', '12', '18'].map(h => <option key={h}>{h}</option>)}</select></label>
      <button disabled={busy}>{busy ? 'Загрузка…' : 'Загрузить выпуск ↗'}</button>
      {runs.length > 0 && <label className="saved">Сохранённые выпуски<select disabled={busy} value={weather?.id ?? ''} onChange={e => {if (e.target.value) void load(e.target.value);}}><option value="">Выбрать из кеша</option>{runs.map(r => <option key={r.id} value={r.id}>{dt(r.run)} UTC</option>)}</select></label>}
    </form>
    {error && <div className="error" role="alert">{error}</div>}
    {weather?.points ? <>
      <div className="weather-info"><span className="dot"/> {weather.hours} часов · выпуск {dt(weather.run)} UTC · {weather.cached ? 'из кеша' : 'сохранён'}<a href={`/api/weather/${weather.id}/raw`}>Исходный JSON ↓</a></div>
      <div className="tabs">{Object.entries(variables).map(([key, [label]]) => <button key={key} className={key === variable ? 'active' : ''} onClick={() => setVariable(key)}>{label}</button>)}</div>
      <Chart points={weather.points} metric={variable} unit={variables[variable][1]} color="#587ca9"/>
      <p className="muted small">Ячейка модели: {weather.grid_latitude.toFixed(5)}, {weather.grid_longitude.toFixed(5)} · получено {dt(weather.retrieved_at)} UTC · отсутствует значений: {Object.values(weather.missing_values).reduce((a,b) => a+b,0)}</p>
    </> : !busy && <div className="weather-empty"><span>↗</span><div><strong>Посмотрите, что прогнозировали раньше</strong><p>Выберите дату и выпуск. Повторная загрузка использует сохранённый ответ.</p></div></div>}
    <div className="notice compact">Время выпуска не равно времени публикации. Историческая доступность ещё не подтверждена: эти данные пока нельзя считать готовыми для честного бэктеста.</div>
    <p className="small muted attribution">Погодные данные: <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">Open-Meteo</a> / ECMWF · <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noreferrer">CC BY 4.0</a></p>
  </section>;
}

function TurbineForm({onCreated, onCancel}: {onCreated: (t: Dataset) => Promise<void>; onCancel: () => void}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const submitting = useRef(false);
  return <section className="panel turbine-form"><h2>Параметры турбины</h2><p className="muted">Координаты нужны для подбора погодных данных. CSV можно импортировать после создания.</p>
    <form onSubmit={async e => {
      e.preventDefault();
      if (submitting.current) return;
      const data = new FormData(e.currentTarget);
      submitting.current = true; setBusy(true); setError('');
      try {
        const t = await api<Dataset>('/api/turbines', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({name: data.get('name'), latitude: Number(data.get('latitude')), longitude: Number(data.get('longitude'))})});
        await onCreated(t);
      } catch (e) { setError((e as Error).message); }
      finally { submitting.current = false; setBusy(false); }
    }}>
      <fieldset className="turbine-form-fields" disabled={busy}>
        <label className="turbine-form-wide">Название<input name="name" required maxLength={100} placeholder="Например, Турбина 1" autoFocus/></label>
        <label>Широта, °<input name="latitude" required type="number" step="any" min="-90" max="90" placeholder="43.645150"/></label>
        <label>Долгота, °<input name="longitude" required type="number" step="any" min="-180" max="180" placeholder="78.535604"/></label>
        <div className="filters turbine-form-wide">
          <button type="submit">{busy ? 'Сохраняем…' : 'Создать турбину'}</button>
          <button type="button" className="secondary-action" onClick={onCancel}>Отмена</button>
        </div>
      </fieldset>
    </form>{error && <div className="error" role="alert">{error}</div>}
  </section>;
}

function ImportPanel({turbine, onImported, disabled, onBusyChange}: {turbine: Dataset; onImported: () => Promise<void>; disabled: boolean; onBusyChange: (busy: boolean) => void}) {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  useEffect(() => { onBusyChange(busy); }, [busy, onBusyChange]);
  useEffect(() => () => onBusyChange(false), [onBusyChange]);
  return <section className="panel"><div className="section-heading"><h2>Измерения · {turbine.name}</h2><span className="tag">Широта {turbine.latitude}° · долгота {turbine.longitude}°</span></div>
    <p className="muted">Загрузите CSV в UTF-8, до 25 МБ. Исходный файл сохранится, данные пройдут проверку перед импортом.</p>
    {turbine.has_data && <p className="muted">Новый импорт заменит активный набор измерений этой турбины. Предыдущие исходники сохранятся.</p>}
    <form className="filters" onSubmit={async e => {e.preventDefault(); if (!file || busy || disabled) return; setBusy(true); setError(''); setSuccess('');
      const body = new FormData(); body.append('file', file);
      try { const result = await api<Dataset>(`/api/turbines/${turbine.id}/import`, {method: 'POST', body}); await onImported(); setSuccess(`Импортировано ${n(result.rows)} строк. Полных часов: ${n(result.complete_hours)}.`); }
      catch (e) { setError((e as Error).message); } finally { setBusy(false); }
    }}><label>Файл измерений<input type="file" accept=".csv,text/csv" disabled={busy || disabled} required onChange={e => setFile(e.target.files?.[0] ?? null)}/></label><button disabled={!file || busy || disabled}>{busy ? 'Проверяем и импортируем…' : 'Импортировать CSV'}</button></form>
    {error && <div className="error" role="alert">{error}</div>}{success && <p role="status">{success}</p>}
    <details><summary>Какие столбцы нужны?</summary><p>Статистическое время; Средняя скорость ветра(m/s); Нормализованная активная мощность; Средняя температура окружающей среды(°C).</p><p>Разделитель — запятая; десятичный знак — точка. Время: YYYY-MM-DD HH:MM:SS, шаг 10 минут. Мощность в диапазоне 0–1. Файлы организаторов подходят без изменений.</p></details>
  </section>;
}

function App() {
  const pathname = window.location.pathname.replace(/\/$/, '');
  const page = pathname === '/turbines/new' ? 'new-turbine' : pathname === '/sources' ? 'sources' : pathname === '/forecast' ? 'forecast' : pathname === '/weather' ? 'weather' : 'turbines';
  const pageTitle = {'new-turbine':'Добавить турбину',turbines:'Турбины',forecast:'Прогноз',weather:'Архив погоды',sources:'Источники'}[page];
  useEffect(() => { document.title = `NoIdea · ${pageTitle}`; }, [pageTitle]);
  useEffect(() => {
    if (!pathname) {
      const legacy:Record<string,string> = {'#sources':'/sources','#agent':'/forecast','#weather':'/weather'};
      window.location.replace(legacy[window.location.hash] || '/turbines');
    }
  }, []);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [id, setId] = useState<number | null>(null);
  const [turbineActionBusy, setTurbineActionBusy] = useState(false);
  const [importBusy, setImportBusy] = useState(false);
  const [trashVersion, setTrashVersion] = useState(0);
  const [start, setStart] = useState('2026-01-25');
  const [end, setEnd] = useState('2026-01-31');
  const [points, setPoints] = useState<Point[]>([]);
  const [metric, setMetric] = useState('power');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [catalogError, setCatalogError] = useState('');
  const [sourcesBusy, setSourcesBusy] = useState(false);
  const request = useRef(0);
  const catalogRequest = useRef(0);
  async function refresh(selectId?: number) {
    const current = ++catalogRequest.current;
    try {
      const data = await api<{items: Dataset[]}>('/api/turbines');
      if (current !== catalogRequest.current) return;
      setDatasets(data.items); setCatalogError('');
      const requested = selectId ?? id ?? (Number(new URLSearchParams(window.location.search).get('selected')) || Number(localStorage.getItem('wind-selected-turbine')));
      setId(data.items.some(t => t.id === requested) ? requested : data.items[0]?.id ?? null);
    } catch (e) { if (current === catalogRequest.current) setCatalogError((e as Error).message); }
  }
  async function onDeleted(deletedId: number) {
    setDatasets(current => current.filter(t => t.id !== deletedId));
    setId(current => current === deletedId ? null : current);
    if (Number(localStorage.getItem('wind-selected-turbine')) === deletedId) localStorage.removeItem('wind-selected-turbine');
    setTrashVersion(value => value + 1);
    await refresh();
  }

  useEffect(() => { void refresh(); }, []);
  useEffect(() => { if (id !== null) localStorage.setItem('wind-selected-turbine', String(id)); }, [id]);
  async function loadSeries(turbine: number, from: string, to: string) {
    const current = ++request.current; setLoading(true); setError(''); setPoints([]);
    try {
      const data = await api<{points: Point[]}>(`/api/turbines/${turbine}/series?start=${from}&end=${to}`);
      if (current === request.current) setPoints(data.points);
    } catch (e) { if (current === request.current) setError((e as Error).message); }
    finally { if (current === request.current) setLoading(false); }
  }
  useEffect(() => {
    ++request.current; setPoints([]); setError(''); setLoading(false);
    const selected = datasets.find(t => t.id === id);
    if (page === 'turbines' && selected?.has_data) {
      const last = selected.end.slice(0,10);
      const first = new Date(Date.parse(last) - 6 * 86400000).toISOString().slice(0,10);
      setStart(first); setEnd(last); void loadSeries(selected.id, first, last);
    }
  }, [id, datasets]);
  const d = datasets.find(d => d.id === id);
  const metrics: Record<string, [string, string]> = {power: ['Мощность', 'доля номинала'], wind_speed: ['Ветер', 'м/с'], temperature: ['Температура', '°C']};
  return <div className="app">
    <aside><div className="brand"><span className="brand-icon">⌁</span><div>NoIdea<span>WIND INTELLIGENCE</span></div></div><div className="nav-label">Рабочее пространство</div><nav aria-label="Навигация">{[
      ['turbines', '▦', 'Турбины'], ['forecast', '◎', 'Прогноз'], ['weather', '↗', 'Архив погоды'], ['sources', '⚙', 'Источники'],
    ].map(([section, icon, label]) => {
      const selected = page === section || (page === 'new-turbine' && section === 'turbines');
      return <a key={section} className={`nav-item${selected ? ' selected' : ''}`} href={`/${section}`} aria-current={selected ? 'page' : undefined}>{icon} <span>{label}</span></a>;
    })}</nav><div className="aside-bottom"><span className="dot"/> HackAlem AI<div>Этап 1 · Исследование данных</div></div></aside>
    <main><header><span>Проект / <strong>{pageTitle}</strong></span><a href="/docs" target="_blank" rel="noreferrer">API ↗</a></header>
      {page === 'new-turbine' ? <>
        <div className="page-title"><div><div className="eyebrow">Новая турбина</div><h1>Добавить турбину</h1><p>Укажите название и местоположение.</p></div></div>
        <TurbineForm onCancel={() => window.location.assign('/turbines')} onCreated={async t => { window.location.assign(`/turbines?selected=${t.id}`); }}/>
      </> : page === 'sources' ? <>
        <div className="page-title"><div><div className="eyebrow">Погодные данные</div><h1>Источники</h1><p>Выберите источник или добавьте новый сайт.</p></div></div>
        {catalogError && <div className="error" role="alert">{catalogError}</div>}
        {datasets.length > 0 && <label className="source-turbine">Турбина для подбора данных<select disabled={sourcesBusy} value={id??''} onChange={e=>setId(Number(e.target.value))}>{datasets.map(t=><option key={t.id} value={t.id}>{t.name}{t.has_data?'':' · нет измерений'}</option>)}</select></label>}
        <SourcesPanel turbine={d} onBusyChange={setSourcesBusy}/>
      </> : page === 'forecast' ? <>
        <div className="page-title"><div><div className="eyebrow">Расчёт мощности</div><h1>Прогноз</h1><p>Выберите турбины и период, на который нужен прогноз.</p></div></div>
        {catalogError && <div className="error" role="alert">{catalogError}</div>}
        <ForecastPage turbines={datasets}/>
      </> : page === 'weather' ? <>
        <div className="page-title"><div><div className="eyebrow">Open-Meteo / ECMWF</div><h1>Архив погоды</h1><p>Просмотр отдельных исторических выпусков прогноза для выбранной турбины.</p></div></div>
        {catalogError && <div className="error" role="alert">{catalogError}</div>}
        {datasets.length > 0 ? <>
          <label className="source-turbine">Турбина<select value={id??''} onChange={e=>setId(Number(e.target.value))}>{datasets.map(t=><option key={t.id} value={t.id}>{t.name}</option>)}</select></label>
          {d && <WeatherPanel key={d.id} turbineId={d.id}/>}
        </> : <section className="panel"><h2>Добавьте турбину</h2><p className="muted">Координаты турбины нужны для загрузки погодных данных.</p><a href="/turbines">Перейти к турбинам →</a></section>}
      </> : <>
      <div className="page-title" id="measurements"><div><div className="eyebrow">От измерений к прогнозу</div><h1>Турбины</h1><p>Добавьте турбину и загрузите измерения, чтобы начать работу с данными.</p></div><span className="stage">Этап 01 / Данные</span></div>
      <div className="turbines">{datasets.map(t => <button key={t.id} className={t.id===id?'active':''} disabled={turbineActionBusy || importBusy} onClick={() => setId(t.id)}>{t.name}<span>{t.has_data ? 'Измерения загружены' : 'Нет измерений'}</span></button>)}<a className="add-turbine-link" href="/turbines/new">+ Добавить турбину</a></div>
      {d && <TurbineDelete key={d.id} turbine={d} disabled={turbineActionBusy || importBusy} onBusyChange={setTurbineActionBusy} onDeleted={onDeleted}/>}
      <DeletedTurbines version={trashVersion} disabled={turbineActionBusy || importBusy} onBusyChange={setTurbineActionBusy} onRestored={async restoredId => { await refresh(restoredId); }}/>
      {!datasets.length && <div className="panel"><h2>Начните со своей турбины</h2><p className="muted">1. Укажите название и координаты. 2. Импортируйте CSV с измерениями. 3. Проверьте данные и загрузите архив погоды.</p><div className="notice compact">После импорта откройте «Прогноз»: там показано, есть ли проверенная ML-модель для этой турбины. Если её нет, доступен резервный прогноз по последней мощности.</div></div>}
      {d && <ImportPanel key={`import-${d.id}`} turbine={d} onImported={() => refresh(d.id)} disabled={turbineActionBusy} onBusyChange={setImportBusy}/>}
      {d?.has_data && <div className="turbine-actions"><a href="/forecast">Получить прогноз →</a><a href="/sources">Подключить источник погоды →</a></div>}
      {catalogError && <div className="error" role="alert">{catalogError}</div>}
      {d?.has_data && <><div className="stats">
        <div><span>Исходных измерений</span><strong>{n(d.rows)}</strong><small>Шаг 10 минут</small></div>
        <div><span>Покрытие периода</span><strong>{(d.coverage*100).toFixed(2)}<em>%</em></strong><small>{dt(d.start).slice(0,10)} — {dt(d.end).slice(0,10)}</small></div>
        <div><span>Пропущено отметок</span><strong>{n(d.missing_slots)}</strong><small>{n(d.gap_count)} разрывов ряда</small></div>
        <div><span>Полных часов</span><strong>{n(d.complete_hours)}</strong><small>из {n(d.hours)} · по 6 измерений</small></div>
      </div>{d.dataset_kind==='demo' && <div className="notice">Демонстрационная выборка: последние 7 дней исходных CSV. Для полной истории импортируйте файлы по инструкции в README.</div>}</>}
      {d?.has_data && <><section className="panel"><div className="section-heading"><div><div className="eyebrow">Наблюдения</div><h2>История измерений</h2></div><span className="tag">Почасовые данные</span></div>
        <form className="filters" onSubmit={e=>{e.preventDefault(); void loadSeries(d.id,start,end);}}>
          <label>Начало периода<input type="date" required value={start} onChange={e=>setStart(e.target.value)}/></label><label>Конец периода<input type="date" required value={end} onChange={e=>setEnd(e.target.value)}/></label>
          <button disabled={loading}>Показать</button><a className="export" href={`/api/turbines/${id}/export?start=${start}&end=${end}`}>Скачать CSV ↓</a>
        </form>
        <div className="tabs">{Object.entries(metrics).map(([key,[label]])=><button key={key} className={key===metric?'active':''} onClick={()=>setMetric(key)}>{label}</button>)}</div>
        {error && <div className="error" role="alert">{error}</div>}
        {loading ? <div className="empty" role="status">Загружаем измерения…</div> : <Chart points={points} metric={metric} unit={metrics[metric][1]} fixed={metric==='power'?[0,1]:undefined}/>}
        <div className="chart-footer"><span><i/> Полные часы · среднее 6 измерений</span><span>Неполные часы — разрывы, не нули</span></div>
      </section>
      <div className="notice"><strong>Время источника не подтверждено.</strong> Сопоставление с погодой поддерживает постоянный UTC+6, но настройки часов оборудования ещё не подтверждены. Измерения пока не совмещены с погодой UTC. Мощность нормализована, это не кВт·ч.</div>
      </>}
      {d?.has_data && <section className="panel"><div className="section-heading"><div><div className="eyebrow">Контроль качества</div><h2>Полнота и происхождение</h2></div><span className="tag">Без заполнения пропусков</span></div>
        <div className="quality-bar"><span style={{width:`${100*d.complete_hours/d.hours}%`}}/><span style={{width:`${100*d.partial_hours/d.hours}%`}}/><span style={{width:`${100*d.missing_hours/d.hours}%`}}/></div>
        <div className="quality-legend"><span>● Полные: {n(d.complete_hours)}</span><span>● Неполные: {n(d.partial_hours)}</span><span>● Нет валидных значений: {n(d.missing_hours)}</span></div>
        {d.largest_gaps.length>0 ? <div className="table-wrap"><table><caption>Крупнейшие разрывы исходного ряда</caption><thead><tr><th>Первая отсутствующая отметка</th><th>Последняя отсутствующая отметка</th><th>Пропущено × 10 мин</th></tr></thead><tbody>{d.largest_gaps.map(g=><tr key={g.start}><td>{dt(g.start)}</td><td>{dt(g.end)}</td><td>{n(g.missing_slots)}</td></tr>)}</tbody></table></div> : <p className="muted">Пропусков временных отметок в импортированном периоде нет.</p>}
        <details><summary>Исходный файл и контрольная сумма</summary><p>{d.source_name}</p><code>SHA-256: {d.sha256}</code></details>
      </section>}
      </>}
      <footer>NoIdea / HackAlem AI <span>Ваши файлы сохраняются на сервере приложения. LLM запускается только по кнопке.</span></footer>
    </main>
  </div>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
