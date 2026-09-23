import {useEffect, useRef, useState} from 'react';

type Turbine = {id: number; name: string; has_data: boolean};
type ExecutionMode = 'tools' | 'llm';
type TimeConfig = {measurement_timezone: string; timestamp_semantics: string};
type ReplayRequest = TimeConfig & {
  turbine_ids: number[]; start_date: string; end_date: string; issue_hour_utc: number;
  horizon: number; execution_mode: ExecutionMode;
};
type ReplayJob = {
  id: string; status: 'queued' | 'running' | 'completed' | 'partial' | 'failed';
  progress: {completed: number; total: number; succeeded: number; failed: number};
  request: ReplayRequest; created_at: string; updated_at: string;
  input_snapshots?: Record<string, {rated_power_kw?: number | null; capacity_source_url?: string | null}>;
  errors: {turbine_id?: number; issue_at?: string; detail: string}[];
  summary?: {coverage_hours?: number; expected_hours?: number; [key: string]: unknown};
};
type Automation = TimeConfig & {
  enabled: boolean; turbine_ids: number[]; horizon: number; execution_mode: ExecutionMode;
};
type AutomationAPI = Omit<Automation, keyof TimeConfig> & {measurement_timezone: string | null; timestamp_semantics: string | null};
type AutomationEvent = {
  id: string; event: 'data_updated' | 'weather_updated'; turbine_id: number;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | 'superseded';
  created_at: string; updated_at: string; issue_at?: string; requested_issue_at?: string;
  error?: string; settings: AutomationAPI;
  report?: {id: string; status: string; summary: string; input_tokens?: number; output_tokens?: number};
};
type WeatherResult = {
  run: string; available_at: string; automation_event_id?: string; automation_error?: string;
};
type PanelProps = {turbines: Turbine[]; configured: boolean | null};
const DAY = 86400000;
const active = (job: ReplayJob) => job.status === 'queued' || job.status === 'running';
const timeLabel = (value: string) => value.slice(0, 16).replace('T', ' ');
const statusLabel: Record<ReplayJob['status'], string> = {
  queued: 'В очереди', running: 'Выполняется', completed: 'Завершён', partial: 'Завершён с пропусками', failed: 'Ошибка',
};

async function api<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === 'string' ? body.detail : `Ошибка запроса (${response.status}). Проверьте настройки.`;
    throw new Error(detail);
  }
  return response.json();
}

function write<T>(url: string, method: 'POST' | 'PUT', value: unknown) {
  return api<T>(url, {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
}

function parseDate(value: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const timestamp = Date.parse(`${value}T00:00:00Z`);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString().slice(0, 10) === value ? timestamp : null;
}

function TurbineChoices({turbines, selected, onChange}: {
  turbines: Turbine[]; selected: number[]; onChange: (ids: number[]) => void;
}) {
  return <div className="replay-turbines">{turbines.map(turbine => <label key={turbine.id} className="source-check">
    <input type="checkbox" checked={selected.includes(turbine.id)} disabled={!turbine.has_data}
      onChange={event => onChange(event.target.checked ? [...selected, turbine.id] : selected.filter(id => id !== turbine.id))}/>
    <span>{turbine.name}{!turbine.has_data && ' · нет измерений'}</span>
  </label>)}</div>;
}

function TimeFields({value, onChange}: {value: TimeConfig; onChange: (patch: Partial<TimeConfig>) => void}) {
  return <>
    <label>Часовой пояс CSV<select required value={value.measurement_timezone} onChange={event => onChange({measurement_timezone: event.target.value})}>
      <option value="">Не подтверждён</option><option value="Etc/GMT-6">UTC+6 · постоянное смещение</option><option value="Asia/Almaty">Asia/Almaty · с переводом часов</option><option value="UTC">UTC</option>
    </select></label>
    <label>Отметка времени CSV<select required value={value.timestamp_semantics} onChange={event => onChange({timestamp_semantics: event.target.value})}>
      <option value="">Не подтверждена</option><option value="interval_start">Начало 10-минутного интервала</option><option value="interval_end">Конец 10-минутного интервала</option>
    </select></label>
  </>;
}

function ExecutionField({value, onChange, configured}: {
  value: ExecutionMode; onChange: (mode: ExecutionMode) => void; configured: boolean | null;
}) {
  return <label>Исполнение<select value={value} onChange={event => onChange(event.target.value as ExecutionMode)}>
    <option value="tools">Инструменты · без LLM</option>
    <option value="llm" disabled={configured !== true}>LLM-агент · API-кредиты{configured === false ? ' · ключ не настроен' : ''}</option>
  </select></label>;
}

export function ReplayPanel({turbines, configured}: PanelProps) {
  const [request, setRequest] = useState<ReplayRequest>({
    turbine_ids: [], measurement_timezone: '', timestamp_semantics: '', start_date: '2026-01-31', end_date: '2026-02-28',
    issue_hour_utc: 12, horizon: 48, execution_mode: 'tools',
  });
  const [jobs, setJobs] = useState<ReplayJob[]>([]);
  const [selectedId, setSelectedId] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [historyError, setHistoryError] = useState('');
  const [pollError, setPollError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const pending = useRef(false);
  const selectedJob = jobs.find(job => job.id === selectedId) || jobs[0];
  const runningIds = jobs.filter(active).map(job => job.id).join(',');
  const first = parseDate(request.start_date);
  const last = parseDate(request.end_date);
  const days = first !== null && last !== null ? Math.floor((last - first) / DAY) + 1 : 0;
  const validIds = request.turbine_ids.filter(id => turbines.some(t => t.id === id && t.has_data));
  const lastIssue = last === null ? Infinity : last + request.issue_hour_utc * 3600000;
  const dateValid = days > 0 && days <= 366 && lastIssue <= Date.now();
  const executions = days > 0 ? days * validIds.length : 0;
  const withinBudget = request.execution_mode !== 'llm' || executions <= 128;
  const ready = dateValid && validIds.length > 0 && validIds.length <= 32 && withinBudget && !!request.measurement_timezone && !!request.timestamp_semantics;
  const update = (patch: Partial<ReplayRequest>) => setRequest(value => ({...value, ...patch}));

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setHistoryError('');
    api<{items: ReplayJob[]}>('/api/replays', {signal: controller.signal})
      .then(result => {
        if (controller.signal.aborted) return;
        setJobs(result.items);
        setSelectedId(current => result.items.some(job => job.id === current) ? current : (result.items.find(active)?.id || result.items[0]?.id || ''));
      })
      .catch(cause => {if (!controller.signal.aborted) setHistoryError((cause as Error).message);})
      .finally(() => {if (!controller.signal.aborted) setLoading(false);});
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    setPollError('');
    if (!runningIds) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    async function poll() {
      try {
        const updated = await Promise.all(runningIds.split(',').map(id => api<ReplayJob>(`/api/replays/${id}`, {signal: controller.signal})));
        if (controller.signal.aborted) return;
        failures = 0;
        setPollError('');
        setJobs(current => current.map(job => updated.find(item => item.id === job.id) || job));
      } catch (cause) {
        if (controller.signal.aborted) return;
        failures += 1;
        setPollError(`${(cause as Error).message} ${failures >= 3 ? 'Автообновление приостановлено. Нажмите «Обновить список», чтобы проверить состояние.' : 'Повторим проверку статуса.'}`);
      }
      if (!controller.signal.aborted && failures < 3) timer = setTimeout(() => void poll(), 4000);
    }
    void poll();
    return () => {controller.abort(); if (timer) clearTimeout(timer);};
  }, [runningIds, refresh]);

  async function launch() {
    if (pending.current || runningIds || loading || historyError) return;
    setError('');
    if (!ready) {setError('Выберите турбины с измерениями, подтвердите настройки времени и задайте корректный период выпусков.'); return;}
    if (request.execution_mode === 'llm' && configured !== true) {setError('Для LLM-режима нужен API-ключ на сервере.'); return;}
    pending.current = true;
    setSubmitting(true);
    try {
      const job = await write<ReplayJob>('/api/replays', 'POST', {...request, turbine_ids: validIds});
      setJobs(current => [job, ...current.filter(item => item.id !== job.id)]);
      setSelectedId(job.id);
    } catch (cause) {setError((cause as Error).message);}
    finally {pending.current = false; setSubmitting(false);}
  }

  return <>
    <section className="panel replay-panel">
      <div className="section-heading"><div><div className="eyebrow">День за днём</div><h2>Месячный прогон</h2></div><span className="tag">Февраль 2026</span></div>
      <p className="muted">Сервер последовательно воспроизведёт ежедневные выпуски прогноза. Февральский режим работает без новых измерений SCADA за февраль: отсутствие телеметрии и погоды отражается в результатах каждого выпуска.</p>
      {!turbines.length ? <p className="muted">Сначала <a href="/turbines">создайте турбины и импортируйте измерения</a>.</p> : <form onSubmit={event => {event.preventDefault(); void launch();}}>
        <fieldset disabled={submitting || !!runningIds}>
          <legend>Турбины для прогона</legend>
          <TurbineChoices turbines={turbines} selected={request.turbine_ids} onChange={ids => update({turbine_ids: ids})}/>
          <div className="filters"><TimeFields value={request} onChange={update}/></div>
          <p className="small muted">Настройки времени применяются ко всем выбранным турбинам. Если настройки различаются, создайте отдельные прогоны. UTC+6 — гипотеза, которую нужно выбрать явно.</p>
        </fieldset>
        <fieldset disabled={submitting || !!runningIds}>
          <legend>Расписание выпусков · UTC</legend>
          <div className="filters">
            <label>Первый день выпуска<input type="date" required value={request.start_date} max={request.end_date || undefined} onChange={event => update({start_date: event.target.value})}/></label>
            <label>Последний день, включительно<input type="date" required value={request.end_date} min={request.start_date || undefined} max={new Date().toISOString().slice(0, 10)} onChange={event => update({end_date: event.target.value})}/></label>
            <label>Ежедневный час выпуска, UTC<select value={request.issue_hour_utc} onChange={event => update({issue_hour_utc: Number(event.target.value)})}>{Array.from({length: 24}, (_, hour) => <option key={hour} value={hour}>{String(hour).padStart(2, '0')}:00</option>)}</select></label>
            <label>Горизонт<select value={request.horizon} onChange={event => update({horizon: Number(event.target.value)})}><option value={24}>24 часа</option><option value={48}>48 часов</option></select></label>
            <ExecutionField value={request.execution_mode} onChange={mode => update({execution_mode: mode})} configured={configured}/>
          </div>
          {executions > 0 && <p className="replay-estimate">{days} дней × {validIds.length} турбин = <strong>{executions} выпусков</strong> по {request.horizon} часов. Горизонты могут выходить за последний день выпусков.</p>}
          {first !== null && last !== null && !dateValid && <p className="error" role="alert">Выберите период до 366 дней с выпуском не позднее текущего момента.</p>}
          {!withinBudget && <p className="error" role="alert">Платный прогон ограничен 128 выпусками по всем выбранным турбинам. Уменьшите период или число турбин.</p>}
          {validIds.length > 32 && <p className="error" role="alert">В одном прогоне можно выбрать не более 32 турбин.</p>}
        </fieldset>
        <div className="notice compact">{request.execution_mode === 'tools' ? 'Режим «Инструменты» выполняет заданную последовательность расчёта без LLM и без расхода OpenAI-кредитов. Это детерминированный прогон, а не решения языковой модели.' : `Режим «LLM-агент» запускает настоящего агента для каждого выпуска и расходует API-кредиты. В выбранной очереди до ${executions} запусков агента.`} Заполнение формы само по себе ничего не запускает.</div>
        <div className="filters"><button disabled={submitting || loading || !!historyError || !!runningIds || !ready || (request.execution_mode === 'llm' && configured !== true)}>{submitting ? 'Создаём задачу…' : request.execution_mode === 'llm' ? 'Запустить прогон с LLM' : 'Запустить прогон инструментов'}</button></div>
        {!!runningIds && <p className="muted" role="status">Прогон уже выполняется на сервере. Дождитесь его завершения перед новым запуском. Закрытие страницы не отменяет принятую задачу.</p>}
      </form>}
      {error && <div className="error" role="alert">{error}</div>}
    </section>
    <section className="panel replay-results">
      <div className="section-heading"><h2>Прогоны и выгрузки</h2><button className="source-secondary" type="button" disabled={loading || submitting} onClick={() => setRefresh(value => value + 1)}>Обновить список</button></div>
      {loading && <p className="muted" role="status">Загружаем прогоны…</p>}
      {historyError && <div className="error" role="alert">{historyError}</div>}
      {pollError && <div className="error" role="alert">{pollError}</div>}
      {!!jobs.length && <label className="replay-job-picker">Сохранённый прогон<select value={selectedJob?.id || ''} onChange={event => setSelectedId(event.target.value)}>{jobs.map(job => <option value={job.id} key={job.id}>{timeLabel(job.created_at)} UTC · {job.request.start_date} — {job.request.end_date} · {statusLabel[job.status]}</option>)}</select></label>}
      {!loading && !historyError && !jobs.length && <p className="muted">Прогонов пока нет.</p>}
      {selectedJob && <article className="replay-job">
        <div className="section-heading"><h3>{statusLabel[selectedJob.status]}</h3><span className="tag">{selectedJob.request.execution_mode === 'llm' ? 'LLM-агент' : 'Инструменты · без LLM'}</span></div>
        <p className="muted">Выпуски {selectedJob.request.start_date} — {selectedJob.request.end_date}, ежедневно в {String(selectedJob.request.issue_hour_utc).padStart(2, '0')}:00 UTC · горизонт {selectedJob.request.horizon} ч.</p>
        <p className="small muted">Турбины: {selectedJob.request.turbine_ids.map(id => turbines.find(t => t.id === id)?.name || `#${id}`).join(', ')} · CSV: {selectedJob.request.measurement_timezone} · {selectedJob.request.timestamp_semantics === 'interval_start' ? 'начало интервала' : 'конец интервала'}.</p>
        <progress max={Math.max(selectedJob.progress.total, 1)} value={selectedJob.progress.completed} aria-label="Выполнено выпусков"/>
        <p role="status">Обработано {selectedJob.progress.completed} из {selectedJob.progress.total} · успешно {selectedJob.progress.succeeded} · с ошибками {selectedJob.progress.failed}.</p>
        {selectedJob.summary && typeof selectedJob.summary.coverage_hours === 'number' && typeof selectedJob.summary.expected_hours === 'number' && <p className="muted">Покрытие прогнозами: {selectedJob.summary.coverage_hours} из {selectedJob.summary.expected_hours} ожидаемых часовых отметок.</p>}
        {!active(selectedJob) && <div className="replay-exports">
          <a href={`/api/replays/${selectedJob.id}/export?kind=forecasts`} download>Прогнозы CSV ↓</a>
          <a href={`/api/replays/${selectedJob.id}/export?kind=coverage`} download>Покрытие CSV ↓</a>
          {selectedJob.request.turbine_ids.every(id => (selectedJob.input_snapshots?.[String(id)]?.rated_power_kw || 0) > 0) && <a href={`/api/replays/${selectedJob.id}/export?kind=plant`} download>ВЭС · кВт и кВт·ч CSV ↓</a>}
          <a href={`/api/replays/${selectedJob.id}/export?kind=report`} download>Отчёт JSON ↓</a>
        </div>}
        <p className="small muted">Выгрузка ВЭС доступна, если номинальные мощности всех выбранных турбин подтверждены до создания этого прогона. Пересчёт в кВт предполагает, что исходная мощность нормирована на номинал. Выгрузка суммирует только выбранные турбины; пропущенный прогноз не заменяется нулём.</p>
        <p className="notice compact">Готовность и покрытие показывают, где удалось получить прогноз. Без фактической мощности за февраль нельзя оценить MAE/RMSE этого месяца. Пропуски и ошибки не считаются успешными прогнозами.</p>
        {!!selectedJob.errors.length && <details><summary>Ошибки выпусков ({selectedJob.errors.length})</summary><div className="table-wrap"><table><thead><tr><th>Турбина</th><th>Момент выпуска, UTC</th><th>Причина</th></tr></thead><tbody>{selectedJob.errors.map((item, index) => <tr key={`${item.turbine_id}-${item.issue_at}-${index}`}><td>{item.turbine_id === undefined ? 'Весь прогон' : turbines.find(t => t.id === item.turbine_id)?.name || `#${item.turbine_id}`}</td><td>{item.issue_at ? timeLabel(item.issue_at) : '—'}</td><td>{item.detail}</td></tr>)}</tbody></table></div></details>}
      </article>}
    </section>
  </>;
}

const DEFAULT_AUTOMATION: Automation = {enabled: false, turbine_ids: [], measurement_timezone: '', timestamp_semantics: '', horizon: 48, execution_mode: 'tools'};
const editableAutomation = (value: AutomationAPI): Automation => ({...value, measurement_timezone: value.measurement_timezone || '', timestamp_semantics: value.timestamp_semantics || ''});

function WeatherRefresh({settings, turbines, disabled, onBusyChange, onUpdated}: {
  settings: Automation; turbines: Turbine[]; disabled: boolean;
  onBusyChange: (busy: boolean) => void; onUpdated: () => void;
}) {
  const [issue, setIssue] = useState('');
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<{id: number; issue: string; result?: WeatherResult; error?: string}[]>([]);
  const pending = useRef(false);
  const ids = settings.turbine_ids.filter(id => turbines.some(t => t.id === id));
  const timestamp = /^\d{4}-\d{2}-\d{2}T\d{2}:00$/.test(issue) ? Date.parse(`${issue}:00Z`) : NaN;
  const validIssue = Number.isFinite(timestamp) && new Date(timestamp).toISOString().slice(0, 16) === issue && timestamp <= Date.now();
  const paid = settings.enabled && settings.execution_mode === 'llm';

  async function refreshWeather() {
    if (pending.current || disabled || !validIssue || !ids.length) return;
    pending.current = true;
    setBusy(true);
    onBusyChange(true);
    setResults([]);
    const requestIssue = new Date(timestamp).toISOString();
    try {
      for (const id of ids) {
        try {
          const result = await write<WeatherResult>('/api/weather/gfs', 'POST', {turbine_id: id, issue_at: requestIssue, horizon: settings.horizon});
          setResults(current => [...current, {id, issue: requestIssue, result}]);
        } catch (cause) {setResults(current => [...current, {id, issue: requestIssue, error: (cause as Error).message}]);}
        onUpdated();
      }
    } finally {pending.current = false; setBusy(false); onBusyChange(false);}
  }

  return <section className="automation-weather">
    <h3>Обновить выпуск NOAA GFS</h3>
    <p className="muted">Загрузим архивный прогноз для сохранённого списка турбин: {ids.map(id => turbines.find(t => t.id === id)?.name || `#${id}`).join(', ') || 'турбины не выбраны'}. Горизонт — {settings.horizon} ч. Доступность выпуска проверяется на указанный момент UTC.</p>
    <form onSubmit={event => {event.preventDefault(); void refreshWeather();}}>
      <fieldset disabled={busy || disabled}>
        <div className="filters">
          <label>Момент расчёта, UTC<input type="datetime-local" required step={3600} value={issue} max={new Date().toISOString().slice(0, 13) + ':00'} onChange={event => setIssue(event.target.value)}/></label>
          <button type="submit" disabled={!validIssue || !ids.length}>{busy ? 'Обновляем погоду…' : settings.enabled ? 'Обновить погоду и пересчитать' : 'Обновить погоду'}</button>
        </div>
      </fieldset>
      <p className="notice compact">{settings.enabled ? paid ? 'Каждый новый погодный выпуск поставит пересчёт на этот момент в очередь LLM-агента и расходует API-кредиты. Повтор той же ревизии не создаёт новое событие.' : 'Каждый новый погодный выпуск поставит пересчёт на этот момент в очередь инструментов, без LLM. Повтор той же ревизии не создаёт новое событие.' : 'Автоматизация выключена: кнопка только загрузит и проверит погоду, без запуска агента.'} Загрузка Open-Meteo на странице архива не запускает этот пересчёт.</p>
    </form>
    {!!results.length && <ul className="automation-weather-results">{results.map(item => <li key={item.id}>
      <strong>{turbines.find(t => t.id === item.id)?.name || `#${item.id}`}</strong> · {timeLabel(item.issue)} UTC
      {item.error ? <p className="error" role="alert">{item.error}</p> : item.result && <>
        <p className="small muted">Погода проверена: выпуск {timeLabel(item.result.run)} UTC, опубликован к {timeLabel(item.result.available_at)} UTC.</p>
        <p className={item.result.automation_error ? 'error' : 'muted'}>{item.result.automation_error || (item.result.automation_event_id ? 'Событие принято. Результат пересчёта появится в журнале ниже.' : 'Нового события пересчёта нет: автоматизация выключена либо эта ревизия уже обработана.')}</p>
      </>}
    </li>)}</ul>}
  </section>;
}

function AutomationHistory({turbines, refreshVersion}: {turbines: Turbine[]; refreshVersion: number}) {
  const [events, setEvents] = useState<AutomationEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let failures = 0;
    setLoading(true);
    setError('');
    async function poll() {
      try {
        const result = await api<{items: AutomationEvent[]}>('/api/automation/events', {signal: controller.signal});
        if (controller.signal.aborted) return;
        setEvents(result.items);
        setError('');
        failures = 0;
        if (result.items.some(event => event.status === 'queued' || event.status === 'running')) timer = setTimeout(() => void poll(), 4000);
      } catch (cause) {
        if (controller.signal.aborted) return;
        failures += 1;
        setError(`${(cause as Error).message}${failures >= 3 ? ' Нажмите «Обновить журнал», чтобы повторить проверку.' : ''}`);
        if (failures < 3) timer = setTimeout(() => void poll(), 4000);
      } finally {if (!controller.signal.aborted) setLoading(false);}
    }
    void poll();
    return () => {controller.abort(); if (timer) clearTimeout(timer);};
  }, [refreshVersion, refresh]);
  const labels: Record<AutomationEvent['status'], string> = {queued: 'В очереди', running: 'Выполняется', completed: 'Готово', failed: 'Ошибка', cancelled: 'Отменено', superseded: 'Есть более новые измерения'};
  return <section className="automation-history">
    <div className="section-heading"><h3>Журнал автоматических пересчётов</h3><button className="source-secondary" type="button" disabled={loading} onClick={() => setRefresh(value => value + 1)}>Обновить журнал</button></div>
    {loading && <p className="muted" role="status">Загружаем события…</p>}
    {error && <p className="error" role="alert">{error}</p>}
    {!loading && !error && !events.length && <p className="muted">Событий пока нет. Включение автоматизации само по себе не запускает расчёт.</p>}
    {events.slice(0, 20).map(event => <details key={event.id}>
      <summary>{turbines.find(t => t.id === event.turbine_id)?.name || `#${event.turbine_id}`} · {event.event === 'weather_updated' ? 'Обновление GFS' : 'Импорт измерений'} · {labels[event.status]} · {timeLabel(event.created_at)} UTC</summary>
      <p className="small muted">{event.settings.execution_mode === 'llm' ? 'LLM-агент · API-кредиты' : 'Инструменты · без LLM'}{(event.issue_at || event.requested_issue_at) && ` · момент расчёта ${timeLabel(event.issue_at || event.requested_issue_at!)} UTC`}</p>
      {event.error && <p className="error">{event.error}</p>}
      {event.report && <><p>{event.report.summary}</p>{event.report.status === 'forecast_saved' && <p><a href={`/api/forecasts/${event.report.id}`} download={`forecast-${event.report.id}.json`}>Скачать прогноз JSON ↓</a></p>}</>}
    </details>)}
    {events.length > 20 && <p className="small muted">Показаны последние 20 событий.</p>}
  </section>;
}

export function AutomationPanel({turbines, configured}: PanelProps) {
  const [settings, setSettings] = useState<Automation>(DEFAULT_AUTOMATION);
  const [saved, setSaved] = useState<Automation | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [weatherBusy, setWeatherBusy] = useState(false);
  const [eventVersion, setEventVersion] = useState(0);
  const pending = useRef(false);
  const dirty = saved !== null && JSON.stringify(settings) !== JSON.stringify(saved);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError('');
    api<AutomationAPI>('/api/automation', {signal: controller.signal}).then(value => {
      if (!controller.signal.aborted) {const editable = editableAutomation(value); setSettings(editable); setSaved(editable); setNotice('');}
    }).catch(cause => {if (!controller.signal.aborted) setError((cause as Error).message);})
      .finally(() => {if (!controller.signal.aborted) setLoading(false);});
    return () => controller.abort();
  }, [refresh]);

  async function save() {
    if (pending.current || loading || weatherBusy || saved === null) return;
    setError('');
    setNotice('');
    const activeIds = settings.turbine_ids.filter(id => turbines.some(t => t.id === id && t.has_data));
    if (settings.enabled && (!activeIds.length || !settings.measurement_timezone || !settings.timestamp_semantics)) {setError('Для автоматизации выберите турбины с измерениями и укажите настройки времени CSV.'); return;}
    if (settings.enabled && settings.execution_mode === 'llm' && configured !== true) {setError('Для автоматического LLM-агента нужен настроенный API-ключ.'); return;}
    pending.current = true;
    setBusy(true);
    try {
      const response = await write<AutomationAPI>('/api/automation', 'PUT', {...settings, turbine_ids: activeIds, measurement_timezone: settings.measurement_timezone || null, timestamp_semantics: settings.timestamp_semantics || null});
      const value = editableAutomation(response);
      setSettings(value);
      setSaved(value);
      setNotice(value.enabled ? 'Автоматизация включена. Новые события будут обрабатываться с сохранёнными настройками.' : 'Автоматизация выключена. Новые события не будут запускать расчёт автоматически.');
    } catch (cause) {setError((cause as Error).message);}
    finally {pending.current = false; setBusy(false);}
  }

  const update = (patch: Partial<Automation>) => {setSettings(value => ({...value, ...patch})); setNotice('');};
  return <section className="panel replay-panel automation-panel">
    <div className="section-heading"><div><div className="eyebrow">Реакция на обновления</div><h2>Автоматический пересчёт</h2></div><span className="tag">{saved === null ? 'Статус не загружен' : saved.enabled ? 'Включён' : 'Выключен'}</span></div>
    <p className="muted">После импорта измерений сервер пересчитает последний сохранённый момент прогноза, а после обновления NOAA GFS — указанный момент. Без истории используются последние доступные измерения. По умолчанию автоматизация выключена.</p>
    {loading && <p className="muted" role="status">Загружаем настройки…</p>}
    <form onSubmit={event => {event.preventDefault(); void save();}}>
      <fieldset disabled={loading || busy || weatherBusy || saved === null}>
        <label className="source-check automation-switch"><input type="checkbox" checked={settings.enabled} onChange={event => update({enabled: event.target.checked})}/><span>Запускать расчёт при новых событиях</span></label>
        <TurbineChoices turbines={turbines} selected={settings.turbine_ids} onChange={ids => update({turbine_ids: ids})}/>
        {!turbines.length && <p className="muted"><a href="/turbines">Добавьте турбину и измерения</a>, чтобы настроить пересчёт.</p>}
        <div className="filters">
          <TimeFields value={settings} onChange={update}/>
          <label>Горизонт<select value={settings.horizon} onChange={event => update({horizon: Number(event.target.value)})}><option value={24}>24 часа</option><option value={48}>48 часов</option></select></label>
          <ExecutionField value={settings.execution_mode} onChange={mode => update({execution_mode: mode})} configured={configured}/>
        </div>
      </fieldset>
      <div className="notice compact">{settings.execution_mode === 'llm' ? 'LLM-режим расходует API-кредиты при каждом обработанном событии. Включение переключателя и сохранение разрешают эти последующие автоматические вызовы.' : 'Режим «Инструменты» выполняет заданный цикл без LLM и не расходует OpenAI-кредиты.'} Изменения применяются только после нажатия «Сохранить».</div>
      <div className="filters"><button type="submit" formNoValidate disabled={loading || busy || weatherBusy || saved === null || !dirty}>{busy ? 'Сохраняем…' : 'Сохранить настройки'}</button><button type="button" className="secondary-action" disabled={loading || busy || weatherBusy} onClick={() => setRefresh(value => value + 1)}>Перечитать настройки</button></div>
      {dirty && <p className="small muted">Есть несохранённые изменения. Сейчас на сервере автоматизация {saved?.enabled ? 'включена' : 'выключена'}.</p>}
    </form>
    {error && <div className="error" role="alert">{error}</div>}
    {notice && <p role="status" className="muted">{notice}</p>}
    {saved && <><WeatherRefresh settings={saved} turbines={turbines} disabled={loading || busy || dirty || (saved.enabled && saved.execution_mode === 'llm' && configured !== true)} onBusyChange={setWeatherBusy} onUpdated={() => setEventVersion(value => value + 1)}/>{dirty && <p className="small muted">Перед обновлением погоды сохраните настройки.</p>}</>}
    <AutomationHistory turbines={turbines} refreshVersion={eventVersion}/>
  </section>;
}
