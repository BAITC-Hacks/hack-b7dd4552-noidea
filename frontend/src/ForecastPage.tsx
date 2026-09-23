import {useEffect, useRef, useState} from 'react';

type Turbine = {id: number; name: string; has_data: boolean; start?: string; end?: string};
type TimeSettings = {timezone: string; semantics: string};
type AgentRequest = {
  turbine_id: number; issue_at: string; horizon: number; measurement_timezone: string | null;
  timestamp_semantics: string | null; event: 'manual';
};
type ToolCheck = {tool: string; result: Record<string, unknown>};
type Preflight = {ok: boolean; checks: ToolCheck[]};
type AgentReport = {
  id: string; status: string; summary?: string; error?: string; request: AgentRequest;
  input_tokens: number; output_tokens: number; steps: (ToolCheck & {step: number})[];
};
type BatchResult = {
  turbine: Turbine; request: AgentRequest; phase: 'queued' | 'checking' | 'ready' | 'blocked' | 'running' | 'done' | 'error' | 'cancelled';
  preflight?: Preflight; report?: AgentReport; error?: string;
};
type MLModel = {
  turbine_id: number; model_type: string; timezone: string; timestamp_semantics: string;
  usable_from: string; training_end: string; promoted: boolean; source_matches: boolean;
  validation_mae?: number | null; persistence_validation_mae?: number | null;
  validation_start?: string; validation_end?: string; control_start?: string; control_end?: string;
};

const HOUR = 3600000;
const shownTime = (value: string) => value.slice(0, 16).replace('T', ' ');
const formatInput = (value: number) => new Date(value).toISOString().slice(0, 16);
function requestPeriod(request: AgentRequest): string {
  const issue = Date.parse(request.issue_at);
  if (!Number.isFinite(issue) || !Number.isFinite(request.horizon)) return 'Период не указан';
  return `${shownTime(new Date(issue + HOUR).toISOString())} — ${shownTime(new Date(issue + request.horizon * HOUR).toISOString())} UTC`;
}
const settingsDefault: TimeSettings = {timezone: '', semantics: ''};
const phaseNames: Record<BatchResult['phase'], string> = {
  queued: 'В очереди', checking: 'Проверяем данные', ready: 'Данные готовы', blocked: 'Расчёт заблокирован',
  running: 'Агент работает', done: 'Запуск завершён', error: 'Ошибка', cancelled: 'Не запущено: очередь остановлена',
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === 'string' ? body.detail : `Ошибка запроса (${response.status}). Проверьте параметры.`;
    throw new Error(detail);
  }
  return response.json();
}

function post<T>(path: string, request: AgentRequest) {
  return api<T>(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(request)});
}

function parseHour(value: string): number | null {
  if (!/^\d{4}-\d{2}-\d{2}T\d{2}:00$/.test(value)) return null;
  const parsed = Date.parse(`${value}:00Z`);
  return Number.isFinite(parsed) && formatInput(parsed) === value ? parsed : null;
}

function checkText(check: ToolCheck): string {
  const result = check.result;
  const messages: Record<string, string> = {
    no_measurements: 'Нет импортированных измерений.',
    time_unconfirmed: 'Укажите часовой пояс и смысл временной отметки CSV.',
    no_complete_history: 'До выбранного момента расчёта нет полного часа измерений.',
    stale_telemetry: `Измерения устарели${typeof result.age_hours === 'number' ? ` на ${result.age_hours} ч` : ''}. Допустимый возраст — 2 часа. Выберите период рядом с импортированными данными.`,
    validation_failed: 'Не удалось подготовить измерения. Проверьте выбранные настройки времени.',
  };
  if (result.ok && result.time_configured === false) return messages.time_unconfirmed;
  if (result.ok) {
    if (check.tool === 'prepare_features' && typeof result.available_at === 'string') return `Полное измерение доступно с ${shownTime(result.available_at)} UTC; возраст — ${result.age_hours} ч.`;
    return 'Проверка пройдена.';
  }
  return messages[String(result.code)] || (typeof result.detail === 'string' ? result.detail : String(result.code || 'Проверка не пройдена.'));
}

function Checks({checks}: {checks: ToolCheck[]}) {
  return <ul className="forecast-checks">{checks.map((check, index) => <li key={`${check.tool}-${index}`}>
    <strong>{check.tool === 'inspect_data' ? 'Измерения' : check.tool === 'prepare_features' ? 'Готовность к расчёту' : check.tool}:</strong> {checkText(check)}
  </li>)}</ul>;
}

function Report({report}: {report: AgentReport}) {
  const saved = report.steps.some(step => step.tool === 'save_forecast' && step.result.ok === true);
  return <div>
    <p><strong>{saved ? 'Прогноз сохранён' : 'Прогноз не сохранён'}</strong> · {report.status}</p>
    <p style={{whiteSpace: 'pre-wrap'}}>{report.summary || report.error}</p>
    {report.error && report.summary && <p className="error">{report.error}</p>}
    <p className="small muted">Токены: {report.input_tokens.toLocaleString('ru-RU')} вход / {report.output_tokens.toLocaleString('ru-RU')} выход</p>
    {saved && <p><a href={`/api/forecasts/${report.id}`} download={`forecast-${report.id}.json`}>Скачать прогноз JSON ↓</a></p>}
    <details><summary>Журнал действий агента</summary>{report.steps.map((step, index) => <details key={`${step.step}-${index}`}>
      <summary>{step.step}. {step.tool} — {step.result.ok ? 'выполнено' : String(step.result.code || 'ошибка')}</summary>
      <pre style={{whiteSpace: 'pre-wrap', overflowWrap: 'anywhere'}}>{JSON.stringify(step.result, null, 2)}</pre>
    </details>)}</details>
  </div>;
}

function ModelInfo({turbineId, settings, issue, models, loading, error}: {
  turbineId: number; settings: TimeSettings; issue: number | null;
  models: MLModel[]; loading: boolean; error: string;
}) {
  if (loading) return <p className="small muted" role="status">Проверяем доступность ML…</p>;
  if (error) return <p className="small muted">Не удалось проверить список ML-моделей. При запуске сервер проверит доступность модели заново.</p>;
  if (!settings.timezone || !settings.semantics) return <p className="small muted">Укажите часовой пояс и смысл отметки CSV, чтобы проверить подходящую модель. До этого запуск заблокирован.</p>;

  const candidates = models.filter(model => model.turbine_id === turbineId && model.model_type === 'telemetry_hist_gradient_boosting');
  const matching = candidates.filter(model => model.timezone === settings.timezone && model.timestamp_semantics === settings.semantics);
  const current = matching.filter(model => model.source_matches);
  const promoted = current.filter(model => model.promoted);
  const available = promoted.filter(model => issue !== null && Number.isFinite(Date.parse(model.usable_from)) && issue >= Date.parse(model.usable_from));
  const byNewest = (items: MLModel[]) => [...items].sort((a, b) => Date.parse(b.usable_from) - Date.parse(a.usable_from))[0];
  const model = byNewest(available) || byNewest(promoted) || byNewest(current) || byNewest(matching);
  const mlAvailable = available.length > 0;
  let reason = '';
  let title = 'Резервный baseline';
  if (!candidates.length) reason = 'Для этой турбины ещё нет обученной ML-модели.';
  else if (!matching.length) reason = 'Нет ML-модели с выбранными настройками времени CSV.';
  else if (!current.length) reason = 'После обучения измерения изменились. Для нового CSV модель нужно переобучить.';
  else if (!promoted.length) reason = 'Модель не прошла отбор по результатам валидации.';
  else if (issue === null) {title = 'ML готова к проверке даты'; reason = 'Выберите начало прогноза, чтобы проверить доступность модели на момент расчёта.';}
  else if (mlAvailable) {title = 'ML доступна'; reason = 'Бустинг использует прошлые измерения и календарные признаки. Агент проверит данные перед расчётом.';}
  else if (model && Number.isFinite(Date.parse(model.usable_from))) reason = `На выбранный момент расчёта ML ещё не была доступна. Её можно использовать с ${shownTime(model.usable_from)} UTC.`;
  else reason = 'В метаданных модели не указан допустимый момент начала использования.';

  return <div className="forecast-model-info">
    <p className="muted"><strong>{title}.</strong> {reason}{title === 'Резервный baseline' && ' Будет повторяться последняя известная мощность.'}</p>
    {model && <details><summary>Как проверена модель</summary>
      {model.training_end && <p>Обучение: измерения раньше {shownTime(model.training_end)} UTC.</p>}
      {model.validation_start && model.validation_end && <p>Валидация: {shownTime(model.validation_start)} — {shownTime(model.validation_end)} UTC.</p>}
      {typeof model.validation_mae === 'number' && Number.isFinite(model.validation_mae) && <p>MAE на валидации: ML {model.validation_mae.toFixed(3)}{typeof model.persistence_validation_mae === 'number' && Number.isFinite(model.persistence_validation_mae) && <> · baseline {model.persistence_validation_mae.toFixed(3)}</>}. Доля номинальной мощности; меньше — лучше.</p>}
      {model.control_start && model.control_end && <p>Контроль: {shownTime(model.control_start)} — {shownTime(model.control_end)} UTC. Концы периодов не включены.</p>}
      <p>Контроль — ретроспективная проверка, а не официальный нетронутый тест. Гипотеза часового пояса выбиралась по всей доступной истории. Показанные ошибки относятся к валидации и не гарантируют точность нового прогноза.</p>
    </details>}
  </div>;
}

export function ForecastPage({turbines}: {turbines: Turbine[]}) {
  const [selected, setSelected] = useState<number[]>([]);
  const [settings, setSettings] = useState<Record<number, TimeSettings>>({});
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [statusError, setStatusError] = useState('');
  const [formError, setFormError] = useState('');
  const [busy, setBusy] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [results, setResults] = useState<BatchResult[]>([]);
  const [history, setHistory] = useState<Record<number, AgentReport[]>>({});
  const [historyError, setHistoryError] = useState('');
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyVersion, setHistoryVersion] = useState(0);
  const [models, setModels] = useState<MLModel[]>([]);
  const [modelsLoading, setModelsLoading] = useState(true);
  const [modelsError, setModelsError] = useState('');
  const inFlight = useRef(false);
  const stopRequested = useRef(false);
  const selectedTurbines = turbines.filter(t => selected.includes(t.id));
  const historyIds = selectedTurbines.map(t => t.id).join(',');
  const firstHour = parseHour(start);
  const lastHour = parseHour(end);
  const horizon = firstHour !== null && lastHour !== null ? (lastHour - firstHour) / HOUR + 1 : null;
  const issue = firstHour === null ? null : firstHour - HOUR;
  const datesValid = horizon !== null && horizon >= 24 && horizon <= 48 && Number.isInteger(horizon) && issue !== null && issue <= Date.now();

  useEffect(() => {
    const controller = new AbortController();
    api<{configured: boolean}>('/api/agent/status', {signal: controller.signal})
      .then(value => setConfigured(value.configured))
      .catch(error => {if (!controller.signal.aborted) setStatusError((error as Error).message);});
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setModelsLoading(true);
    setModelsError('');
    api<{items: MLModel[]}>('/api/ml/models', {signal: controller.signal})
      .then(value => {if (!controller.signal.aborted) setModels(value.items);})
      .catch(error => {if (!controller.signal.aborted) {setModels([]); setModelsError((error as Error).message);}})
      .finally(() => {if (!controller.signal.aborted) setModelsLoading(false);});
    return () => controller.abort();
  }, [historyVersion]);

  useEffect(() => {
    const controller = new AbortController();
    const ids = historyIds ? historyIds.split(',').map(Number) : [];
    setHistoryError('');
    setHistoryLoading(ids.length > 0);
    if (!ids.length) {setHistory({}); return () => controller.abort();}
    void Promise.all(ids.map(async id => {
      const data = await api<{items: AgentReport[]}>(`/api/agent/runs?turbine_id=${id}`, {signal: controller.signal});
      return [id, data.items] as const;
    })).then(items => {
      if (!controller.signal.aborted) setHistory(Object.fromEntries(items));
    }).catch(error => {
      if (!controller.signal.aborted) {setHistory({}); setHistoryError((error as Error).message);}
    }).finally(() => {if (!controller.signal.aborted) setHistoryLoading(false);});
    return () => controller.abort();
  }, [historyIds, historyVersion]);

  useEffect(() => {
    if (!busy) return;
    const warn = (event: BeforeUnloadEvent) => {event.preventDefault(); event.returnValue = '';};
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [busy]);

  function updateSettings(id: number, value: Partial<TimeSettings>) {
    setSettings(current => ({...current, [id]: {...settingsDefault, ...current[id], ...value}}));
  }

  function updateResult(id: number, update: Partial<BatchResult>) {
    setResults(current => current.map(result => result.turbine.id === id ? {...result, ...update} : result));
  }

  async function run(generate: boolean) {
    if (inFlight.current) return;
    setFormError('');
    if (!selectedTurbines.length) {setFormError('Выберите хотя бы одну турбину.'); return;}
    if (!datesValid || issue === null || horizon === null) {setFormError('Укажите от 24 до 48 почасовых отметок включительно. Момент расчёта за час до начала не может быть в будущем.'); return;}
    if (generate && configured !== true) {setFormError('Для запуска агента нужен API-ключ на сервере. Проверка данных доступна без ключа.'); return;}
    const jobs = selectedTurbines.map(turbine => ({turbine, request: {
      turbine_id: turbine.id, issue_at: new Date(issue).toISOString(), horizon,
      measurement_timezone: settings[turbine.id]?.timezone || null,
      timestamp_semantics: settings[turbine.id]?.semantics || null, event: 'manual' as const,
    }}));
    inFlight.current = true;
    stopRequested.current = false;
    setStopping(false);
    setBusy(true);
    setResults(jobs.map(({turbine, request}) => ({turbine, request, phase: 'queued'})));
    try {
      for (const job of jobs) {
        const id = job.turbine.id;
        if (stopRequested.current) {updateResult(id, {phase: 'cancelled'}); continue;}
        updateResult(id, {phase: 'checking'});
        try {
          const preflight = await post<Preflight>('/api/agent/preflight', job.request);
          updateResult(id, {preflight, phase: preflight.ok ? 'ready' : 'blocked'});
          if (!preflight.ok || !generate) continue;
          if (stopRequested.current) {updateResult(id, {phase: 'cancelled'}); continue;}
          updateResult(id, {phase: 'running'});
          const report = await post<AgentReport>('/api/agent/runs', job.request);
          updateResult(id, {report, phase: 'done'});
          setHistoryVersion(value => value + 1);
        } catch (error) {updateResult(id, {phase: 'error', error: (error as Error).message});}
      }
    } finally {
      inFlight.current = false;
      setBusy(false);
      setStopping(false);
      setHistoryVersion(value => value + 1);
    }
  }

  const currentReportIds = new Set(results.flatMap(result => result.report ? [result.report.id] : []));
  return <>
    <section className="panel forecast-panel">
      <div className="section-heading"><div><div className="eyebrow">Параметры расчёта</div><h2>Получить прогноз</h2></div><span className="tag">{configured === null ? 'Проверяем подключение…' : configured ? 'OpenAI подключён' : 'Ключ не настроен'}</span></div>
      <p className="muted">Выберите турбины и общий период прогноза. Для каждой турбины агент проверит измерения и сохранит отдельный результат.</p>
      {!turbines.length ? <p className="muted">Сначала <a href="/turbines">добавьте турбину и импортируйте измерения</a>.</p> : <form onSubmit={event => {event.preventDefault(); void run(true);}}>
        <fieldset disabled={busy}>
          <legend>Турбины</legend>
          <div className="forecast-turbines">{turbines.map(turbine => {
            const checked = selected.includes(turbine.id);
            const timeSettings = settings[turbine.id] || settingsDefault;
            return <div className={`forecast-turbine${checked ? ' selected' : ''}`} key={turbine.id}>
              <label className="source-check"><input type="checkbox" checked={checked} disabled={!turbine.has_data} onChange={event => setSelected(current => event.target.checked ? [...current, turbine.id] : current.filter(id => id !== turbine.id))}/><strong>{turbine.name}</strong></label>
              <p className="small muted">{turbine.has_data ? `Измерения: ${shownTime(turbine.start || '')} — ${shownTime(turbine.end || '')} · время CSV` : 'Нет измерений — импортируйте CSV на странице «Турбины».'}</p>
              {checked && <><div className="filters">
                <label>Часовой пояс CSV<select value={timeSettings.timezone} onChange={event => updateSettings(turbine.id, {timezone: event.target.value})}>
                  <option value="">Не подтверждён</option><option value="Etc/GMT-6">UTC+6 · постоянное смещение</option><option value="Asia/Almaty">Asia/Almaty · с переводом часов</option><option value="UTC">UTC</option>
                </select></label>
                <label>Отметка времени CSV<select value={timeSettings.semantics} onChange={event => updateSettings(turbine.id, {semantics: event.target.value})}>
                  <option value="">Не подтверждена</option><option value="interval_start">Начало 10-минутного интервала</option><option value="interval_end">Конец 10-минутного интервала</option>
                </select></label>
              </div><ModelInfo turbineId={turbine.id} settings={timeSettings} issue={issue} models={models} loading={modelsLoading} error={modelsError}/></>}
            </div>;
          })}</div>
        </fieldset>
        <fieldset disabled={busy}>
          <legend>Период прогноза · UTC</legend>
          <div className="filters forecast-dates">
            <label>Первая отметка прогноза<input type="datetime-local" step="3600" required value={start} max={formatInput(Math.floor(Date.now() / HOUR) * HOUR + HOUR)} onChange={event => setStart(event.target.value)}/></label>
            <label>Последняя отметка, включительно<input type="datetime-local" step="3600" required value={end} min={firstHour === null ? undefined : formatInput(firstHour + 23 * HOUR)} max={firstHour === null ? undefined : formatInput(firstHour + 47 * HOUR)} onChange={event => setEnd(event.target.value)}/></label>
            <button type="button" disabled={firstHour === null} onClick={() => {if (firstHour !== null) setEnd(formatInput(firstHour + 23 * HOUR));}}>24 часа</button>
            <button type="button" disabled={firstHour === null} onClick={() => {if (firstHour !== null) setEnd(formatInput(firstHour + 47 * HOUR));}}>48 часов</button>
          </div>
          <p className="muted">Все даты здесь — UTC. Диапазон включает обе крайние отметки: например, с 1 февраля 01:00 до 2 февраля 00:00 — это 24 значения.</p>
          {issue !== null && <p className="forecast-cutoff"><strong>Момент расчёта и отсечения данных: {shownTime(new Date(issue).toISOString())} UTC.</strong> Используются только измерения, доступные к этому моменту.{horizon !== null && horizon > 0 && <> Отметок прогноза: {horizon}.</>}</p>}
          {issue !== null && issue > Date.now() && <p className="error" role="alert">Момент расчёта ещё не наступил. Начните прогноз не позднее следующего целого часа UTC.</p>}
          {horizon !== null && (horizon < 24 || horizon > 48) && <p className="error" role="alert">Выберите от 24 до 48 почасовых отметок включительно.</p>}
        </fieldset>
        <div className="notice compact">Агент автоматически использует прошедшую отбор ML-модель, если совпадают измерения и настройки времени, а модель уже доступна на момент расчёта. Иначе используется резервный baseline — повтор последней известной мощности. ML работает по прошлой телеметрии и календарю, без будущей погоды. Часовой пояс CSV выбираете вы; UTC+6 остаётся гипотезой.</div>
        <div className="filters forecast-actions">
          <button type="button" disabled={busy || !selectedTurbines.length || !datesValid} onClick={() => void run(false)}>Проверить данные бесплатно</button>
          <button type="submit" disabled={busy || configured !== true || !selectedTurbines.length || !datesValid}>{busy ? 'Обрабатываем очередь…' : `Получить прогноз${selectedTurbines.length > 1 ? 'ы' : ''}`}</button>
          {busy && <button type="button" disabled={stopping} onClick={() => {stopRequested.current = true; setStopping(true);}}> {stopping ? 'Останавливаем после текущего запроса…' : 'Остановить очередь'}</button>}
        </div>
        <p className="small muted">Проверка данных не вызывает LLM и не расходует API-кредиты. «Получить прогноз» запускает платного агента отдельно для каждой готовой турбины. Проверка повторяется перед каждым запуском. Очередь выполняется последовательно.</p>
        {busy && <p className="notice compact" role="status">Дождитесь завершения на этой странице. Остановка очереди не отменяет уже начатый запрос; следующие турбины запускаться не будут.</p>}
      </form>}
      {statusError && <div className="error" role="alert">{statusError}</div>}
      {formError && <div className="error" role="alert">{formError}</div>}
    </section>
    {results.length > 0 && <section className="panel forecast-results" aria-live="polite">
      <h2>Результаты текущей очереди</h2>
      {results.map(result => <article className="forecast-result" key={result.turbine.id}>
        <div className="section-heading"><h3>{result.turbine.name}</h3><span className="tag">{phaseNames[result.phase]}</span></div>
        <p><strong>Период: {requestPeriod(result.request)}</strong></p>
        <p className="small muted">Расчёт на {shownTime(result.request.issue_at)} UTC · горизонт {result.request.horizon} ч · CSV: {result.request.measurement_timezone || 'пояс не задан'} · {result.request.timestamp_semantics === 'interval_start' ? 'начало интервала' : result.request.timestamp_semantics === 'interval_end' ? 'конец интервала' : 'смысл отметки не задан'}</p>
        {result.preflight && <><Checks checks={result.preflight.checks}/>{!result.preflight.ok && <p className="muted">Агент не запускался, API-кредиты на этот расчёт не использованы.</p>}</>}
        {result.error && <div className="error" role="alert">{result.error}</div>}
        {result.report && <Report report={result.report}/>}
      </article>)}
    </section>}
    <section className="panel forecast-history"><div className="section-heading"><h2>История выбранных турбин</h2><button className="source-secondary" type="button" disabled={busy || historyLoading || !selectedTurbines.length} onClick={() => setHistoryVersion(value => value + 1)}>Обновить</button></div>
      <p className="muted">Последние сохранённые запуски, до 20 для каждой турбины. Просмотр и скачивание не запускают агента повторно.</p>
      {!selectedTurbines.length && <p className="muted">Выберите турбины выше, чтобы увидеть их историю.</p>}
      {historyLoading && <p className="muted" role="status">Загружаем историю…</p>}
      {historyError && <div className="error" role="alert">{historyError}</div>}
      {!historyLoading && !historyError && selectedTurbines.map(turbine => {
        const reports = (history[turbine.id] || []).filter(report => !currentReportIds.has(report.id));
        return <div key={turbine.id}><h3>{turbine.name}</h3>{!reports.length && <p className="muted">Предыдущих запусков нет.</p>}{reports.map(report => {
          return <details key={report.id}><summary>{requestPeriod(report.request)} · {report.status}</summary>
            <p className="small muted">Момент расчёта: {shownTime(report.request.issue_at)} UTC · CSV: {report.request.measurement_timezone || 'пояс не задан'} · {report.request.timestamp_semantics === 'interval_start' ? 'начало интервала' : report.request.timestamp_semantics === 'interval_end' ? 'конец интервала' : 'смысл отметки не задан'}</p>
            <Report report={report}/>
          </details>;
        })}</div>;
      })}
    </section>
  </>;
}
