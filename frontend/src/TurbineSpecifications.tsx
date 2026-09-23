import {useRef, useState} from 'react';

type Specifications = {
  rated_power_kw?: number | null; manufacturer?: string | null; turbine_model?: string | null;
  capacity_source_url?: string | null; capacity_status?: string | null;
};
type Turbine = Specifications & {id: number; name: string; latitude: number; longitude: number};
type Candidate = {
  osm_node_id: number; latitude: number; longitude: number; distance_m: number;
  rated_power_kw?: number | null; manufacturer?: string | null; model?: string | null; source_url: string;
};
type SearchResult = {current: Specifications; candidates: Candidate[]; confirmation_required: boolean; warning?: string | null};
const number = (value: number) => value.toLocaleString('ru-RU', {maximumFractionDigits: 1});

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Не удалось получить характеристики (${response.status}).`);
  }
  return response.json();
}

function SourceLink({url}: {url?: string | null}) {
  if (!url || !/^https?:\/\//i.test(url)) return null;
  return <a href={url} target="_blank" rel="noreferrer">Источник ↗</a>;
}

export function TurbineSpecifications({turbine, disabled = false, onBusyChange, onApplied}: {
  turbine: Turbine; disabled?: boolean; onBusyChange?: (busy: boolean) => void;
  onApplied: () => Promise<void>;
}) {
  const [current, setCurrent] = useState<Specifications>(turbine);
  const [candidates, setCandidates] = useState<Candidate[] | null>(null);
  const [busy, setBusy] = useState<'search' | number | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [warning, setWarning] = useState('');
  const pending = useRef(false);
  const mapBounds = [Math.max(-180, turbine.longitude - .009), Math.max(-90, turbine.latitude - .006), Math.min(180, turbine.longitude + .009), Math.min(90, turbine.latitude + .006)].join(',');
  const mapParams = new URLSearchParams({bbox: mapBounds, layer: 'mapnik', marker: `${turbine.latitude},${turbine.longitude}`});
  const mapLink = `https://www.openstreetmap.org/?mlat=${turbine.latitude}&mlon=${turbine.longitude}#map=16/${turbine.latitude}/${turbine.longitude}`;

  async function search() {
    if (pending.current || disabled) return;
    pending.current = true; setBusy('search'); onBusyChange?.(true); setError(''); setNotice('');
    try {
      const response = await api<SearchResult>(`/api/turbines/${turbine.id}/specifications`);
      setCurrent(response.current); setCandidates(response.candidates); setWarning(response.warning || '');
    } catch (cause) {setError((cause as Error).message);}
    finally {pending.current = false; setBusy(null); onBusyChange?.(false);}
  }

  async function apply(candidate: Candidate) {
    if (pending.current || disabled) return;
    pending.current = true; setBusy(candidate.osm_node_id); onBusyChange?.(true); setError(''); setNotice('');
    try {
      const response = await api<Specifications | {current: Specifications}>(`/api/turbines/${turbine.id}/specifications`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({osm_node_id: candidate.osm_node_id}),
      });
      setCurrent('current' in response ? response.current : response);
      setNotice('Характеристики сохранены с вашим подтверждением и ссылкой на OpenStreetMap.');
      await onApplied();
    } catch (cause) {setError((cause as Error).message);}
    finally {pending.current = false; setBusy(null); onBusyChange?.(false);}
  }

  return <section className="panel turbine-specifications">
    <div className="section-heading"><div><div className="eyebrow">Местоположение и оборудование</div><h2>Характеристики · {turbine.name}</h2></div><span className="tag">{current.capacity_status === 'user_confirmed_osm' ? 'OSM подтверждён пользователем' : current.rated_power_kw ? 'Номинал указан' : 'Номинал не задан'}</span></div>
    <div className="turbine-spec-layout">
      <div className="turbine-map">
        <iframe title={`Местоположение турбины ${turbine.name} на OpenStreetMap`} src={`https://www.openstreetmap.org/export/embed.html?${mapParams}`} loading="lazy" referrerPolicy="no-referrer"/>
        <p className="small muted">{turbine.latitude}°, {turbine.longitude}° · <a href={mapLink} target="_blank" rel="noreferrer">Открыть карту ↗</a></p>
      </div>
      <div>
        <dl className="turbine-spec-values"><div><dt>Производитель</dt><dd>{current.manufacturer || 'Не указан'}</dd></div><div><dt>Модель</dt><dd>{current.turbine_model || 'Не указана'}</dd></div><div><dt>Номинальная мощность</dt><dd>{current.rated_power_kw ? `${number(current.rated_power_kw)} кВт` : 'Не указана'}</dd></div></dl>
        <SourceLink url={current.capacity_source_url}/>
        <p className="small muted">Пересчёт в кВт предполагает, что исходная мощность нормирована на номинал.</p>
        <p className="muted">Найдём ближайшие ветрогенераторы в OpenStreetMap. Сопоставьте объект и характеристики с вашей турбиной перед применением.</p>
        <button className="source-secondary" type="button" disabled={disabled || busy !== null} onClick={() => void search()}>{busy === 'search' ? 'Ищем в OpenStreetMap…' : 'Найти характеристики'}</button>
      </div>
    </div>
    <p className="notice compact">OpenStreetMap — общедоступная карта, а не паспорт оборудования. Близость координат помогает найти кандидата, но не подтверждает, что это ваша турбина. Сохранение характеристик выполняется только по кнопке «Применить».</p>
    {candidates !== null && !candidates.length && <p className="muted">Подходящих объектов рядом не найдено. Координаты и характеристики турбины не изменены.</p>}
    {warning && <p className="notice compact">{warning}</p>}
    {!!candidates?.length && <div className="turbine-spec-candidates">{candidates.map(candidate => <article key={candidate.osm_node_id}>
      <h3>{[candidate.manufacturer, candidate.model].filter(Boolean).join(' · ') || 'Ветрогенератор OpenStreetMap'}</h3>
      <p className="muted">{candidate.rated_power_kw ? `${number(candidate.rated_power_kw)} кВт` : 'Мощность не указана'} · {number(candidate.distance_m)} м от заданных координат</p>
      <p className="small muted">{candidate.latitude}°, {candidate.longitude}° · <SourceLink url={candidate.source_url}/></p>
      <button className="source-primary" type="button" disabled={disabled || busy !== null || !candidate.rated_power_kw} onClick={() => void apply(candidate)}>{busy === candidate.osm_node_id ? 'Сохраняем…' : 'Применить к этой турбине'}</button>
    </article>)}</div>}
    {error && <p className="error" role="alert">{error}</p>}
    {notice && <p className="muted" role="status">{notice}</p>}
  </section>;
}
