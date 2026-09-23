import {useEffect, useState} from 'react';
import {DiscoveryPanel} from './DiscoveryPanel';
type Mapping = {rows_path:string; time_field:string; wind_field:string; temperature_field:string; timezone:string; wind_unit:string; temperature_unit:string; wind_height_m:number|null; timestamp_semantics:string};
type Source = {id?:string; revision?:number; name:string; url:string; format:string; delimiter:string; trusted:boolean; enabled:boolean; notes:string; mapping:Mapping};
type Result = {rows:number; preview:Record<string,unknown>[]; batch_id?:string};
type Proposal = {mapping:Mapping; explanation:string; unresolved:string[]; validation:{ok:boolean; error?:string; rows?:number}; usage:{input_tokens:number;output_tokens:number}};
const empty = ():Source => ({name:'',url:'',format:'json',delimiter:',',trusted:false,enabled:true,notes:'',mapping:{rows_path:'',time_field:'',wind_field:'',temperature_field:'',timezone:'',wind_unit:'',temperature_unit:'',wind_height_m:null,timestamp_semantics:''}});
async function request<T>(path:string, method='GET', body?:unknown):Promise<T> {
  const response=await fetch('/api/sources'+path,{method,headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined});
  const value=await response.json();
  if(!response.ok) throw new Error(typeof value.detail==='string'?value.detail:JSON.stringify(value.detail));
  return value;
}
export function SourcesPanel({turbine}:{turbine?:{id:number;name:string;has_data:boolean;start:string;end:string;latitude:number;longitude:number}}) {
  const [items,setItems]=useState<Source[]>([]),[draft,setDraft]=useState<Source>(empty),[busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState('');
  const [proposal,setProposal]=useState<Proposal|null>(null),[result,setResult]=useState<Result|null>(null),[dirty,setDirty]=useState(false);
  const refresh=async()=>setItems((await request<{items:Source[]}>('')).items);
  useEffect(()=>{void refresh().catch(e=>setError(e.message));},[]);
  const change=(values:Partial<Source>)=>{setDraft(d=>({...d,...values}));setDirty(true);setResult(null);setProposal(null);setMessage('');};
  const field=(key:keyof Mapping,value:string|number|null)=>change({mapping:{...draft.mapping,[key]:value}});
  const task=async(fn:()=>Promise<void>)=>{setBusy(true);setError('');setMessage('');setResult(null);try{await fn();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  const choose=(source:Source)=>{setDraft(source);setDirty(false);setProposal(null);setResult(null);setError('');setMessage('');};
  const body=()=>{const {id,revision,...values}=draft;void id;void revision;return values;};
  return <section id="sources" className="panel sources-panel">
    <div className="section-heading"><div><div className="eyebrow">Подключение данных</div><h2>Доверенные источники</h2></div><span className="tag">JSON / CSV</span></div>
    <DiscoveryPanel key={turbine?.id??"empty"} turbine={turbine} onLoaded={refresh}/>
    <h3>Ручная настройка подключения</h3>
    <p className="muted">Добавьте публичный адрес данных. Агент предложит соответствие полей по образцу и выдержке из документации. Неизвестные параметры задайте вручную.</p>
    <p className="small muted">Архив ECMWF выше использует отдельный встроенный адаптер. Эти подключения сохраняют дополнительные данные; baseline-прогноз их пока не использует.</p>
    <div className="tabs">{items.map(s=><button disabled={busy} key={s.id} onClick={()=>choose(s)} className={draft.id===s.id?'active':''}>{s.name}{!s.enabled?' · выключен':''}</button>)}<button disabled={busy} onClick={()=>choose(empty())}>+ Новый источник</button><button disabled={busy} onClick={()=>choose({...empty(),name:'Open-Meteo · пример за 25.01.2026',url:'https://archive-api.open-meteo.com/v1/archive?latitude=43.64515&longitude=78.535604&start_date=2026-01-25&end_date=2026-01-25&hourly=wind_speed_100m,temperature_2m&models=era5&timezone=GMT&wind_speed_unit=ms',notes:'Open-Meteo Historical Weather API: GMT = UTC; wind_speed_100m — скорость ветра на высоте 100 м в м/с; temperature_2m — температура в °C. Значения мгновенные, время ISO 8601. Документация: https://open-meteo.com/en/docs/historical-weather-api'})}>Заполнить пример Open-Meteo</button></div>
    <form onSubmit={e=>{e.preventDefault();void task(async()=>{const saved=await request<Source>(draft.id?'/'+draft.id:'',draft.id?'PUT':'POST',body());setDraft(saved);setDirty(false);setResult(null);await refresh();setMessage('Настройки сохранены. Проверьте загрузку.');});}}>
      <fieldset disabled={busy}>
        <div className="source-grid">
          <label>Название<input required value={draft.name} onChange={e=>change({name:e.target.value})}/></label>
          <label>Формат<select value={draft.format} onChange={e=>change({format:e.target.value})}><option value="json">JSON</option><option value="csv">CSV UTF-8</option></select></label>
          <label className="source-wide">URL с параметрами запроса (без ключей и паролей)<input type="url" required placeholder="https://example.org/weather.json" value={draft.url} onChange={e=>change({url:e.target.value})}/></label>
          {draft.format==='csv'?<label>Разделитель<select value={draft.delimiter} onChange={e=>change({delimiter:e.target.value})}><option value=",">Запятая</option><option value=";">Точка с запятой</option><option value={'\t'}>Табуляция</option></select></label>:<label>Путь к данным JSON<input placeholder="hourly или data.rows; пусто — корень" value={draft.mapping.rows_path} onChange={e=>field('rows_path',e.target.value)}/></label>}
          <label>Поле времени<input value={draft.mapping.time_field} onChange={e=>field('time_field',e.target.value)}/></label>
          <label>Поле скорости ветра<input value={draft.mapping.wind_field} onChange={e=>field('wind_field',e.target.value)}/></label>
          <label>Единицы ветра<select value={draft.mapping.wind_unit} onChange={e=>field('wind_unit',e.target.value)}><option value="">Неизвестны</option><option>m/s</option><option>km/h</option><option>knots</option></select></label>
          <label>Высота ветра, м<input type="number" min="0.1" max="1000" step="any" value={draft.mapping.wind_height_m??''} onChange={e=>field('wind_height_m',e.target.value?Number(e.target.value):null)}/></label>
          <label>Часовой пояс IANA<input placeholder="UTC, Asia/Almaty, Etc/GMT-6" value={draft.mapping.timezone} onChange={e=>field('timezone',e.target.value)}/></label>
          <label>Смысл временной отметки<select value={draft.mapping.timestamp_semantics} onChange={e=>field('timestamp_semantics',e.target.value)}><option value="">Неизвестен</option><option value="instant">Момент измерения</option><option value="interval_start">Начало интервала</option><option value="interval_end">Конец интервала</option></select></label>
          <label>Поле температуры (необязательно)<input value={draft.mapping.temperature_field} onChange={e=>field('temperature_field',e.target.value)}/></label>
          <label>Единицы температуры<select value={draft.mapping.temperature_unit} onChange={e=>field('temperature_unit',e.target.value)}><option value="">Неизвестны</option><option value="C">°C</option><option value="K">K</option><option value="F">°F</option></select></label>
          <label className="source-wide">Выдержка из документации: единицы, высота, время<textarea maxLength={4000} rows={3} value={draft.notes} onChange={e=>change({notes:e.target.value})}/></label>
        </div>
        <label className="source-check"><input type="checkbox" checked={draft.trusted} onChange={e=>change({trusted:e.target.checked})}/> Доверяю этому адресу и разрешаю загрузку</label>
        <label className="source-check"><input type="checkbox" checked={draft.enabled} onChange={e=>change({enabled:e.target.checked})}/> Источник включён</label>
        <div className="filters"><button type="submit">Сохранить настройки</button><button type="button" disabled={!draft.trusted||!draft.enabled||!draft.url||!draft.name} onClick={()=>void task(async()=>{setProposal(await request<Proposal>('/suggest','POST',body()));})}>Предложить настройки агентом</button><button type="button" disabled={!draft.id||dirty||!draft.trusted||!draft.enabled} onClick={()=>void task(async()=>{setResult(await request<Result>('/'+draft.id+'/preview','POST'));})}>Проверить загрузку</button><button type="button" disabled={!draft.id||dirty||!draft.trusted||!draft.enabled} onClick={()=>void task(async()=>{setResult(await request<Result>('/'+draft.id+'/load','POST'));setMessage('Данные проверены и сохранены.');})}>Загрузить и сохранить</button></div>
      </fieldset>
    </form>
    <p className="muted small">Агент получает образец ответа и текст документации. Только эта кнопка использует API-кредиты. Предложение не сохраняется автоматически. Лимит ответа источника — 2 МБ. Постоянный UTC+6 обозначается Etc/GMT-6.</p>
    {busy&&<p role="status">Выполняется запрос…</p>}{error&&<div className="error" role="alert">{error}</div>}{message&&<p role="status">{message}</p>}
    {proposal&&<div className="notice"><strong>Предложение агента</strong><p>{proposal.explanation}</p>{proposal.unresolved.length>0&&<p>Уточнить: {proposal.unresolved.join('; ')}</p>}<p>Проверка образца: {proposal.validation.ok?`${proposal.validation.rows} строк`:proposal.validation.error}</p><pre>{JSON.stringify(proposal.mapping,null,2)}</pre><p>Токены: {proposal.usage.input_tokens} / {proposal.usage.output_tokens}</p><button disabled={busy} onClick={()=>{setDraft(d=>({...d,mapping:proposal.mapping}));setDirty(true);setProposal(null);setResult(null);setMessage('Предложение перенесено в форму. Проверьте поля и сохраните настройки.');}}>Перенести в форму</button></div>}
    {result&&<div><p>Проверено строк: {result.rows}. Время UTC, ветер м/с, температура °C.</p><pre>{JSON.stringify(result.preview,null,2)}</pre>{result.batch_id&&<a href={'/api/sources/batches/'+result.batch_id} target="_blank" rel="noreferrer">Открыть сохранённые данные JSON ↗</a>}<p className="muted small">Время публикации не подтверждено: историческая доступность для прогнозирования не установлена.</p></div>}
  </section>;
}
