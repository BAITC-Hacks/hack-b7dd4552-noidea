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
export function SourcesPanel({turbine,onBusyChange}:{onBusyChange?:(busy:boolean)=>void;turbine?:{id:number;name:string;has_data:boolean;start:string;end:string;latitude:number;longitude:number}}) {
  const [discoveryBusy,setDiscoveryBusy]=useState(false);
  const [view,setView]=useState<'list'|'new'|'builtin'|'edit'>('list');
  const [items,setItems]=useState<Source[]>([]),[draft,setDraft]=useState<Source>(empty),[busy,setBusy]=useState(false),[error,setError]=useState(''),[message,setMessage]=useState('');
  const [proposal,setProposal]=useState<Proposal|null>(null),[result,setResult]=useState<Result|null>(null),[dirty,setDirty]=useState(false);
  const active=busy||discoveryBusy;
  useEffect(()=>{
    onBusyChange?.(active);
    if(!active)return;
    const warn=(event:BeforeUnloadEvent)=>{event.preventDefault();event.returnValue='';};
    window.addEventListener('beforeunload',warn);
    return()=>window.removeEventListener('beforeunload',warn);
  },[active,onBusyChange]);
  const refresh=async()=>setItems((await request<{items:Source[]}>('')).items);
  useEffect(()=>{void refresh().catch(e=>setError(e.message));},[]);
  const change=(values:Partial<Source>)=>{setDraft(d=>({...d,...values}));setDirty(true);setResult(null);setProposal(null);setMessage('');};
  const field=(key:keyof Mapping,value:string|number|null)=>change({mapping:{...draft.mapping,[key]:value}});
  const task=async(fn:()=>Promise<void>)=>{setBusy(true);setError('');setMessage('');setResult(null);try{await fn();}catch(e){setError((e as Error).message);}finally{setBusy(false);}};
  const choose=(source:Source)=>{setView(source.id?'edit':'new');setDraft(source);setDirty(false);setProposal(null);setResult(null);setError('');setMessage('');};
  const body=()=>{const {id,revision,...values}=draft;void id;void revision;return values;};
  return <section id="sources" className="panel sources-panel">
    <div className="section-heading"><h2>{view==='list'?'Подключённые источники':view==='builtin'?'Open-Meteo':view==='new'?'Добавление источника':draft.name}</h2>{view==='list'?<button className="source-primary" onClick={()=>choose(empty())}>+ Добавить источник</button>:<button className="source-secondary" disabled={active} onClick={()=>{setView('list');setError('');setMessage('');}}>← Все источники</button>}</div>
    {active&&<p className="notice compact" role="status">Дождитесь завершения запроса перед переходом на другую страницу.</p>}
    {error&&<div className="error" role="alert">{error}</div>}
    {view==='list'&&<div className="source-cards">
      <article className="source-card"><div className="section-heading"><h3>Open-Meteo</h3><span className="tag">По умолчанию</span></div><p className="muted">Архив погоды, исторические выпуски и текущие прогнозы.</p><span className="small muted">open-meteo.com</span><button className="source-primary" onClick={()=>setView('builtin')}>Открыть источник →</button></article>
      {items.map(source=><article className="source-card" key={source.id}><div className="section-heading"><h3>{source.name}</h3><span className="tag">{!source.enabled?'Выключен':source.trusted?'Включён':'Не подтверждён'}</span></div><p className="muted">{new URL(source.url).hostname}</p><span className="small muted">{source.format.toUpperCase()} · версия {source.revision}</span><button className="source-secondary" onClick={()=>choose(source)}>Настроить →</button></article>)}
    </div>}
    {view==='builtin'&&<><p className="muted">Источник доступен по умолчанию. Подберите данные по выбранной турбине или откройте архив отдельных выпусков.</p><p><a href="/weather">Перейти к архиву выпусков ECMWF ↗</a></p><DiscoveryPanel key={'open-meteo-'+(turbine?.id??'empty')} turbine={turbine} initialSite="open-meteo.com" onLoaded={refresh} onBusyChange={setDiscoveryBusy} disabled={busy}/></>}
    {(view==='new'||view==='edit')&&<>
    {view==='new'&&<DiscoveryPanel key={'new-'+(turbine?.id??'empty')} turbine={turbine} onLoaded={refresh} onBusyChange={setDiscoveryBusy} disabled={busy}/>}
    <details className="source-advanced" open={view==='edit'} key={view}><summary>{view==='edit'?'Настройки подключения':'Настроить API вручную'}</summary>
    <p className="muted">Укажите адрес JSON/CSV и соответствие полей. Агент может предложить настройки по образцу ответа.</p>
    <form onSubmit={e=>{e.preventDefault();void task(async()=>{const saved=await request<Source>(draft.id?'/'+draft.id:'',draft.id?'PUT':'POST',body());setDraft(saved);setView('edit');setDirty(false);setResult(null);await refresh();setMessage('Настройки сохранены. Проверьте загрузку.');});}}>
      <fieldset disabled={active}>
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
    {busy&&<p role="status">Выполняется запрос…</p>}{message&&<p role="status">{message}</p>}
    {proposal&&<div className="notice"><strong>Предложение агента</strong><p>{proposal.explanation}</p>{proposal.unresolved.length>0&&<p>Уточнить: {proposal.unresolved.join('; ')}</p>}<p>Проверка образца: {proposal.validation.ok?`${proposal.validation.rows} строк`:proposal.validation.error}</p><pre>{JSON.stringify(proposal.mapping,null,2)}</pre><p>Токены: {proposal.usage.input_tokens} / {proposal.usage.output_tokens}</p><button disabled={active} onClick={()=>{setDraft(d=>({...d,mapping:proposal.mapping}));setDirty(true);setProposal(null);setResult(null);setMessage('Предложение перенесено в форму. Проверьте поля и сохраните настройки.');}}>Перенести в форму</button></div>}
    {result&&<div><p>Проверено строк: {result.rows}. Время UTC, ветер м/с, температура °C.</p><pre>{JSON.stringify(result.preview,null,2)}</pre>{result.batch_id&&<a href={'/api/sources/batches/'+result.batch_id} target="_blank" rel="noreferrer">Открыть сохранённые данные JSON ↗</a>}<p className="muted small">Время публикации не подтверждено: историческая доступность для прогнозирования не установлена.</p></div>}
    </details>
    </>}
  </section>;
}
