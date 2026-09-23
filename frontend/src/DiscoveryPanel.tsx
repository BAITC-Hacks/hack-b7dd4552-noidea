import {useEffect, useRef, useState} from 'react';
type Turbine = {id:number;name:string;has_data:boolean;start:string;end:string;latitude:number;longitude:number};
type Plan = {id:string;status:string;explanation:string;context:{turbine_id:number;start:string;end:string;latitude:number;longitude:number};request:{site:string;purpose:string};periods?:string[][];chunks:{url:string}[];completed:{batch_id:string;rows:number}[];input_tokens:number;output_tokens:number;documents:{url:string}[];preview?:unknown[];steps:unknown[];plan?:{mapping:{wind_height_m:number}}};
async function api<T>(path:string,body?:unknown):Promise<T>{
  const r=await fetch('/api/discovery'+path,{method:body?'POST':'GET',headers:body?{'Content-Type':'application/json'}:undefined,body:body?JSON.stringify(body):undefined});
  const data=await r.json();if(!r.ok)throw new Error(typeof data.detail==='string'?data.detail:JSON.stringify(data.detail));return data;
}
export function DiscoveryPanel({turbine,onLoaded,initialSite='',onBusyChange,disabled=false}:{turbine?:Turbine;onLoaded:()=>Promise<void>;initialSite?:string;onBusyChange?:(busy:boolean)=>void;disabled?:boolean}){
  const [site,setSite]=useState(initialSite),[purpose,setPurpose]=useState('history'),[plan,setPlan]=useState<Plan|null>(null),[busy,setBusy]=useState(false),[error,setError]=useState('');
  const stop=useRef(false);
  useEffect(()=>{onBusyChange?.(busy);},[busy,onBusyChange]);
  useEffect(()=>()=>{onBusyChange?.(false);},[onBusyChange]);
  const storageKey='wind-discovery-'+(turbine?.id??'none')+'-'+(initialSite||'new');
  useEffect(()=>{let active=true;const saved=localStorage.getItem(storageKey);if(saved)void api<Plan>('/plans/'+saved).then(p=>{if(active){setPlan(p);setSite(p.request.site);setPurpose(p.request.purpose);}}).catch(()=>{if(active)localStorage.removeItem(storageKey);});return()=>{active=false;stop.current=true;};},[storageKey]);
  async function run(){localStorage.removeItem(storageKey);setBusy(true);setError('');setPlan(null);stop.current=false;try{const p=await api<Plan>('/plan',{site,turbine_id:turbine!.id,purpose});setPlan(p);localStorage.setItem(storageKey,p.id);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function load(){if(!plan)return;setBusy(true);setError('');stop.current=false;try{let p=plan;while(p.status==='ready'&&!stop.current){p=await api<Plan>('/plans/'+p.id+'/next',{});setPlan(p);}await onLoaded();}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  return <div className="discovery"><h3>Подобрать данные по сайту</h3>
    <p className="muted">Введите сайт. Агент найдёт документацию и составит запрос по координатам и периоду выбранной турбины.</p>
    {turbine?.has_data?<p className="small muted">{turbine.name} · {turbine.latitude}, {turbine.longitude} · измерения {turbine.start?.slice(0,10)} — {turbine.end?.slice(0,10)}</p>:<p className="notice">Сначала создайте турбину и импортируйте измерения.</p>}
    <form className="filters" onSubmit={e=>{e.preventDefault();void run();}}>
      <label>Сайт или страница документации<input required placeholder="open-meteo.com" value={site} disabled={busy||disabled} onChange={e=>setSite(e.target.value)}/></label>
      <label>Для чего нужны данные<select disabled={busy||disabled} value={purpose} onChange={e=>setPurpose(e.target.value)}><option value="history">Проверить историю и время</option><option value="historical_forecast">Исторические прогнозы для ML</option><option value="forecast">Текущий прогноз погоды</option></select></label>
      <button disabled={busy||disabled||!turbine?.has_data}>{busy?'Выполняется…':'Найти и проверить запрос'}</button>
    </form>
    <p className="small muted">Запуск разрешает чтение сайта и его поддоменов и использует API-кредиты: до 6 обращений к модели. Агент получает координаты и описание периода, без полного CSV. Пробная загрузка — один день. Остальной период загружается после просмотра результата.</p>
    {error&&<div className="error" role="alert">{error}</div>}
    {plan&&<div><p><strong>{plan.status==='ready'?'Запрос проверен':plan.status==='complete'?'Загрузка завершена':'Нужно уточнение'}</strong></p><p>{plan.explanation}</p>
      {plan.periods&&<p>Период: {plan.periods[0][0]} — {plan.periods[plan.periods.length-1][1]} · {plan.chunks.length} частей · загружено {plan.completed.length}. Высота ветра источника: {plan.plan?.mapping.wind_height_m} м.</p>}
      <p className="small muted">Для истории добавлен запас по одному дню: часовой пояс измерений ещё не подтверждён. Данные не совмещаются с CSV автоматически. Историческая доступность прогнозов требует отдельной проверки; baseline их не использует.</p>
      {plan.status==='ready'&&<div className="filters"><button type="button" disabled={busy||disabled} onClick={()=>void load()}>{plan.completed.length?'Продолжить загрузку':'Загрузить весь период'}</button>{busy&&<button type="button" onClick={()=>{stop.current=true;}}>Остановить после текущей части</button>}</div>}
      {plan.completed.length>0&&<details><summary>Сохранённые части ({plan.completed.length})</summary>{plan.completed.map((part,i)=><p key={part.batch_id}><a href={'/api/sources/batches/'+part.batch_id} target="_blank" rel="noreferrer">Часть {i+1} · {part.rows} строк ↗</a></p>)}</details>}
      <details><summary>Документация, запрос и журнал</summary>{plan.documents.map(d=><p key={d.url}><a href={d.url} target="_blank" rel="noreferrer">{d.url}</a></p>)}<p>Токены: {plan.input_tokens} / {plan.output_tokens}</p><pre>{JSON.stringify({url:plan.chunks[0]?.url,preview:plan.preview,steps:plan.steps},null,2)}</pre></details>
    </div>}
  </div>;
}
