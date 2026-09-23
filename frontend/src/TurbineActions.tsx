import {useEffect, useId, useRef, useState} from 'react';

type Turbine = {id: number; name: string};
type DeletedTurbine = Turbine & {deleted_at: string; has_data: boolean};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === 'string' ? body.detail : `Ошибка запроса (${response.status}). Попробуйте ещё раз.`);
  }
  return response.json();
}

export function TurbineDelete({turbine, disabled = false, onBusyChange, onDeleted}: {
  turbine: Turbine;
  disabled?: boolean;
  onBusyChange?: (busy: boolean) => void;
  onDeleted: (id: number) => Promise<void>;
}) {
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [deleted, setDeleted] = useState(false);
  const [error, setError] = useState('');
  const pending = useRef(false);
  const currentId = useRef(turbine.id);
  const headingId = useId();

  useEffect(() => {
    currentId.current = turbine.id;
    setConfirming(false);
    setDeleted(false);
    setError('');
  }, [turbine.id]);

  async function remove() {
    if (pending.current || disabled || deleted) return;
    const id = turbine.id;
    pending.current = true;
    setBusy(true);
    setError('');
    onBusyChange?.(true);
    let completed = false;
    try {
      await api<{id: number; deleted: boolean}>(`/api/turbines/${id}`, {method: 'DELETE'});
      completed = true;
      if (currentId.current === id) {setDeleted(true); setConfirming(false);}
      await onDeleted(id);
    } catch (cause) {
      if (currentId.current === id) setError(`${completed ? 'Турбина уже перемещена в корзину, но не удалось обновить список. Обновите страницу. ' : ''}${(cause as Error).message}`);
    } finally {
      pending.current = false;
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  return <div className="turbine-delete">
    {!confirming && !deleted && <button type="button" className="source-secondary" disabled={disabled || busy} onClick={() => {setError(''); setConfirming(true);}}>Удалить турбину</button>}
    {confirming && <div className="turbine-delete-confirm" role="group" aria-labelledby={headingId}>
      <h3 id={headingId}>Удалить «{turbine.name}»?</h3>
      <p className="muted">Турбина исчезнет из рабочих списков. Данные сохранятся, турбину можно будет восстановить из раздела «Удалённые турбины».</p>
      <div className="turbine-delete-actions">
        <button type="button" className="source-secondary" disabled={disabled || busy} onClick={() => {setConfirming(false); setError('');}}>Отмена</button>
        <button type="button" className="source-primary" disabled={disabled || busy} onClick={() => void remove()}>{busy ? 'Перемещаем в корзину…' : 'Да, удалить турбину'}</button>
      </div>
    </div>}
    {deleted && <p className="muted" role="status">Турбина перемещена в корзину. Данные сохранены.</p>}
    {error && <div className="error" role="alert">{error}</div>}
  </div>;
}

export function DeletedTurbines({version = 0, disabled = false, onBusyChange, onRestored}: {
  version?: number;
  disabled?: boolean;
  onBusyChange?: (busy: boolean) => void;
  onRestored: (id: number) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const [items, setItems] = useState<DeletedTurbine[]>([]);
  const [loading, setLoading] = useState(false);
  const [restoringId, setRestoringId] = useState<number | null>(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [reload, setReload] = useState(0);
  const pending = useRef(false);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    setLoading(true);
    setError('');
    api<{items: DeletedTurbine[]}>('/api/turbines/deleted', {signal: controller.signal})
      .then(result => {if (!controller.signal.aborted) setItems(result.items);})
      .catch(cause => {if (!controller.signal.aborted) setError((cause as Error).message);})
      .finally(() => {if (!controller.signal.aborted) setLoading(false);});
    return () => controller.abort();
  }, [open, version, reload]);

  async function restore(turbine: DeletedTurbine) {
    if (pending.current || disabled || loading) return;
    pending.current = true;
    setRestoringId(turbine.id);
    setError('');
    setNotice('');
    onBusyChange?.(true);
    let completed = false;
    try {
      await api<Turbine>(`/api/turbines/${turbine.id}/restore`, {method: 'POST'});
      completed = true;
      setItems(current => current.filter(item => item.id !== turbine.id));
      setReload(value => value + 1);
      setNotice(`«${turbine.name}» восстановлена и доступна в рабочем списке.`);
      await onRestored(turbine.id);
    } catch (cause) {
      setError(`${completed ? 'Турбина уже восстановлена, но не удалось обновить рабочий список. Обновите страницу. ' : ''}${(cause as Error).message}`);
      if (completed) setNotice('');
    } finally {
      pending.current = false;
      setRestoringId(null);
      onBusyChange?.(false);
    }
  }

  return <details className="turbine-trash" onToggle={event => setOpen(event.currentTarget.open)}>
    <summary>Удалённые турбины</summary>
    <p className="muted">Здесь хранятся турбины, убранные из рабочих списков. Их измерения и сохранённые результаты можно вернуть вместе с турбиной.</p>
    <button type="button" className="source-secondary" disabled={disabled || loading || restoringId !== null} onClick={() => setReload(value => value + 1)}>Обновить список</button>
    {loading && <p className="muted" role="status">Загружаем корзину…</p>}
    {error && <div className="error" role="alert">{error}</div>}
    {notice && <p className="muted" role="status">{notice}</p>}
    {!loading && !error && !items.length && <p className="muted">Удалённых турбин нет.</p>}
    {!loading && <div className="turbine-trash-list">{items.map(turbine => <div className="turbine-trash-item" key={turbine.id}>
      <div><strong>{turbine.name}</strong><p className="small muted">{turbine.has_data ? 'Измерения сохранены' : 'Без импортированных измерений'} · удалена {turbine.deleted_at ? turbine.deleted_at.slice(0, 16).replace('T', ' ') + ' UTC' : '—'}</p></div>
      <button type="button" className="source-secondary" disabled={disabled || restoringId !== null} onClick={() => void restore(turbine)}>{restoringId === turbine.id ? 'Восстанавливаем…' : 'Восстановить'}</button>
    </div>)}</div>}
  </details>;
}
