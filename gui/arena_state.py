"""GUI-independent Arena helpers (V7: real episodes only, no previews)."""
from pathlib import Path
from engine.vault import read_json


def frame_summary(frame):
    """Metrics from a sampled decision; t=0 carries position only."""
    from engine.world import ACTIONS
    action = frame.get('action')
    action_name = ACTIONS[action] if type(action) is int and 0 <= action < len(ACTIONS) else '—'
    fields = [f'passo {frame.get("t", "—")}', f'azione {action_name}']
    obs = frame.get('observation')
    if obs:
        fields.append('preda visibile' if obs[0] else 'preda fuori vista')
    source = frame.get('decoder', {}).get('source')
    if source:
        fields.append(f'decoder {source}')
    if 'captured' in frame:
        fields.append('catturata' if frame['captured'] else 'in corso')
    if 'knowledge_entries' in frame:
        fields.append(f'RAM {frame["knowledge_entries"]} voci')
    events = frame.get('world_events') or {}
    if events.get('due_tick') is not None:
        fields.append(f'contatto {events["tag_tick"]}, conferma prevista {events["due_tick"]}')
    prey_action = events.get('prey_action')
    if type(prey_action) is int and 0 <= prey_action < len(ACTIONS):
        fields.append(f'azione preda {ACTIONS[prey_action]}')
    return ' · '.join(fields)


def brain_difference(after, before):
    """Readable structural and memory differences between two exported brains.

    Neutral categories only: the interface never attributes a cause.
    """
    lines = []
    na, nb = after.get('nodes', []), before.get('nodes', [])
    ra, rb = after.get('rules', []), before.get('rules', [])
    lines.append(f'Moduli: {len(nb)} → {len(na)}' + (f'  (larghezza {before.get("shape")} → {after.get("shape")})'
                                                   if after.get('shape') != before.get('shape') else ''))
    common = min(len(na), len(nb), len(ra), len(rb))
    if len(na) != len(nb):
        lines.append('Nota: il numero di moduli è cambiato; i moduli sono confrontati per posizione, '
                     'quindi dopo un’aggiunta o una rimozione le differenze per modulo sono indicative.')
    wired = [i for i in range(common) if (na[i]['a'], na[i]['b'], ra[i].get('hold')) != (nb[i]['a'], nb[i]['b'], rb[i].get('hold'))]
    toggled = [i for i in range(common) if na[i].get('active') != nb[i].get('active')]
    lut = [i for i in range(common) if list(ra[i].get('lut', [])) != list(rb[i].get('lut', []))]
    lut_bits = sum(bin(int(x) ^ int(y)).count('1') for i in lut for x, y in zip(ra[i]['lut'], rb[i]['lut']))
    keys = ('shift_a', 'shift_b', 'operator_a', 'operator_b', 'axis_a', 'axis_b', 'delay_a', 'delay_b', 'write', 'shape')
    tensor = [i for i in range(common) if any(ra[i].get(k) != rb[i].get(k) for k in keys)]
    lines.append(f'Tabelle LUT modificate: {len(lut)} moduli, {lut_bits} bit' + (f'  {lut[:12]}' if lut else ''))
    lines.append(f'Operatori, spostamenti, ritardi o maschere modificati: {len(tensor)}' + (f'  {tensor[:12]}' if tensor else ''))
    lines.append(f'Collegamenti cambiati (ingressi, hold): {len(wired)}' + (f'  {wired[:12]}' if wired else ''))
    if toggled:
        lines.append(f'Moduli attivati/disattivati: {toggled[:12]}')
    if (after.get('outputs'), after.get('output_lanes')) != (before.get('outputs'), before.get('output_lanes')):
        lines.append(f'Uscite: {before.get("outputs")}/{before.get("output_lanes")} → {after.get("outputs")}/{after.get("output_lanes")}')
    else:
        lines.append('Uscite invariate')

    def rows(brain, name):
        block = brain.get(name)
        if not isinstance(block, dict) or block.get('rows') is None:
            return None
        return {row[0]: tuple(row[1:4]) for row in block['rows']}

    for name, label in (('contextual_memory', 'Memoria contestuale'), ('knowledge', 'Memoria breve')):
        a, b = rows(after, name), rows(before, name)
        if a is None or b is None:
            lines.append(f'{label}: non disponibile in uno dei due cervelli')
            continue
        new = len(a.keys() - b.keys()); gone = len(b.keys() - a.keys())
        changed = sum(1 for k in a.keys() & b.keys() if a[k] != b[k])
        lines.append(f'{label}: {len(b)} → {len(a)} righe · nuove {new} · rimosse {gone} · modificate {changed}')
    ea = (after.get('associative_memory') or {}).get('entries')
    eb = (before.get('associative_memory') or {}).get('entries')
    if ea is not None and eb is not None:
        lines.append(f'Memoria associativa (HDC): {len(eb)} → {len(ea)} voci')
    return '\n'.join(lines)
