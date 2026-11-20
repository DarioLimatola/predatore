//! Compiled circuit and the decide/feedback cycle (port of VectorBooleanNet.decide/feedback).
use crate::bits;
use crate::hdc::{AssociativeCore, Codec};
use crate::key::Key;
use crate::lru::Ordered;
use crate::memory::{ContextMemoryCore, TERMINAL_CREDIT};
use std::collections::{HashMap, VecDeque};

pub const MAX_CONTEXT_HISTORY: usize = 8;
pub const MAX_DELAY: usize = 3;

/// One operand transform as disjoint shifted masks (a bit permutation), optionally
/// combined with the value itself (neighbour operators).
#[derive(Clone, Debug)]
pub struct Plan {
    pub base: bool,
    pub xor: bool,
    pub groups: Vec<Vec<(Vec<u64>, isize)>>,
}

#[derive(Clone, Debug)]
pub struct Rule {
    pub a: usize,
    pub b: usize,
    pub hold: i64,
    pub write: Vec<u64>,
    pub lut: [Vec<u64>; 4],
    pub plan_a: Plan,
    pub plan_b: Plan,
    pub delay_a: usize,
    pub delay_b: usize,
    /// Plain rotation shifts, used when tensor operators are disabled.
    pub shift_a: usize,
    pub shift_b: usize,
}

#[derive(Clone, Debug)]
pub struct Circuit {
    pub sensors: usize,
    pub action_bits: usize,
    pub actions: usize,
    pub width: usize,
    pub nw: usize,
    pub nodes: usize,
    pub rules: Vec<Rule>,
    pub outputs: Vec<usize>,
    pub output_lanes: Vec<usize>,
    pub ticks: usize,
    pub trace_depth: usize,
    pub memory_capacity: usize,
    pub broadcast: bool,
    pub ancestors: Vec<usize>,
    pub active: Vec<usize>,
    pub codec: Codec,
}

#[derive(Clone, Copy, Debug)]
pub struct Flags {
    pub memory: bool,
    pub tensor: bool,
    pub sparse: bool,
    pub associative: bool,
    pub contextual: bool,
    pub hdc: bool,
    pub plasticity: bool,
    pub circuit_plasticity: bool,
    pub collect_experience: bool,
    pub emit_events: bool,
}

pub type Predictions = Ordered<(Key, usize), u64>;
pub type CircuitEvidence = crate::lru::FxMap<(usize, usize), [Vec<u64>; 4]>;

#[derive(Clone, Debug)]
pub struct Payload {
    pub observation: u64,
    pub history: Vec<(u64, usize)>,
}

#[derive(Clone, Debug)]
pub struct Event {
    pub context_key: Option<Key>,
    pub eligible: Vec<(Payload, usize)>,
    pub observation: u64,
    pub history: Vec<(u64, usize)>,
    pub previous: u64,
    pub valid: u64,
    pub action: usize,
    pub feedback: u8,
    pub next: u64,
    pub terminal: bool,
}

#[derive(Clone, Debug)]
pub struct LastDecoder {
    pub context_key: Option<Key>,
    pub context_override: Option<usize>,
    pub proposed: usize,
    pub address: Key,
    pub action: usize,
    pub source: usize, // index into ORIGINS
}

pub const ORIGINS: [&str; 4] = ["circuit", "short", "context", "hdc"];

/// Working state; reset() at the start of every episode.
#[derive(Clone, Debug)]
pub struct Runtime {
    pub vectors: Vec<u64>,
    pub context: u64,
    pub context_valid: u64,
    pub observable_history: VecDeque<(u64, usize)>,
    pub context_trace: VecDeque<(Payload, usize)>,
    pub current_context_key: Option<Key>,
    pub circuit_trace: VecDeque<(bool, Vec<(usize, usize, Vec<u64>)>)>,
    pub last: Option<(Key, usize, u64, u64, u64)>,
    pub trace: VecDeque<(Key, usize, u64, u64, u64)>,
    pub input_history: VecDeque<Vec<u64>>,
    pub last_payload: Option<Payload>,
    pub update_count: u64,
    pub captured: bool,
    pub next_observation: u64,
    pub decoder_counts: [u64; 4],
    pub last_decoder: Option<LastDecoder>,
    pub last_prediction_error: Option<u64>,
}

impl Runtime {
    pub fn new(c: &Circuit) -> Runtime {
        Runtime {
            vectors: vec![0; c.nodes * c.nw],
            context: 0,
            context_valid: 0,
            observable_history: VecDeque::new(),
            context_trace: VecDeque::new(),
            current_context_key: None,
            circuit_trace: VecDeque::new(),
            last: None,
            trace: VecDeque::new(),
            input_history: VecDeque::new(),
            last_payload: None,
            update_count: 0,
            captured: false,
            next_observation: 0,
            decoder_counts: [0; 4],
            last_decoder: None,
            last_prediction_error: None,
        }
    }
}

fn push_bounded<T>(q: &mut VecDeque<T>, v: T, max: usize) {
    if max == 0 {
        return;
    }
    q.push_back(v);
    while q.len() > max {
        q.pop_front();
    }
}

pub struct Brain<'a> {
    pub c: &'a Circuit,
    pub f: Flags,
    pub ctx: &'a mut ContextMemoryCore,
    pub short: &'a mut ContextMemoryCore,
    pub assoc: &'a mut AssociativeCore,
    pub preds: &'a mut Predictions,
    pub evid: &'a mut CircuitEvidence,
    pub rt: Runtime,
    pub events: Vec<Event>,
    s: Scratch,
}

/// Reusable buffers: the decision loop allocates nothing per step.
struct Scratch {
    inputs: Vec<u64>,
    new: Vec<u64>,
    val: Vec<u64>,
    oa: Vec<u64>,
    ob: Vec<u64>,
    t1: Vec<u64>,
    t2: Vec<u64>,
    acc: Vec<u64>,
    activity: Vec<u64>,      // nodes * 4 * nw
    tracked: Vec<bool>,      // per node, set when activity was written this tick
    is_ancestor: Vec<bool>,
    spare: Vec<Vec<u64>>,    // recycled input_history buffers
}

#[inline]
fn apply_perm(v: &[u64], group: &[(Vec<u64>, isize)], width: usize, out: &mut [u64], t1: &mut [u64], t2: &mut [u64]) {
    for x in out.iter_mut() { *x = 0; }
    let n = out.len();
    if n == 1 {
        let x = v[0];
        let mut o = 0u64;
        for (mask, shift) in group {
            let m = x & mask[0];
            o |= if *shift >= 0 { if *shift >= 64 { 0 } else { m << *shift } } else { let s = -*shift; if s >= 64 { 0 } else { m >> s } };
        }
        out[0] = o & bits::top_mask(width);
        return;
    }
    for (mask, shift) in group {
        for i in 0..n { t1[i] = v[i] & mask[i]; }
        if *shift >= 0 { bits::shl(t1, *shift as usize, t2); } else { bits::shr(t1, (-*shift) as usize, t2); }
        for i in 0..n { out[i] |= t2[i]; }
    }
    bits::mask_in_place(out, width);
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn operand(c: &Circuit, f: &Flags, inputs: &[u64], hist: &VecDeque<Vec<u64>>, source: usize, delay: usize, shift: usize,
           plan: &Plan, out: &mut [u64], val: &mut [u64], t1: &mut [u64], t2: &mut [u64], acc: &mut [u64]) {
    let nw = c.nw;
    let w = c.width;
    let delay = if f.memory { delay } else { 0 };
    if delay == 0 {
        val.copy_from_slice(&inputs[source * nw..(source + 1) * nw]);
    } else if hist.len() >= delay {
        let h = &hist[hist.len() - delay];
        val.copy_from_slice(&h[source * nw..(source + 1) * nw]);
    } else {
        for x in val.iter_mut() { *x = 0; }
    }
    if !f.tensor {
        bits::rotate(val, shift, w, out, t1);
        return;
    }
    if plan.base { out.copy_from_slice(val); } else { for x in out.iter_mut() { *x = 0; } }
    for group in plan.groups.iter() {
        apply_perm(val, group, w, acc, t1, t2);
        if plan.xor { for i in 0..nw { out[i] ^= acc[i]; } } else { for i in 0..nw { out[i] |= acc[i]; } }
    }
    bits::mask_in_place(out, w);
}

impl<'a> Brain<'a> {
    pub fn new(c: &'a Circuit, f: Flags, ctx: &'a mut ContextMemoryCore, short: &'a mut ContextMemoryCore,
               assoc: &'a mut AssociativeCore, preds: &'a mut Predictions, evid: &'a mut CircuitEvidence) -> Self {
        let nw = c.nw;
        let total = (c.sensors + c.nodes) * nw;
        let mut is_ancestor = vec![false; c.nodes];
        for &a in c.ancestors.iter() { is_ancestor[a] = true; }
        Brain {
            c, f, ctx, short, assoc, preds, evid,
            rt: Runtime::new(c),
            events: Vec::new(),
            s: Scratch {
                inputs: vec![0; total], new: vec![0; c.nodes * nw], val: vec![0; nw], oa: vec![0; nw], ob: vec![0; nw],
                t1: vec![0; nw + 1], t2: vec![0; nw + 1], acc: vec![0; nw],
                activity: vec![0; c.nodes * 4 * nw], tracked: vec![false; c.nodes], is_ancestor, spare: Vec::new(),
            },
        }
    }

    pub fn reset(&mut self) {
        let old = std::mem::replace(&mut self.rt, Runtime::new(self.c));
        for buf in old.input_history { self.s.spare.push(buf); }
    }

    pub fn decide(&mut self, obs: u64) -> usize {
        let c = self.c;
        let nw = c.nw;
        let w = c.width;
        let sn = c.sensors;
        let address = {
            let mut k = Key::from_u64(obs);
            k.or_shifted(self.rt.context, sn as u32);
            k.or_shifted(self.rt.context_valid, 2 * sn as u32);
            k
        };
        let history: Vec<(u64, usize)> = if self.f.memory { self.rt.observable_history.iter().copied().collect() } else { Vec::new() };
        let key = c.codec.encode(obs, &history);
        self.rt.current_context_key = Some(key);
        self.rt.last_payload = Some(Payload { observation: obs, history });
        let collect_activity = self.f.collect_experience && self.f.circuit_plasticity;
        let indices: &[usize] = if self.f.sparse { &c.ancestors } else { &c.active };
        let s = &mut self.s;
        {
            let mixed = &mut s.val;
            for i in 0..sn {
                let bit = (obs >> i) & 1;
                if c.broadcast {
                    for x in mixed.iter_mut() { *x = if bit == 1 { u64::MAX } else { 0 }; }
                    bits::mask_in_place(mixed, w);
                } else {
                    bits::rotate(&[obs], i % w, w, mixed, &mut s.t1[..nw]);
                    if bit == 1 {
                        for x in mixed.iter_mut() { *x = !*x; }
                        bits::mask_in_place(mixed, w);
                    }
                }
                s.inputs[i * nw..(i + 1) * nw].copy_from_slice(mixed);
            }
        }
        let top = bits::top_mask(w);
        for _ in 0..c.ticks {
            if self.f.memory {
                s.inputs[sn * nw..].copy_from_slice(&self.rt.vectors);
            } else {
                for x in s.inputs[sn * nw..].iter_mut() { *x = 0; }
            }
            for x in s.new.iter_mut() { *x = 0; }
            if collect_activity { for t in s.tracked.iter_mut() { *t = false; } }
            for &i in indices.iter() {
                let r = &c.rules[i];
                operand(c, &self.f, &s.inputs, &self.rt.input_history, r.a, r.delay_a, r.shift_a, &r.plan_a,
                        &mut s.oa, &mut s.val, &mut s.t1[..nw], &mut s.t2[..nw], &mut s.acc);
                operand(c, &self.f, &s.inputs, &self.rt.input_history, r.b, r.delay_b, r.shift_b, &r.plan_b,
                        &mut s.ob, &mut s.val, &mut s.t1[..nw], &mut s.t2[..nw], &mut s.acc);
                let tracked = collect_activity && s.is_ancestor[i];
                for k in 0..nw {
                    let m = if k == nw - 1 { top } else { u64::MAX };
                    let a = s.oa[k];
                    let b = s.ob[k];
                    let na = (a ^ m) & m;
                    let nb = (b ^ m) & m;
                    let hold = if r.hold >= 0 && self.f.memory { s.inputs[r.hold as usize * nw + k] } else { 0 };
                    let write = r.write[k] & !hold & m;
                    if tracked {
                        let base = i * 4 * nw;
                        s.activity[base + k] = na & nb & write;
                        s.activity[base + nw + k] = na & b & write;
                        s.activity[base + 2 * nw + k] = a & nb & write;
                        s.activity[base + 3 * nw + k] = a & b & write;
                    }
                    let value = (na & nb & r.lut[0][k]) | (na & b & r.lut[1][k]) | (a & nb & r.lut[2][k]) | (a & b & r.lut[3][k]);
                    let old = s.inputs[(sn + i) * nw + k];
                    s.new[i * nw + k] = ((old & !write) | (value & write)) & m;
                }
                if tracked { s.tracked[i] = true; }
            }
            std::mem::swap(&mut self.rt.vectors, &mut s.new);
            if self.f.memory {
                let mut buf = if self.rt.input_history.len() >= MAX_DELAY { self.rt.input_history.pop_front().unwrap() } else { s.spare.pop().unwrap_or_default() };
                buf.clear();
                buf.extend_from_slice(&s.inputs);
                self.rt.input_history.push_back(buf);
            }
        }
        let mut proposed = 0usize;
        for (bit, (&n, &lane)) in c.outputs.iter().zip(c.output_lanes.iter()).enumerate() {
            let v = bits::bit(&self.rt.vectors[n * nw..(n + 1) * nw], lane);
            proposed |= (v as usize) << bit;
        }
        let mut action = proposed;
        let mut origin = 0usize;
        let mut contextual_action = None;
        if self.f.associative {
            if let Some(a) = self.short.choose(&address, action) {
                action = a;
                origin = 1;
            }
            if self.f.contextual {
                contextual_action = self.ctx.choose(&key, action);
                if let Some(a) = contextual_action {
                    action = a;
                    origin = 2;
                } else if self.f.hdc {
                    let anchor = obs & self.assoc.sensor_mask;
                    for nb in self.assoc.bucket_tail(anchor) {
                        self.sync_associative(&nb);
                    }
                    let rejected = self.ctx.evidence(&key).map(|e| e.bad).unwrap_or(0);
                    let ctx = &*self.ctx;
                    let allowed = |k: &Key| ctx.rows.contains(k);
                    if let Some(a) = self.assoc.choose(&key, action, rejected, &allowed) {
                        action = a;
                        origin = 3;
                    }
                }
            }
        }
        self.rt.decoder_counts[origin] += 1;
        self.rt.last_decoder = Some(LastDecoder { context_key: Some(key), context_override: contextual_action, proposed, address, action, source: origin });
        if collect_activity {
            let s = &self.s;
            let mut patterns = Vec::new();
            for &node in c.ancestors.iter() {
                if !s.tracked[node] { continue; }
                for pattern in 0..4 {
                    let base = node * 4 * nw + pattern * nw;
                    let l: Vec<u64> = (0..nw).map(|k| s.activity[base + k] & c.rules[node].write[k]).collect();
                    if !bits::is_zero(&l) {
                        patterns.push((node, pattern, l));
                    }
                }
            }
            push_bounded(&mut self.rt.circuit_trace, (action == proposed, patterns), c.trace_depth);
        }
        let payload = self.rt.last_payload.clone().unwrap();
        push_bounded(&mut self.rt.context_trace, (payload, action), c.trace_depth);
        if self.f.memory {
            push_bounded(&mut self.rt.observable_history, (obs, action), MAX_CONTEXT_HISTORY);
        } else {
            self.rt.observable_history.clear();
        }
        let last = (address, action, obs, self.rt.context, self.rt.context_valid);
        self.rt.last = Some(last);
        push_bounded(&mut self.rt.trace, last, c.trace_depth);
        self.rt.context = if self.f.memory { obs } else { 0 };
        self.rt.context_valid = self.f.memory as u64;
        action
    }

    pub fn sync_associative(&mut self, key: &Key) {
        let (p, n) = self.ctx.evidence(key).map(|e| (e.good, e.bad)).unwrap_or((0, 0));
        let differs = match self.assoc.entries.get(key) {
            None => true,
            Some(old) => (old.positive, old.negative) != (p, n),
        };
        if differs {
            self.assoc.observe(*key, p, n);
        }
    }

    pub fn transition(&mut self, next_obs: u64, captured: bool) {
        self.rt.next_observation = next_obs;
        self.rt.captured = captured;
    }

    fn reinforce(&mut self, node: usize, pattern: usize, lanes: &[u64], positive: bool) {
        let nw = self.c.nw;
        let w = self.c.width;
        let entry = self.evid.entry((node, pattern)).or_insert_with(|| [vec![0; nw], vec![0; nw], vec![0; nw], vec![0; nw]]);
        let (li, hi) = if positive { (0, 1) } else { (2, 3) };
        for k in 0..nw {
            let m = if k == nw - 1 { bits::top_mask(w) } else { u64::MAX };
            let low = entry[li][k];
            let high = entry[hi][k];
            let write = lanes[k] & !(low & high) & m;
            entry[li][k] = low ^ write;
            entry[hi][k] = high ^ (low & write);
        }
    }

    /// reward: the float the world returned (only its sign is used here).
    pub fn feedback(&mut self, reward: f64) -> Result<(), &'static str> {
        let Some((address, action, obs, context, valid)) = self.rt.last else { return Ok(()) };
        let positive = reward > 0.0;
        let negative = reward < 0.0;
        let next_obs = self.rt.next_observation;
        let pkey = (address, action);
        self.rt.last_prediction_error = self.preds.get(&pkey).map(|p| p ^ next_obs);
        if self.f.plasticity && self.f.circuit_plasticity && reward != 0.0 {
            let mut credited: HashMap<(usize, usize), Vec<u64>> = HashMap::new();
            let mut order: Vec<(usize, usize)> = Vec::new();
            for (aligned, patterns) in self.rt.circuit_trace.iter().rev() {
                if !aligned { continue; }
                for (node, pattern, lanes) in patterns.iter() {
                    let e = credited.entry((*node, *pattern)).or_insert_with(|| { order.push((*node, *pattern)); vec![0; lanes.len()] });
                    for (x, y) in e.iter_mut().zip(lanes.iter()) { *x |= *y; }
                }
            }
            for k in order {
                let lanes = credited.remove(&k).unwrap();
                self.reinforce(k.0, k.1, &lanes, positive);
            }
        }
        if self.f.plasticity {
            if positive || negative {
                self.short.update(address, action, positive, 1, false).map_err(|_| "invalid update")?;
                self.rt.update_count += 1;
                if let Some(key) = self.rt.current_context_key {
                    self.ctx.update(key, action, positive, 1, false).map_err(|_| "invalid update")?;
                    self.sync_associative(&key);
                }
            }
            if self.rt.captured {
                let items: Vec<(Payload, usize)> = self.rt.context_trace.iter().cloned().collect();
                for (payload, act) in items {
                    let key = self.c.codec.encode(payload.observation, &payload.history);
                    self.ctx.update(key, act, true, TERMINAL_CREDIT, true).map_err(|_| "invalid update")?;
                    self.sync_associative(&key);
                }
            }
            if self.preds.contains(&pkey) {
                *self.preds.get_mut(&pkey).unwrap() = next_obs;
            } else {
                self.preds.insert(pkey, next_obs);
            }
            self.preds.move_to_end(&pkey);
            if self.preds.len() > self.c.memory_capacity {
                self.preds.pop_first();
            }
        }
        if self.f.collect_experience && self.f.emit_events {
            let lp = self.rt.last_payload.clone().unwrap();
            self.events.push(Event {
                context_key: self.rt.current_context_key,
                eligible: if self.rt.captured { self.rt.context_trace.iter().cloned().collect() } else { Vec::new() },
                observation: obs,
                history: lp.history,
                previous: context,
                valid,
                action,
                feedback: (positive as u8) | ((negative as u8) << 1),
                next: next_obs,
                terminal: self.rt.captured,
            });
        }
        self.rt.captured = false;
        Ok(())
    }
}
