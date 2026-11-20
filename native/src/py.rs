//! Python bindings. Each class mirrors the Python API of the object it replaces.
use crate::bits;
use crate::canon;
use crate::circuit::{Brain, Circuit, CircuitEvidence, Event, Flags, Plan, Predictions, Rule, ORIGINS};
use crate::hdc::{AssociativeCore, Codec, Symbols};
use crate::key::Key;
use crate::memory::{self, ContextMemoryCore, Meta, Row, PLANES, SOURCE_LIMIT};
use crate::world::{self, Geometry, Kind};
use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyTuple};

static VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Globally unique mutation stamp: equal stamps mean equal content (copies keep it).
fn bump() -> u64 {
    VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

fn verr<T>(msg: &str) -> PyResult<T> {
    Err(PyValueError::new_err(msg.to_string()))
}

fn key_of(obj: &Bound<'_, PyAny>) -> PyResult<Key> {
    let s: String = match obj.extract::<String>() {
        Ok(s) => s,
        Err(_) => obj.str()?.to_string(),
    };
    match Key::parse(&s) {
        Some(k) if k.to_decimal() == s => Ok(k),
        _ => verr("Invalid memory address"),
    }
}

fn int_to_words(obj: &Bound<'_, PyAny>, nw: usize) -> PyResult<Vec<u64>> {
    if let Ok(v) = obj.extract::<u64>() {
        let mut out = vec![0u64; nw.max(1)];
        out[0] = v;
        if nw == 0 { return Ok(vec![]); }
        return Ok(out);
    }
    let neg: bool = obj.lt(0)?;
    if neg {
        return verr("negative bit vector");
    }
    let bytes = obj.call_method1("to_bytes", (nw * 8, "little")).map_err(|_| PyValueError::new_err("bit vector wider than its width"))?;
    let b: &[u8] = bytes.cast::<PyBytes>().map_err(|_| PyValueError::new_err("bytes"))?.as_bytes();
    Ok((0..nw).map(|i| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap())).collect())
}

fn words_to_int(py: Python<'_>, v: &[u64]) -> PyResult<Py<PyAny>> {
    if v.iter().skip(1).all(|&x| x == 0) {
        let x = v.first().copied().unwrap_or(0);
        return Ok(x.into_pyobject(py)?.into_any().unbind());
    }
    let mut b = Vec::with_capacity(v.len() * 8);
    for w in v {
        b.extend_from_slice(&w.to_le_bytes());
    }
    let int = py.get_type::<pyo3::types::PyInt>();
    Ok(int.call_method1("from_bytes", (PyBytes::new(py, &b), "little"))?.unbind())
}

fn row_from_py(obj: &Bound<'_, PyAny>, mask: u64) -> PyResult<Row> {
    let groups: Vec<Vec<u64>> = obj.extract()?;
    if groups.len() != 2 || groups.iter().any(|g| g.len() != PLANES) {
        return verr("Invalid Boolean planes");
    }
    let mut row = memory::EMPTY;
    for (gi, g) in groups.iter().enumerate() {
        for (i, &v) in g.iter().enumerate() {
            row[gi][i] = v & mask;
        }
    }
    Ok(row)
}

fn row_to_py<'py>(py: Python<'py>, row: &Row) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, [PyList::new(py, row[0].iter())?, PyList::new(py, row[1].iter())?])
}

// ---------------------------------------------------------------- ContextMemory

#[pyclass(module = "predatore_native")]
#[derive(Clone)]
pub struct ContextMemory {
    pub core: ContextMemoryCore,
    pub version: u64,
    pub digest_cache: Option<(u64, String)>,
}

#[pymethods]
impl ContextMemory {
    #[getter]
    fn version(&self) -> u64 { self.version }
    fn __copy__(&self) -> Self { self.clone() }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self { self.clone() }
    /// sha256 of json.dumps(self.export(), sort_keys=True); cached per version.
    fn digest(&mut self) -> String {
        if let Some((v, d)) = &self.digest_cache { if *v == self.version { return d.clone(); } }
        let keys: Vec<Key> = self.core.rows.iter().map(|(k, _)| k).collect();
        let mut j = String::with_capacity(keys.len() * 200 + 32);
        j.push_str("{\"clock\": ");
        j.push_str(&self.core.clock.to_string());
        j.push_str(", \"rows\": [");
        for (n, k) in keys.iter().enumerate() {
            if n > 0 { j.push_str(", "); }
            let (p, t, h) = self.core.packed(k);
            let e = self.core.rows.get(k).unwrap();
            j.push('[');
            canon::string(&mut j, &k.to_decimal());
            j.push_str(", ");
            canon::string(&mut j, &p);
            for x in [&t, &h] {
                j.push_str(", ");
                match x { Some(v) => canon::string(&mut j, v), None => j.push_str("null") }
            }
            let m = &e.meta;
            j.push_str(", {\"decayed\": ");
            j.push_str(&m.decayed.to_string());
            j.push_str(", \"last\": ");
            j.push_str(&m.last.to_string());
            j.push_str(", \"signature\": ");
            canon::string(&mut j, &m.signature);
            j.push_str(", \"sources\": [");
            for (i, src) in m.sources.iter().enumerate() {
                if i > 0 { j.push_str(", "); }
                canon::string(&mut j, src);
            }
            j.push_str("], \"updates\": ");
            j.push_str(&m.updates.to_string());
            j.push_str("}]");
        }
        j.push_str("]}");
        let d = canon::sha256_hex(&j);
        self.digest_cache = Some((self.version, d.clone()));
        d
    }
    #[new]
    #[pyo3(signature = (capacity, actions=8))]
    fn new(capacity: usize, actions: usize) -> PyResult<Self> {
        if capacity < 1 || !(2..=64).contains(&actions) {
            return verr("Invalid memory budget");
        }
        Ok(ContextMemory { core: ContextMemoryCore::new(capacity, actions), version: bump(), digest_cache: None })
    }
    #[getter]
    fn capacity(&self) -> usize { self.core.capacity }
    #[getter]
    fn actions(&self) -> usize { self.core.actions }
    #[getter]
    fn mask(&self) -> u64 { self.core.mask }
    #[getter]
    fn clock(&self) -> u64 { self.core.clock }
    #[setter]
    fn set_clock(&mut self, v: u64) { self.version = bump(); self.core.clock = v; }
    #[getter]
    fn rows(slf: Bound<'_, Self>) -> RowsView { RowsView { mem: slf.unbind(), bank: 0 } }
    #[getter]
    fn terminal(slf: Bound<'_, Self>) -> RowsView { RowsView { mem: slf.unbind(), bank: 1 } }
    #[getter]
    fn teachers(slf: Bound<'_, Self>) -> RowsView { RowsView { mem: slf.unbind(), bank: 2 } }
    #[getter]
    fn meta(slf: Bound<'_, Self>) -> RowsView { RowsView { mem: slf.unbind(), bank: 3 } }

    fn clear(&mut self) { self.version = bump(); self.core.clear(); }
    fn copy(&self) -> Self { self.clone() }
    fn __len__(&self) -> usize { self.core.rows.len() }

    #[staticmethod]
    fn empty(py: Python<'_>) -> PyResult<Bound<'_, PyList>> { row_to_py(py, &memory::EMPTY) }

    #[staticmethod]
    #[pyo3(signature = (row, mask=255))]
    fn supported(row: &Bound<'_, PyAny>, mask: u64) -> PyResult<u64> {
        Ok(memory::supported(&row_from_py(row, mask)?, mask))
    }

    fn _ensure(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> { self.version = bump(); self.core.ensure(key_of(key)?); Ok(()) }
    fn _expire(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> { self.version = bump(); self.core.expire(&key_of(key)?); Ok(()) }

    #[pyo3(signature = (key, action, positive, weight=1, *, channel="personal"))]
    fn update(&mut self, key: &Bound<'_, PyAny>, action: i64, positive: bool, weight: i64, channel: &str) -> PyResult<()> {
        self.version = bump();
        if action < 0 || !(1..=3).contains(&weight) || (channel != "personal" && channel != "terminal") {
            return verr("Invalid evidence update");
        }
        self.core.update(key_of(key)?, action as usize, positive, weight as u32, channel == "terminal")
            .map_err(|_| PyValueError::new_err("Invalid evidence update"))
    }

    fn set_row(&mut self, key: &Bound<'_, PyAny>, row: &Bound<'_, PyAny>) -> PyResult<()> {
        self.version = bump();
        let r = row_from_py(row, self.core.mask)?;
        self.core.set_row(key_of(key)?, r);
        Ok(())
    }

    fn set_teacher(&mut self, key: &Bound<'_, PyAny>, row: &Bound<'_, PyAny>, sources: &Bound<'_, PyAny>, signature: String) -> PyResult<bool> {
        self.version = bump();
        let r = row_from_py(row, self.core.mask)?;
        let sources: Vec<String> = sources.try_iter()?.map(|x| x?.extract::<String>()).collect::<PyResult<_>>()?;
        Ok(self.core.set_teacher(key_of(key)?, r, sources, signature))
    }

    fn evidence<'py>(&mut self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        self.version = bump();
        let Ok(k) = key_of(key) else { return Ok(None) };
        match self.core.evidence(&k) {
            None => Ok(None),
            Some(e) => {
                let margin: Bound<'py, PyAny> = match e.margin {
                    Some(m) => PyList::new(py, m.iter())?.into_any(),
                    None => PyTuple::empty(py).into_any(),
                };
                Ok(Some(PyTuple::new(py, [e.good.into_pyobject(py)?.into_any(), e.bad.into_pyobject(py)?.into_any(), margin])?))
            }
        }
    }

    fn choose(&mut self, key: &Bound<'_, PyAny>, proposed: usize) -> PyResult<Option<usize>> {
        self.version = bump();
        let Ok(k) = key_of(key) else { return Ok(None) };
        Ok(self.core.choose(&k, proposed))
    }

    fn export<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.version = bump();
        let keys: Vec<Key> = self.core.rows.iter().map(|(k, _)| k).collect();
        let rows = PyList::empty(py);
        for k in keys {
            let (p, t, h) = self.core.packed(&k);
            let e = self.core.rows.get(&k).unwrap();
            let meta = meta_dict(py, &e.meta)?;
            rows.append(PyList::new(py, [
                k.to_decimal().into_pyobject(py)?.into_any(),
                p.into_pyobject(py)?.into_any(),
                t.into_pyobject(py)?.into_any(),
                h.into_pyobject(py)?.into_any(),
                meta.into_any(),
            ])?)?;
        }
        let d = PyDict::new(py);
        d.set_item("clock", self.core.clock)?;
        d.set_item("rows", rows)?;
        Ok(d)
    }

    #[pyo3(signature = (data, codec=None))]
    fn restore(&mut self, data: &Bound<'_, PyAny>, codec: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.version = bump();
        self.core.clear();
        let clock: u64 = data.get_item("clock")?.extract().map_err(|_| PyValueError::new_err("Invalid memory budget"))?;
        self.core.clock = clock;
        let rows = data.get_item("rows")?;
        let n: usize = rows.len()?;
        if n > self.core.capacity {
            return verr("Invalid memory budget");
        }
        let native_codec = match codec {
            Some(c) if !c.is_none() => Some(codec_of(c)?),
            _ => None,
        };
        for item in rows.try_iter()? {
            let item = item?;
            let k: Bound<'_, PyAny> = item.get_item(0)?;
            let ks: String = k.extract().map_err(|_| PyValueError::new_err("Invalid memory address"))?;
            let key = match Key::parse(&ks) {
                Some(key) if !ks.is_empty() && ks.bytes().all(|c| c.is_ascii_digit()) => key,
                _ => return verr("Invalid memory address"),
            };
            if key.to_decimal() != ks {
                return verr("Invalid memory address"); // native keys are canonical integers
            }
            if self.core.rows.contains(&key) {
                return verr("Invalid memory address");
            }
            if let Some(c) = &native_codec {
                if c.validate(&ks).is_none() {
                    return verr("Invalid memory address");
                }
            }
            let meta = item.get_item(4)?;
            let m = Meta {
                last: meta.get_item("last")?.extract()?,
                decayed: meta.get_item("decayed")?.extract()?,
                updates: meta.get_item("updates")?.extract()?,
                sources: meta.get_item("sources")?.extract()?,
                signature: meta.get_item("signature")?.extract()?,
            };
            if m.decayed > clock || m.sources.len() > SOURCE_LIMIT {
                return verr("Invalid memory metadata");
            }
            let unpack = |o: Bound<'_, PyAny>| -> PyResult<Option<(Row, String)>> {
                if o.is_none() { return Ok(None); }
                let s: String = o.extract()?;
                match self.core.unpack(&s) {
                    Some(r) => Ok(Some((r, s))),
                    None => verr("Invalid Boolean planes"),
                }
            };
            let (row, ps) = unpack(item.get_item(1)?)?.ok_or_else(|| PyValueError::new_err("Invalid Boolean planes"))?;
            let term = unpack(item.get_item(2)?)?;
            let teach = unpack(item.get_item(3)?)?;
            let canonical = |s: &str| s.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c));
            let packed = if canonical(&ps) && term.as_ref().map(|t| canonical(&t.1)).unwrap_or(true) && teach.as_ref().map(|t| canonical(&t.1)).unwrap_or(true) {
                Some((ps.clone(), term.as_ref().map(|t| t.1.clone()), teach.as_ref().map(|t| t.1.clone())))
            } else { None };
            self.core.rows.insert(key, memory::Entry {
                row,
                terminal: term.map(|t| t.0),
                teacher: teach.map(|t| t.0),
                meta: m,
                cache: None,
                packed,
            });
        }
        Ok(())
    }
}

fn meta_dict<'py>(py: Python<'py>, m: &Meta) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("last", m.last)?;
    d.set_item("decayed", m.decayed)?;
    d.set_item("updates", m.updates)?;
    d.set_item("sources", PyList::new(py, m.sources.iter())?)?;
    d.set_item("signature", &m.signature)?;
    Ok(d)
}

fn codec_of(c: &Bound<'_, PyAny>) -> PyResult<Codec> {
    Ok(Codec::new(c.getattr("sensors")?.extract()?, c.getattr("history")?.extract()?, c.getattr("action_bits")?.extract()?))
}

/// Read-only dict-like views: rows (0), terminal (1), teachers (2), meta (3).
#[pyclass(module = "predatore_native")]
pub struct RowsView {
    mem: Py<ContextMemory>,
    bank: u8,
}

impl RowsView {
    fn has(core: &ContextMemoryCore, bank: u8, k: &Key) -> bool {
        match core.rows.get(k) {
            None => false,
            Some(e) => match bank {
                1 => e.terminal.is_some(),
                2 => e.teacher.is_some(),
                _ => true,
            },
        }
    }
    fn keys_vec(&self, py: Python<'_>) -> Vec<String> {
        let m = self.mem.borrow(py);
        m.core.rows.iter().filter(|(k, _)| Self::has(&m.core, self.bank, k)).map(|(k, _)| k.to_decimal()).collect()
    }
    fn value<'py>(&self, py: Python<'py>, k: &Key) -> PyResult<Option<Bound<'py, PyAny>>> {
        let m = self.mem.borrow(py);
        let Some(e) = m.core.rows.get(k) else { return Ok(None) };
        Ok(match self.bank {
            0 => Some(row_to_py(py, &e.row)?.into_any()),
            1 => match &e.terminal { Some(r) => Some(row_to_py(py, r)?.into_any()), None => None },
            2 => match &e.teacher { Some(r) => Some(row_to_py(py, r)?.into_any()), None => None },
            _ => Some(meta_dict(py, &e.meta)?.into_any()),
        })
    }
}

#[pymethods]
impl RowsView {
    fn __len__(&self, py: Python<'_>) -> usize {
        if self.bank == 0 || self.bank == 3 { self.mem.borrow(py).core.rows.len() } else { self.keys_vec(py).len() }
    }
    fn __contains__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> bool {
        match key_of(key) {
            Ok(k) => Self::has(&self.mem.borrow(py).core, self.bank, &k),
            Err(_) => false,
        }
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.keys_vec(py))?.into_any().try_iter().map(|i| i.into_any())
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> { PyList::new(py, self.keys_vec(py)) }
    /// Plain dict snapshot (what the Python implementation would hold).
    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for ks in self.keys_vec(py) {
            let k = Key::parse(&ks).unwrap();
            d.set_item(ks, self.value(py, &k)?)?;
        }
        Ok(d)
    }
    fn __copy__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> { self.snapshot(py) }
    fn __deepcopy__<'py>(&self, py: Python<'py>, _memo: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> { self.snapshot(py) }
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mine = self.snapshot(py)?;
        let theirs = match other.cast::<RowsView>() { Ok(v) => v.borrow().snapshot(py)?.into_any(), Err(_) => other.clone() };
        mine.eq(theirs)
    }
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let k = key_of(key).map_err(|_| PyKeyError::new_err(key.clone().unbind()))?;
        self.value(py, &k)?.ok_or_else(|| PyKeyError::new_err(key.clone().unbind()))
    }
    #[pyo3(signature = (key, default=None))]
    fn get<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>, default: Option<Bound<'py, PyAny>>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match key_of(key) {
            Ok(k) => Ok(self.value(py, &k)?.or(default)),
            Err(_) => Ok(default),
        }
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for ks in self.keys_vec(py) {
            let k = Key::parse(&ks).unwrap();
            out.append((ks, self.value(py, &k)?))?;
        }
        Ok(out)
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for ks in self.keys_vec(py) {
            out.append(self.value(py, &Key::parse(&ks).unwrap())?)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- AssociativeMemory core

#[pyclass(module = "predatore_native")]
#[derive(Clone)]
pub struct AssociativeCoreBox {
    pub core: AssociativeCore,
    pub version: u64,
    pub digest_cache: Option<(u64, String)>,
}

fn symbol_words(obj: &Bound<'_, PyAny>, nw: usize) -> PyResult<Vec<u64>> { int_to_words(obj, nw) }

#[pymethods]
impl AssociativeCoreBox {
    #[getter]
    fn version(&self) -> u64 { self.version }
    fn __copy__(&self) -> Self { self.clone() }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self { self.clone() }
    /// sha256 of json.dumps(export, sort_keys=True) with export = {width, capacity, sensor_mask, entries}.
    fn digest(&mut self) -> String {
        if let Some((v, d)) = &self.digest_cache { if *v == self.version { return d.clone(); } }
        let mut j = String::with_capacity(self.core.entries.len() * 200 + 64);
        j.push_str("{\"capacity\": ");
        j.push_str(&self.core.capacity.to_string());
        j.push_str(", \"entries\": [");
        for (n, (k, e)) in self.core.entries.iter().enumerate() {
            if n > 0 { j.push_str(", "); }
            j.push('[');
            canon::string(&mut j, &e.decimal);
            let _ = k;
            j.push_str(", ");
            canon::string(&mut j, &bits::to_hex(&e.vector));
            j.push_str(", ");
            canon::string(&mut j, &e.anchor.to_string());
            j.push_str(&format!(", {}, {}]", e.positive, e.negative));
        }
        j.push_str("], \"sensor_mask\": ");
        canon::string(&mut j, &self.core.sensor_mask.to_string());
        j.push_str(", \"width\": ");
        j.push_str(&self.core.width.to_string());
        j.push('}');
        let d = canon::sha256_hex(&j);
        self.digest_cache = Some((self.version, d.clone()));
        d
    }
    /// symbols: dict with feature_role (list), value (list of 2), action_role, action (list), hist_role, hist (list), tie.
    #[new]
    fn new(sensors: u32, history: u32, action_bits: u32, capacity: usize, width: usize, sensor_mask: u64, symbols: &Bound<'_, PyDict>) -> PyResult<Self> {
        let nw = bits::words(width);
        let get = |name: &str| -> PyResult<Bound<'_, PyAny>> {
            symbols.get_item(name)?.ok_or_else(|| PyValueError::new_err(format!("missing symbol {name}")))
        };
        let list = |name: &str| -> PyResult<Vec<Vec<u64>>> {
            get(name)?.try_iter()?.map(|o| symbol_words(&o?, nw)).collect()
        };
        let value = list("value")?;
        let s = Symbols {
            feature_role: list("feature_role")?,
            value: [value[0].clone(), value[1].clone()],
            action_role: symbol_words(&get("action_role")?, nw)?,
            action: list("action")?,
            hist_role: symbol_words(&get("hist_role")?, nw)?,
            hist: list("hist")?,
            tie: symbol_words(&get("tie")?, nw)?,
        };
        Ok(AssociativeCoreBox { core: AssociativeCore::new(Codec::new(sensors, history, action_bits), capacity, width, sensor_mask, s), version: bump(), digest_cache: None })
    }
    fn copy(&self) -> Self { self.clone() }
    fn encode(&mut self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.version = bump();
        let k = key_of(key)?;
        words_to_int(py, &self.core.encode(&k))
    }
    fn observe(&mut self, key: &Bound<'_, PyAny>, positive: u64, negative: u64) -> PyResult<()> {
        self.version = bump();
        self.core.observe(key_of(key)?, positive, negative);
        Ok(())
    }
    #[pyo3(signature = (key, proposed, rejected=0, allowed=None))]
    fn choose(&mut self, key: &Bound<'_, PyAny>, proposed: usize, rejected: u64, allowed: Option<&Bound<'_, PyAny>>) -> PyResult<Option<usize>> {
        self.version = bump();
        let k = key_of(key)?;
        match allowed {
            None => Ok(self.core.choose(&k, proposed, rejected, &|_| true)),
            Some(a) => {
                // Native rows view: check membership natively; otherwise ask Python.
                if let Ok(view) = a.cast::<RowsView>() {
                    let v = view.borrow();
                    let py = a.py();
                    let mem = v.mem.borrow(py);
                    let bank = v.bank;
                    let core = &mem.core;
                    return Ok(self.core.choose(&k, proposed, rejected, &|x| RowsView::has(core, bank, x)));
                }
                let allowed_keys: std::collections::HashSet<Key> = self.core.bucket_tail(self.core.codec.decode(&k).0 & self.core.sensor_mask)
                    .into_iter().filter(|x| a.contains(x.to_decimal()).unwrap_or(false)).collect();
                Ok(self.core.choose(&k, proposed, rejected, &|x| allowed_keys.contains(x)))
            }
        }
    }
    fn __len__(&self) -> usize { self.core.entries.len() }
    fn entry<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyTuple>>> {
        let Ok(k) = key_of(key) else { return Ok(None) };
        match self.core.entries.get(&k) {
            None => Ok(None),
            Some(e) => Ok(Some(PyTuple::new(py, [words_to_int(py, &e.vector)?.into_bound(py), e.anchor.into_pyobject(py)?.into_any(), e.positive.into_pyobject(py)?.into_any(), e.negative.into_pyobject(py)?.into_any()])?)),
        }
    }
    fn keys(&self) -> Vec<String> { self.core.entries.iter().map(|(k, _)| k.to_decimal()).collect() }
    fn bucket_keys(&self, anchor: u64) -> Vec<String> {
        match self.core.buckets.get(&anchor) {
            None => vec![],
            Some(b) => b.iter().map(|(k, _)| k.to_decimal()).collect(),
        }
    }
    fn bucket_anchors(&self) -> Vec<u64> { self.core.buckets.keys().copied().collect() }
    fn last_lookup<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        if let Some(l) = &self.core.last_lookup {
            d.set_item("neighbors", l.neighbors)?;
            d.set_item("action", l.action)?;
            d.set_item("distances", l.distances.clone())?;
        }
        Ok(d)
    }
    fn export_entries<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let out = PyList::empty(py);
        for (k, e) in self.core.entries.iter() {
            out.append(PyList::new(py, [
                k.to_decimal().into_pyobject(py)?.into_any(),
                bits::to_hex(&e.vector).into_pyobject(py)?.into_any(),
                e.anchor.to_string().into_pyobject(py)?.into_any(),
                e.positive.into_pyobject(py)?.into_any(),
                e.negative.into_pyobject(py)?.into_any(),
            ])?)?;
        }
        Ok(out)
    }
    /// Python-compatible restore of entries; checks every code against the codec.
    fn restore_entries(&mut self, entries: &Bound<'_, PyAny>) -> PyResult<()> {
        self.version = bump();
        let n = entries.len()?;
        if n > self.core.capacity {
            return verr("Associative capacity exceeded");
        }
        let amask = if self.core.codec.actions >= 64 { u64::MAX } else { (1u64 << self.core.codec.actions) - 1 };
        let nw = bits::words(self.core.width);
        for item in entries.try_iter()? {
            let item = item?;
            let ks: String = item.get_item(0)?.extract()?;
            let Some(k) = self.core.codec.validate(&ks) else { return verr("Invalid associative entry") };
            let pos: i64 = item.get_item(3)?.extract()?;
            let neg: i64 = item.get_item(4)?.extract()?;
            if pos < 0 || neg < 0 || pos as u64 > amask || neg as u64 > amask {
                return verr("Invalid associative entry");
            }
            let vs: String = item.get_item(1)?.extract()?;
            let anchor: String = item.get_item(2)?.extract()?;
            let vec = bits::from_hex(&vs, nw).ok_or_else(|| PyValueError::new_err("Associative code inconsistent with context"))?;
            let expected_anchor = self.core.codec.decode(&k).0 & self.core.sensor_mask;
            if vec != self.core.encode(&k) || anchor.trim().parse::<u64>().ok() != Some(expected_anchor) {
                return verr("Associative code inconsistent with context");
            }
            self.core.observe(k, pos as u64, neg as u64);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- Predictions

#[pyclass(module = "predatore_native")]
#[derive(Clone)]
pub struct PredictionTable {
    pub map: Predictions,
    pub version: u64,
    pub digest_cache: Option<(u64, String)>,
}

fn pkey(obj: &Bound<'_, PyAny>) -> PyResult<(Key, usize)> {
    let s: String = obj.extract()?;
    let (a, b) = s.split_once(':').ok_or_else(|| PyKeyError::new_err(s.clone()))?;
    let k = Key::parse(a).filter(|k| k.to_decimal() == a).ok_or_else(|| PyKeyError::new_err(s.clone()))?;
    let act: usize = b.parse().map_err(|_| PyKeyError::new_err(s.clone()))?;
    if act.to_string() != b { return Err(PyKeyError::new_err(s.clone())); }
    Ok((k, act))
}

fn pkey_str(k: &(Key, usize)) -> String { format!("{}:{}", k.0.to_decimal(), k.1) }

#[pymethods]
impl PredictionTable {
    #[getter]
    fn version(&self) -> u64 { self.version }
    fn __copy__(&self) -> Self { self.clone() }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self { self.clone() }
    /// sha256 of json.dumps(list(items)).
    fn digest(&mut self) -> String {
        if let Some((v, d)) = &self.digest_cache { if *v == self.version { return d.clone(); } }
        let mut j = String::from("[");
        for (n, (k, v)) in self.map.iter().enumerate() {
            if n > 0 { j.push_str(", "); }
            j.push('[');
            canon::string(&mut j, &pkey_str(&k));
            j.push_str(&format!(", {}]", v));
        }
        j.push(']');
        let d = canon::sha256_hex(&j);
        self.digest_cache = Some((self.version, d.clone()));
        d
    }
    #[new]
    fn new() -> Self { PredictionTable { map: Predictions::new(), version: bump(), digest_cache: None } }
    fn copy(&self) -> Self { self.clone() }
    fn __len__(&self) -> usize { self.map.len() }
    fn __contains__(&self, key: &Bound<'_, PyAny>) -> bool { pkey(key).map(|k| self.map.contains(&k)).unwrap_or(false) }
    #[pyo3(signature = (key, default=None))]
    fn get(&self, key: &Bound<'_, PyAny>, default: Option<u64>) -> Option<u64> {
        match pkey(key) { Ok(k) => self.map.get(&k).copied().or(default), Err(_) => default }
    }
    fn __getitem__(&self, key: &Bound<'_, PyAny>) -> PyResult<u64> {
        let k = pkey(key)?;
        self.map.get(&k).copied().ok_or_else(|| PyKeyError::new_err(pkey_str(&k)))
    }
    fn __setitem__(&mut self, key: &Bound<'_, PyAny>, value: u64) -> PyResult<()> {
        self.version = bump();
        let k = pkey(key)?;
        if let Some(v) = self.map.get_mut(&k) { *v = value; } else { self.map.insert(k, value); }
        Ok(())
    }
    fn move_to_end(&mut self, key: &Bound<'_, PyAny>) -> PyResult<()> { self.version = bump(); let k = pkey(key)?; self.map.move_to_end(&k); Ok(()) }
    #[pyo3(signature = (last=true))]
    fn popitem(&mut self, last: bool) -> PyResult<(String, u64)> {
        self.version = bump();
        let k = if last { self.map.iter_rev().next().map(|(k, _)| k) } else { self.map.first_key() };
        let k = k.ok_or_else(|| PyKeyError::new_err("popitem(): dictionary is empty"))?;
        let v = self.map.remove(&k).unwrap();
        Ok((pkey_str(&k), v))
    }
    fn clear(&mut self) { self.version = bump(); self.map.clear(); }
    fn keys(&self) -> Vec<String> { self.map.iter().map(|(k, _)| pkey_str(&k)).collect() }
    fn values(&self) -> Vec<u64> { self.map.iter().map(|(_, v)| *v).collect() }
    fn items(&self) -> Vec<(String, u64)> { self.map.iter().map(|(k, v)| (pkey_str(&k), *v)).collect() }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyList::new(py, self.keys())?.into_any().try_iter().map(|i| i.into_any())
    }
}

// ---------------------------------------------------------------- Circuit evidence

#[pyclass(module = "predatore_native")]
#[derive(Clone)]
pub struct EvidenceTable {
    pub width: usize,
    pub map: CircuitEvidence,
    pub version: u64,
    pub digest_cache: Option<(u64, String)>,
}

#[pymethods]
impl EvidenceTable {
    #[getter]
    fn version(&self) -> u64 { self.version }
    fn __copy__(&self) -> Self { self.clone() }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self { self.clone() }
    /// sha256 of json.dumps([[list(k), list(v)] for k, v in sorted(items)]).
    fn digest(&mut self) -> String {
        if let Some((v, d)) = &self.digest_cache { if *v == self.version { return d.clone(); } }
        let mut keys: Vec<(usize, usize)> = self.map.keys().copied().collect();
        keys.sort();
        let mut j = String::from("[");
        for (n, k) in keys.iter().enumerate() {
            if n > 0 { j.push_str(", "); }
            let v = self.map.get(k).unwrap();
            j.push_str(&format!("[[{}, {}], [{}, {}, {}, {}]]", k.0, k.1, canon::decimal(&v[0]), canon::decimal(&v[1]), canon::decimal(&v[2]), canon::decimal(&v[3])));
        }
        j.push(']');
        let d = canon::sha256_hex(&j);
        self.digest_cache = Some((self.version, d.clone()));
        d
    }
    #[new]
    fn new(width: usize) -> Self { EvidenceTable { width, map: CircuitEvidence::default(), version: bump(), digest_cache: None } }
    fn copy(&self) -> Self { self.clone() }
    fn __len__(&self) -> usize { self.map.len() }
    fn clear(&mut self) { self.version = bump(); self.map.clear(); }
    fn __contains__(&self, key: (usize, usize)) -> bool { self.map.contains_key(&key) }
    fn __getitem__<'py>(&self, py: Python<'py>, key: (usize, usize)) -> PyResult<Bound<'py, PyTuple>> {
        match self.map.get(&key) {
            None => Err(PyKeyError::new_err(key)),
            Some(v) => PyTuple::new(py, v.iter().map(|x| words_to_int(py, x)).collect::<PyResult<Vec<_>>>()?),
        }
    }
    #[pyo3(signature = (key, default=None))]
    fn get<'py>(&self, py: Python<'py>, key: (usize, usize), default: Option<Bound<'py, PyAny>>) -> PyResult<Option<Bound<'py, PyAny>>> {
        match self.map.get(&key) {
            None => Ok(default),
            Some(_) => Ok(Some(self.__getitem__(py, key)?.into_any())),
        }
    }
    fn __setitem__(&mut self, key: (usize, usize), value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.version = bump();
        let nw = bits::words(self.width);
        let vals: Vec<Bound<'_, PyAny>> = value.try_iter()?.collect::<PyResult<_>>()?;
        if vals.len() != 4 { return verr("evidence needs four planes"); }
        let v: Vec<Vec<u64>> = vals.iter().map(|o| int_to_words(o, nw)).collect::<PyResult<_>>()?;
        self.map.insert(key, [v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone()]);
        Ok(())
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let mut keys: Vec<(usize, usize)> = self.map.keys().copied().collect();
        keys.sort();
        let out = PyList::empty(py);
        for k in keys {
            out.append((k, self.__getitem__(py, k)?))?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------- Circuit

fn prod(v: &[usize]) -> usize { v.iter().product() }

fn ones_shifted(nbits: usize, start: usize, nw: usize) -> Vec<u64> {
    let mut v = vec![0u64; nw];
    for i in start..start + nbits {
        v[i / 64] |= 1u64 << (i % 64);
    }
    v
}

/// tensor.axis_roll as two disjoint shifted masks.
fn roll_group(shape: &[usize], axis: usize, offset: i64, nw: usize) -> Vec<(Vec<u64>, isize)> {
    let length = shape[axis];
    let off = offset.rem_euclid(length as i64) as usize;
    let stride = prod(&shape[axis + 1..]);
    let slab = length * stride;
    let total = prod(shape);
    let mut low = vec![0u64; nw];
    let mut start = 0;
    while start < total {
        let part = ones_shifted((length - off) * stride, start, nw);
        for i in 0..nw { low[i] |= part[i]; }
        start += slab;
    }
    let all = ones_shifted(total, 0, nw);
    let high: Vec<u64> = (0..nw).map(|i| all[i] ^ low[i]).collect();
    vec![(low, (off * stride) as isize), (high, -(((length - off) * stride) as isize))]
}

fn transpose_group(shape: &[usize], order: &[usize], nw: usize) -> Vec<(Vec<u64>, isize)> {
    let target: Vec<usize> = order.iter().map(|&i| shape[i]).collect();
    let total = prod(shape);
    let mut groups: std::collections::BTreeMap<isize, Vec<u64>> = std::collections::BTreeMap::new();
    for old in 0..total {
        let mut rem = old;
        let mut coords = Vec::with_capacity(shape.len());
        for i in 0..shape.len() {
            let stride = prod(&shape[i + 1..]);
            coords.push(rem / stride);
            rem %= stride;
        }
        let new: usize = order.iter().enumerate().map(|(i, &axis)| coords[axis] * prod(&target[i + 1..])).sum();
        let e = groups.entry(new as isize - old as isize).or_insert_with(|| vec![0u64; nw]);
        e[old / 64] |= 1u64 << (old % 64);
    }
    groups.into_iter().map(|(s, m)| (m, s)).collect()
}

fn rotate_group(width: usize, shift: usize, nw: usize) -> Vec<(Vec<u64>, isize)> {
    let s = shift % width;
    if s == 0 {
        return vec![(ones_shifted(width, 0, nw), 0)];
    }
    vec![(ones_shifted(width - s, 0, nw), s as isize), (ones_shifted(s, width - s, nw), -((width - s) as isize))]
}

fn make_plan(op: &str, shape: &[usize], axis: i64, shift: i64, width: usize, nw: usize) -> PyResult<Plan> {
    let ax = axis.rem_euclid(shape.len() as i64) as usize;
    Ok(match op {
        "rotate" => Plan { base: false, xor: false, groups: vec![rotate_group(width, shift.rem_euclid(width as i64) as usize, nw)] },
        "axis_roll" => Plan { base: false, xor: false, groups: vec![roll_group(shape, ax, shift, nw)] },
        "neighbor_or" | "neighbor_xor" => Plan {
            base: true,
            xor: op == "neighbor_xor",
            groups: vec![roll_group(shape, ax, -1, nw), roll_group(shape, ax, 1, nw)],
        },
        "transpose" => {
            let mut order: Vec<usize> = (0..shape.len()).collect();
            let other = (ax + 1) % shape.len();
            order.swap(ax, other);
            Plan { base: false, xor: false, groups: vec![transpose_group(shape, &order, nw)] }
        }
        _ => return verr("Unknown tensor operator"),
    })
}

#[pyclass(module = "predatore_native")]
pub struct NativeCircuit {
    pub c: Circuit,
}

#[pymethods]
impl NativeCircuit {
    /// spec: dict built by boolean_engine.native_backend.circuit_spec(net).
    #[new]
    fn new(spec: &Bound<'_, PyDict>) -> PyResult<Self> {
        let g = |k: &str| -> PyResult<Bound<'_, PyAny>> { spec.get_item(k)?.ok_or_else(|| PyValueError::new_err(format!("spec.{k}"))) };
        let sensors: usize = g("sensors")?.extract()?;
        let action_bits: usize = g("action_bits")?.extract()?;
        let width: usize = g("width")?.extract()?;
        let shape: Vec<usize> = g("shape")?.extract()?;
        let history: u32 = g("context_history")?.extract()?;
        if sensors > 64 || action_bits > 6 || width == 0 {
            return verr("native kernel supports <= 64 sensors and <= 6 action bits");
        }
        let nw = bits::words(width);
        let nodes: Vec<(usize, usize, bool)> = g("nodes")?.extract()?;
        let rules_py = g("rules")?;
        let mut rules = Vec::with_capacity(nodes.len());
        for (i, r) in rules_py.try_iter()?.enumerate() {
            let r = r?;
            let gi = |k: &str| r.get_item(k);
            let rshape: Vec<usize> = match gi("shape") { Ok(s) => s.extract()?, Err(_) => shape.clone() };
            if prod(&rshape) != width { return verr("Invalid local shape"); }
            let lut_py: Vec<Bound<'_, PyAny>> = gi("lut")?.try_iter()?.collect::<PyResult<_>>()?;
            if lut_py.len() != 4 { return verr("Invalid hold/table"); }
            let lut: Vec<Vec<u64>> = lut_py.iter().map(|o| int_to_words(o, nw)).collect::<PyResult<_>>()?;
            let opt_str = |k: &str, d: &str| -> PyResult<String> { Ok(r.get_item(k).ok().map(|x| x.extract()).transpose()?.unwrap_or_else(|| d.to_string())) };
            let opt_int = |k: &str| -> PyResult<i64> { Ok(r.get_item(k).ok().map(|x| x.extract()).transpose()?.unwrap_or(0)) };
            let shift_a: i64 = gi("shift_a")?.extract()?;
            let shift_b: i64 = gi("shift_b")?.extract()?;
            let op_a = opt_str("operator_a", "rotate")?;
            let op_b = opt_str("operator_b", "rotate")?;
            let plan_a = make_plan(&op_a, &rshape, opt_int("axis_a")?, shift_a, width, nw)?;
            let plan_b = make_plan(&op_b, &rshape, opt_int("axis_b")?, shift_b, width, nw)?;
            rules.push(Rule {
                a: nodes[i].0,
                b: nodes[i].1,
                hold: gi("hold")?.extract()?,
                write: int_to_words(&gi("write")?, nw)?,
                lut: [lut[0].clone(), lut[1].clone(), lut[2].clone(), lut[3].clone()],
                plan_a,
                plan_b,
                delay_a: opt_int("delay_a")? as usize,
                delay_b: opt_int("delay_b")? as usize,
                shift_a: shift_a.rem_euclid(width as i64) as usize,
                shift_b: shift_b.rem_euclid(width as i64) as usize,
            });
        }
        if rules.len() != nodes.len() { return verr("Invalid vector circuit"); }
        let mut ancestors: Vec<usize> = g("ancestors")?.extract()?;
        ancestors.sort();
        let active: Vec<usize> = nodes.iter().enumerate().filter(|(_, n)| n.2).map(|(i, _)| i).collect();
        let c = Circuit {
            sensors,
            action_bits,
            actions: 1 << action_bits,
            width,
            nw,
            nodes: nodes.len(),
            rules,
            outputs: g("outputs")?.extract()?,
            output_lanes: g("output_lanes")?.extract()?,
            ticks: g("internal_ticks")?.extract()?,
            trace_depth: g("trace_depth")?.extract()?,
            memory_capacity: g("memory_capacity")?.extract()?,
            broadcast: g("sensor_encoding")?.extract::<String>()? == "broadcast",
            ancestors,
            active,
            codec: Codec::new(sensors as u32, history, action_bits as u32),
        };
        Ok(NativeCircuit { c })
    }
}

// ---------------------------------------------------------------- Geometry and episodes

#[pyclass(module = "predatore_native")]
pub struct NativeGeometry {
    pub g: Geometry,
}

#[pymethods]
impl NativeGeometry {
    #[new]
    fn new(size: i32, obstacles: Vec<(i32, i32)>) -> Self { NativeGeometry { g: Geometry::new(size, &obstacles) } }
    fn observe_from(&self, observer: (i32, i32), target: (i32, i32), collisions: u64, visibility: i64) -> u64 {
        self.g.observe_from(observer, target, collisions, visibility)
    }
}

fn flags_of(t: &Bound<'_, PyAny>, emit_events: bool) -> PyResult<Flags> {
    let v: Vec<bool> = t.extract()?;
    if v.len() != 9 { return verr("flags: 9 booleans"); }
    Ok(Flags { memory: v[0], tensor: v[1], sparse: v[2], associative: v[3], contextual: v[4], hdc: v[5], plasticity: v[6], circuit_plasticity: v[7], collect_experience: v[8], emit_events })
}

/// One brain's native parts as passed from Python.
type BrainArgs<'py> = (PyRef<'py, NativeCircuit>, Bound<'py, PyAny>, PyRefMut<'py, ContextMemory>, PyRefMut<'py, ContextMemory>, PyRefMut<'py, AssociativeCoreBox>, PyRefMut<'py, PredictionTable>, PyRefMut<'py, EvidenceTable>);

fn history_py<'py>(py: Python<'py>, h: &[(u64, usize)]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for &(o, a) in h { out.append(PyTuple::new(py, [o as usize, a])?)?; }
    Ok(out)
}

fn event_py<'py>(py: Python<'py>, e: &Event) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("context_key", e.context_key.map(|k| k.to_decimal()))?;
    let el = PyList::empty(py);
    for (p, a) in e.eligible.iter() {
        let pd = PyDict::new(py);
        pd.set_item("observation", p.observation)?;
        pd.set_item("history", history_py(py, &p.history)?)?;
        el.append(PyTuple::new(py, [pd.into_any(), a.into_pyobject(py)?.into_any()])?)?;
    }
    d.set_item("eligible_contexts", el)?;
    d.set_item("observation", e.observation)?;
    d.set_item("history", history_py(py, &e.history)?)?;
    d.set_item("previous", e.previous)?;
    d.set_item("valid", e.valid)?;
    d.set_item("action", e.action)?;
    d.set_item("feedback", e.feedback)?;
    d.set_item("next", e.next)?;
    d.set_item("terminal_success", e.terminal as u8)?;
    Ok(d)
}

fn runtime_py<'py>(py: Python<'py>, c: &Circuit, rt: &crate::circuit::Runtime) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    let nw = c.nw;
    let vectors = PyList::empty(py);
    for i in 0..c.nodes {
        vectors.append(words_to_int(py, &rt.vectors[i * nw..(i + 1) * nw])?)?;
    }
    d.set_item("vectors", vectors)?;
    d.set_item("context", rt.context)?;
    d.set_item("context_valid", rt.context_valid)?;
    d.set_item("update_count", rt.update_count)?;
    d.set_item("captured", rt.captured)?;
    d.set_item("next_observation", rt.next_observation)?;
    d.set_item("last_prediction_error", rt.last_prediction_error)?;
    d.set_item("current_context_key", rt.current_context_key.map(|k| k.to_decimal()))?;
    // Complete working state, exactly as the Python reference leaves it after an episode.
    let key_int = |k: &Key| words_to_int(py, &k.0);
    let last_tuple = |l: &(Key, usize, u64, u64, u64)| -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, [key_int(&l.0)?.into_bound(py), l.1.into_pyobject(py)?.into_any(), l.2.into_pyobject(py)?.into_any(),
                          l.3.into_pyobject(py)?.into_any(), l.4.into_pyobject(py)?.into_any()])
    };
    let trace = PyList::empty(py);
    for l in rt.trace.iter() { trace.append(last_tuple(l)?)?; }
    d.set_item("trace", trace)?;
    d.set_item("last", match &rt.last { Some(l) => Some(last_tuple(l)?), None => None })?;
    d.set_item("observable_history", history_py(py, &rt.observable_history.iter().copied().collect::<Vec<_>>())?)?;
    let payload_py = |p: &crate::circuit::Payload| -> PyResult<Bound<'py, PyDict>> {
        let pd = PyDict::new(py);
        pd.set_item("observation", p.observation)?;
        pd.set_item("history", history_py(py, &p.history)?)?;
        Ok(pd)
    };
    let ctx_trace = PyList::empty(py);
    for (p, a) in rt.context_trace.iter() {
        ctx_trace.append(PyTuple::new(py, [payload_py(p)?.into_any(), a.into_pyobject(py)?.into_any()])?)?;
    }
    d.set_item("context_trace", ctx_trace)?;
    d.set_item("last_payload", match &rt.last_payload { Some(p) => Some(payload_py(p)?), None => None })?;
    let inputs = PyList::empty(py);
    for buf in rt.input_history.iter() {
        let row = PyList::empty(py);
        for i in 0..buf.len() / nw { row.append(words_to_int(py, &buf[i * nw..(i + 1) * nw])?)?; }
        inputs.append(row)?;
    }
    d.set_item("input_history", inputs)?;
    let ctrace = PyList::empty(py);
    for (aligned, patterns) in rt.circuit_trace.iter() {
        let pl = PyList::empty(py);
        for (node, pattern, lanes) in patterns.iter() {
            pl.append(PyTuple::new(py, [node.into_pyobject(py)?.into_any(), pattern.into_pyobject(py)?.into_any(), words_to_int(py, lanes)?.into_bound(py)])?)?;
        }
        ctrace.append(PyTuple::new(py, [aligned.into_pyobject(py)?.to_owned().into_any(), pl.into_any()])?)?;
    }
    d.set_item("circuit_trace", ctrace)?;
    let counts = PyDict::new(py);
    for (i, name) in ORIGINS.iter().enumerate() { counts.set_item(name, rt.decoder_counts[i])?; }
    d.set_item("decoder_counts", counts)?;
    if let Some(l) = &rt.last_decoder {
        let ld = PyDict::new(py);
        ld.set_item("context_key", l.context_key.map(|k| k.to_decimal()))?;
        ld.set_item("context_override", l.context_override)?;
        ld.set_item("proposed", l.proposed)?;
        ld.set_item("address", l.address.to_decimal())?;
        ld.set_item("action", l.action)?;
        ld.set_item("source", ORIGINS[l.source])?;
        d.set_item("last_decoder", ld)?;
    }
    Ok(d)
}

/// Runs one whole episode natively. Returns (metrics, events, predator runtime).
#[pyfunction]
#[pyo3(signature = (brain, geometry, kind, delay, pounce, seed, visibility, prey_moves, steps, emit_events, prey=None))]
#[allow(clippy::too_many_arguments)]
pub fn run_episode<'py>(py: Python<'py>, brain: BrainArgs<'py>, geometry: PyRef<'py, NativeGeometry>, kind: &str, delay: u64, pounce: bool,
                        seed: &Bound<'py, PyAny>, visibility: i64, prey_moves: bool, steps: u64, emit_events: bool,
                        prey: Option<BrainArgs<'py>>) -> PyResult<Bound<'py, PyTuple>> {
    let seed64: u64 = {
        let m = py.eval(c"(1<<64)", None, None)?;
        seed.call_method1("__mod__", (m,))?.extract()?
    };
    let k = match kind {
        "delayed" => Kind::Delayed { delay },
        "neural" => Kind::Neural,
        _ => Kind::Plain { pounce },
    };
    let (circ, flags, mut ctx, mut short, mut assoc, mut preds, mut evid) = brain;
    let f = flags_of(&flags, emit_events)?;
    let geo: &Geometry = &geometry.g;
    let circuit: &Circuit = &circ.c;
    let (cc, sc, ac, pc, ec) = (&mut ctx.core, &mut short.core, &mut assoc.core, &mut preds.map, &mut evid.map);
    let mut prey_guard = prey;
    let prey_flags = match &prey_guard { Some(p) => Some(flags_of(&p.1, false)?), None => None };
    let prey_parts = prey_guard.as_mut().map(|(pc_, _, pctx, pshort, passoc, ppreds, pevid)| {
        (&pc_.c, &mut pctx.core, &mut pshort.core, &mut passoc.core, &mut ppreds.map, &mut pevid.map)
    });
    // The whole episode runs without the GIL: other Python threads run other episodes.
    let (result, events, rt_vals) = py.detach(move || {
        let mut b = Brain::new(circuit, f, cc, sc, ac, pc, ec);
        let result = match prey_parts {
            Some((pcirc, pctx, pshort, passoc, ppreds, pevid)) => {
                let mut pb = Brain::new(pcirc, prey_flags.unwrap(), pctx, pshort, passoc, ppreds, pevid);
                world::run_episode(&mut b, Some(&mut pb), geo, &k, seed64, visibility, prey_moves, steps)
            }
            None => world::run_episode(&mut b, None, geo, &k, seed64, visibility, prey_moves, steps),
        };
        let events = std::mem::take(&mut b.events);
        (result, events, b.rt)
    });
    let r = result.map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("capture", r.capture as u8)?;
    d.set_item("steps", r.steps)?;
    d.set_item("reward", r.reward)?;
    d.set_item("stall", r.stall as u8)?;
    d.set_item("blocked", r.blocked)?;
    d.set_item("exploration", r.exploration)?;
    d.set_item("final_distance", r.final_distance)?;
    d.set_item("obstacles", r.obstacles)?;
    d.set_item("predator", r.predator)?;
    d.set_item("prey", r.prey)?;
    let events_py = PyList::empty(py);
    for e in events.iter() {
        events_py.append(event_py(py, e)?)?;
    }
    let rt = runtime_py(py, circuit, &rt_vals)?;
    for v in [&mut ctx.version, &mut short.version, &mut assoc.version, &mut preds.version, &mut evid.version] { *v = bump(); }
    PyTuple::new(py, [d.into_any(), events_py.into_any(), rt.into_any()])
}

/// One episode description for run_batch.
type EpisodeSpec<'py> = (PyRef<'py, NativeGeometry>, String, u64, bool, Bound<'py, PyAny>, i64, bool, u64, Option<BrainArgs<'py>>);

fn seed_mod(py: Python<'_>, seed: &Bound<'_, PyAny>) -> PyResult<u64> {
    let m = py.eval(c"(1<<64)", None, None)?;
    seed.call_method1("__mod__", (m,))?.extract()
}

fn metrics_py<'py>(py: Python<'py>, r: &world::EpisodeResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("capture", r.capture as u8)?;
    d.set_item("steps", r.steps)?;
    d.set_item("reward", r.reward)?;
    d.set_item("stall", r.stall as u8)?;
    d.set_item("blocked", r.blocked)?;
    d.set_item("exploration", r.exploration)?;
    d.set_item("final_distance", r.final_distance)?;
    d.set_item("obstacles", r.obstacles)?;
    Ok(d)
}

/// Several episodes on the same brain in one call, in order, without the GIL.
/// Same as calling run_episode for each, but a thread holds the GIL only at the edges.
/// Returns (list of (metrics, events, decoder_counts), final runtime).
#[pyfunction]
#[pyo3(signature = (brain, episodes, emit_events))]
pub fn run_batch<'py>(py: Python<'py>, brain: BrainArgs<'py>, episodes: Vec<EpisodeSpec<'py>>, emit_events: bool) -> PyResult<Bound<'py, PyTuple>> {
    let (circ, flags, mut ctx, mut short, mut assoc, mut preds, mut evid) = brain;
    let f = flags_of(&flags, emit_events)?;
    let circuit: &Circuit = &circ.c;
    let mut specs = Vec::with_capacity(episodes.len());
    let mut prey_guards = Vec::with_capacity(episodes.len());
    for (geo, kind, delay, pounce, seed, visibility, prey_moves, steps, prey) in episodes.into_iter() {
        let k = match kind.as_str() { "delayed" => Kind::Delayed { delay }, "neural" => Kind::Neural, _ => Kind::Plain { pounce } };
        let pf = match &prey { Some(p) => Some(flags_of(&p.1, false)?), None => None };
        specs.push((geo, k, seed_mod(py, &seed)?, visibility, prey_moves, steps, pf));
        prey_guards.push(prey);
    }
    let (cc, sc, ac, pc, ec) = (&mut ctx.core, &mut short.core, &mut assoc.core, &mut preds.map, &mut evid.map);
    let geos: Vec<&Geometry> = specs.iter().map(|s| &s.0.g).collect();
    let mut prey_parts: Vec<Option<_>> = prey_guards.iter_mut().map(|p| p.as_mut().map(|(pc_, _, pctx, pshort, passoc, ppreds, pevid)| {
        (&pc_.c, &mut pctx.core, &mut pshort.core, &mut passoc.core, &mut ppreds.map, &mut pevid.map)
    })).collect();
    let plan: Vec<(Kind, u64, i64, bool, u64, Option<Flags>)> = specs.iter().map(|s| (match s.1 { Kind::Delayed { delay } => Kind::Delayed { delay }, Kind::Neural => Kind::Neural, Kind::Plain { pounce } => Kind::Plain { pounce } }, s.2, s.3, s.4, s.5, s.6)).collect();
    let (outcomes, rt) = py.detach(move || {
        let mut b = Brain::new(circuit, f, cc, sc, ac, pc, ec);
        let mut out = Vec::with_capacity(plan.len());
        for (i, (k, seed64, visibility, prey_moves, steps, pf)) in plan.iter().enumerate() {
            let r = match prey_parts[i].take() {
                Some((pcirc, pctx, pshort, passoc, ppreds, pevid)) => {
                    let mut pb = Brain::new(pcirc, pf.unwrap(), pctx, pshort, passoc, ppreds, pevid);
                    world::run_episode(&mut b, Some(&mut pb), geos[i], k, *seed64, *visibility, *prey_moves, *steps)
                }
                None => world::run_episode(&mut b, None, geos[i], k, *seed64, *visibility, *prey_moves, *steps),
            };
            let failed = r.is_err();
            out.push((r, std::mem::take(&mut b.events), b.rt.decoder_counts));
            if failed { break; }
        }
        (out, b.rt)
    });
    let results = PyList::empty(py);
    for (r, events, counts) in outcomes.iter() {
        let r = r.as_ref().map_err(|e| PyValueError::new_err(*e))?;
        let ev = PyList::empty(py);
        for e in events.iter() { ev.append(event_py(py, e)?)?; }
        let dc = PyDict::new(py);
        for (i, name) in ORIGINS.iter().enumerate() { dc.set_item(name, counts[i])?; }
        results.append(PyTuple::new(py, [metrics_py(py, r)?.into_any(), ev.into_any(), dc.into_any()])?)?;
    }
    let rtd = runtime_py(py, circuit, &rt)?;
    for v in [&mut ctx.version, &mut short.version, &mut assoc.version, &mut preds.version, &mut evid.version] { *v = bump(); }
    PyTuple::new(py, [results.into_any(), rtd.into_any()])
}

#[pyfunction]
fn rng_draws(seed: u64, n: usize, below: u64) -> Vec<u64> {
    let mut r = tesseract_core::Rng::new(seed);
    (0..n).map(|_| if below == 0 { r.next_u64() } else { r.below(below) }).collect()
}

#[pymodule]
fn predatore_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ContextMemory>()?;
    m.add_class::<RowsView>()?;
    m.add_class::<AssociativeCoreBox>()?;
    m.add_class::<PredictionTable>()?;
    m.add_class::<EvidenceTable>()?;
    m.add_class::<NativeCircuit>()?;
    m.add_class::<NativeGeometry>()?;
    m.add_function(wrap_pyfunction!(run_episode, m)?)?;
    m.add_function(wrap_pyfunction!(run_batch, m)?)?;
    m.add_function(wrap_pyfunction!(rng_draws, m)?)?;
    m.add("KERNEL_VERSION", "7.0.0")?;
    m.add("RNG", "xoshiro256** / SplitMix64 (tesseract-core)")?;
    Ok(())
}
