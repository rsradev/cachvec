//! Python bindings for CashVec.

use cashvec as core;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::borrow::Cow;

create_exception!(_cashvec, CashVecError, PyValueError);

fn err(e: core::CashVecError) -> PyErr {
    CashVecError::new_err(e.to_string())
}

/// A single embedding: numpy float32/float64 1-D array or a sequence of floats.
#[derive(FromPyObject)]
enum Embedding<'py> {
    F32(PyReadonlyArray1<'py, f32>),
    F64(PyReadonlyArray1<'py, f64>),
    List(Vec<f32>),
}

impl Embedding<'_> {
    fn get(&self) -> Cow<'_, [f32]> {
        match self {
            Embedding::F32(a) => match a.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(a.as_array().iter().copied().collect()),
            },
            Embedding::F64(a) => Cow::Owned(a.as_array().iter().map(|x| *x as f32).collect()),
            Embedding::List(v) => Cow::Borrowed(v),
        }
    }
}

/// A 2-D batch of embeddings (one per row).
#[derive(FromPyObject)]
enum Matrix<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

impl Matrix<'_> {
    /// Row-major data, row count, column count.
    fn get(&self) -> (Cow<'_, [f32]>, usize, usize) {
        match self {
            Matrix::F32(a) => {
                let (r, c) = (a.shape()[0], a.shape()[1]);
                // Borrow only when row-major; F-ordered or strided input is copied.
                match a.as_slice() {
                    Ok(s) if a.is_c_contiguous() => (Cow::Borrowed(s), r, c),
                    _ => (Cow::Owned(a.as_array().iter().copied().collect()), r, c),
                }
            }
            Matrix::F64(a) => {
                let (r, c) = (a.shape()[0], a.shape()[1]);
                (Cow::Owned(a.as_array().iter().map(|x| *x as f32).collect()), r, c)
            }
        }
    }
}

fn check_cols(cols: usize, dim: usize) -> PyResult<()> {
    if cols != dim {
        return Err(err(core::CashVecError::DimensionMismatch { expected: dim, actual: cols }));
    }
    Ok(())
}

/// Binary SimHash signature.
#[pyclass(frozen, eq, hash, name = "Fingerprint", module = "cashvec")]
#[derive(PartialEq, Eq, Hash)]
struct Fingerprint(core::BinaryFingerprint);

#[pymethods]
impl Fingerprint {
    /// Number of bits.
    #[getter]
    fn bits(&self) -> usize {
        self.0.bits()
    }
    /// Little-endian packed bytes (bit i is bit i % 8 of byte i // 8).
    fn to_bytes(&self) -> Vec<u8> {
        let n = self.0.bits().div_ceil(8);
        self.0.words().iter().flat_map(|w| w.to_le_bytes()).take(n).collect()
    }
    fn hamming_distance(&self, other: &Fingerprint) -> PyResult<u32> {
        self.0.hamming_distance(&other.0).map_err(err)
    }
    /// Cosine similarity estimated from the Hamming distance (approximate).
    fn estimated_cosine(&self, other: &Fingerprint) -> PyResult<f32> {
        self.0.estimated_cosine(&other.0).map_err(err)
    }
    fn is_similar_to(&self, other: &Fingerprint, min_cosine: f32) -> PyResult<bool> {
        self.0.is_similar_to(&other.0, min_cosine).map_err(err)
    }
    fn __str__(&self) -> String {
        self.0.to_string()
    }
    fn __repr__(&self) -> String {
        format!("{:?}", self.0)
    }
}

/// Dimension-agnostic SimHash.
#[pyclass(frozen, name = "VectorHasher", module = "cashvec")]
struct VectorHasher(core::VectorHasher);

#[pymethods]
impl VectorHasher {
    #[new]
    #[pyo3(signature = (bits, seed=None))]
    fn new(bits: usize, seed: Option<u64>) -> Self {
        Self(match seed {
            Some(s) => core::VectorHasher::with_seed(bits, s),
            None => core::VectorHasher::new(bits),
        })
    }
    #[getter]
    fn bits(&self) -> usize {
        self.0.bits()
    }
    fn hash(&self, py: Python<'_>, v: Embedding<'_>) -> PyResult<Fingerprint> {
        let v = v.get();
        let h = &self.0;
        py.detach(|| h.hash(&v)).map(Fingerprint).map_err(err)
    }
}

/// Fixed-dimension SimHash with precomputed hyperplanes (faster per call).
#[pyclass(frozen, name = "SimHasher", module = "cashvec")]
struct SimHasher(core::SimHasher);

#[pymethods]
impl SimHasher {
    #[new]
    #[pyo3(signature = (dim, bits, seed=None))]
    fn new(dim: usize, bits: usize, seed: Option<u64>) -> PyResult<Self> {
        let r = match seed {
            Some(s) => core::SimHasher::with_seed(dim, bits, s),
            None => core::SimHasher::new(dim, bits),
        };
        r.map(Self).map_err(err)
    }
    #[getter]
    fn dim(&self) -> usize {
        self.0.dim()
    }
    #[getter]
    fn bits(&self) -> usize {
        self.0.bits()
    }
    fn hash(&self, py: Python<'_>, v: Embedding<'_>) -> PyResult<Fingerprint> {
        let v = v.get();
        let h = &self.0;
        py.detach(|| h.hash(&v)).map(Fingerprint).map_err(err)
    }
    /// Hash every row of a 2-D array.
    fn hash_many(&self, py: Python<'_>, m: Matrix<'_>) -> PyResult<Vec<Fingerprint>> {
        let (data, rows, cols) = m.get();
        check_cols(cols, self.0.dim())?;
        let h = &self.0;
        py.detach(|| {
            (0..rows)
                .map(|i| h.hash(&data[i * cols..(i + 1) * cols]).map(Fingerprint))
                .collect::<core::Result<Vec<_>>>()
        })
        .map_err(err)
    }
}

/// Hex string of the exact 128-bit fingerprint of the vector data.
#[pyfunction]
fn exact_fingerprint(v: Embedding<'_>) -> PyResult<String> {
    core::exact_fingerprint(&v.get()).map(|f| f.to_string()).map_err(err)
}

#[pyfunction]
fn cosine_similarity(a: Embedding<'_>, b: Embedding<'_>) -> PyResult<f32> {
    core::cosine_similarity(&a.get(), &b.get()).map_err(err)
}

fn make_config(
    dim: usize,
    num_tables: usize,
    bits_per_table: usize,
    seed: Option<u64>,
) -> core::IndexConfig {
    let mut c = core::IndexConfig::new(dim)
        .with_tables(num_tables)
        .with_bits_per_table(bits_per_table);
    if let Some(s) = seed {
        c = c.with_seed(s);
    }
    c
}

/// In-memory LSH index with exact cosine verification.
#[pyclass(name = "VectorIndex", module = "cashvec")]
struct VectorIndex(core::VectorIndex);

#[pymethods]
impl VectorIndex {
    #[new]
    #[pyo3(signature = (dim, num_tables=16, bits_per_table=12, seed=None))]
    fn new(dim: usize, num_tables: usize, bits_per_table: usize, seed: Option<u64>) -> PyResult<Self> {
        core::VectorIndex::new(make_config(dim, num_tables, bits_per_table, seed))
            .map(Self)
            .map_err(err)
    }
    #[getter]
    fn dim(&self) -> usize {
        self.0.dim()
    }
    fn __len__(&self) -> usize {
        self.0.len()
    }
    fn __contains__(&self, id: &str) -> bool {
        self.0.contains(id)
    }
    fn insert(&mut self, id: String, v: Embedding<'_>) -> PyResult<()> {
        self.0.insert(id, &v.get()).map_err(err)
    }
    /// Insert each row of `matrix` under the matching ID. Stops at the first
    /// error; rows before it stay inserted.
    fn insert_many(&mut self, ids: Vec<String>, matrix: Matrix<'_>) -> PyResult<()> {
        let (data, rows, cols) = matrix.get();
        if ids.len() != rows {
            return Err(PyValueError::new_err(format!(
                "got {} ids for {rows} rows",
                ids.len()
            )));
        }
        check_cols(cols, self.0.dim())?;
        for (i, id) in ids.into_iter().enumerate() {
            self.0.insert(id, &data[i * cols..(i + 1) * cols]).map_err(err)?;
        }
        Ok(())
    }
    fn remove(&mut self, id: &str) -> bool {
        self.0.remove(id)
    }
    /// The stored vector as a float32 numpy array, or None.
    fn get<'py>(&self, py: Python<'py>, id: &str) -> Option<Bound<'py, PyArray1<f32>>> {
        self.0.get(id).map(|v| PyArray1::from_slice(py, v))
    }
    fn fingerprint_of(&self, id: &str) -> Option<String> {
        self.0.fingerprint_of(id).map(|f| f.to_string())
    }
    fn find_exact(&self, v: Embedding<'_>) -> PyResult<Vec<String>> {
        Ok(self.0.find_exact(&v.get()).map_err(err)?.into_iter().map(String::from).collect())
    }
    /// Unverified LSH candidate IDs.
    fn candidates(&self, v: Embedding<'_>) -> PyResult<Vec<String>> {
        Ok(self.0.candidates(&v.get()).map_err(err)?.into_iter().map(String::from).collect())
    }
    /// Top-k `(id, cosine)` pairs with cosine >= threshold, best first.
    #[pyo3(signature = (query, k=10, threshold=0.9))]
    fn search(&self, py: Python<'_>, query: Embedding<'_>, k: usize, threshold: f32) -> PyResult<Vec<(String, f32)>> {
        let q = query.get();
        let idx = &self.0;
        py.detach(|| idx.search(&q, k, threshold)).map(pairs).map_err(err)
    }
    #[pyo3(signature = (query, k=10, threshold=0.9))]
    fn search_brute_force(&self, py: Python<'_>, query: Embedding<'_>, k: usize, threshold: f32) -> PyResult<Vec<(String, f32)>> {
        let q = query.get();
        let idx = &self.0;
        py.detach(|| idx.search_brute_force(&q, k, threshold)).map(pairs).map_err(err)
    }
    /// `search` for every row of a 2-D array, GIL released.
    #[pyo3(signature = (queries, k=10, threshold=0.9))]
    fn search_many(&self, py: Python<'_>, queries: Matrix<'_>, k: usize, threshold: f32) -> PyResult<Vec<Vec<(String, f32)>>> {
        let (data, rows, cols) = queries.get();
        check_cols(cols, self.0.dim())?;
        let idx = &self.0;
        py.detach(|| {
            (0..rows)
                .map(|i| idx.search(&data[i * cols..(i + 1) * cols], k, threshold).map(pairs))
                .collect::<core::Result<Vec<_>>>()
        })
        .map_err(err)
    }
    /// Estimated heap usage in bytes.
    fn memory_usage(&self) -> usize {
        self.0.memory_usage()
    }
}

fn pairs(r: Vec<core::SearchResult>) -> Vec<(String, f32)> {
    r.into_iter().map(|x| (x.id, x.score)).collect()
}

/// A duplicate candidate: `id`, cosine `score`, and whether it is `exact`.
#[pyclass(frozen, get_all, name = "DuplicateMatch", module = "cashvec")]
struct DuplicateMatch {
    id: String,
    score: f32,
    exact: bool,
}

#[pymethods]
impl DuplicateMatch {
    fn __repr__(&self) -> String {
        format!("DuplicateMatch(id={:?}, score={:.6}, exact={})", self.id, self.score, if self.exact { "True" } else { "False" })
    }
}

fn matches(m: Vec<core::DuplicateMatch>) -> Vec<DuplicateMatch> {
    m.into_iter().map(|x| DuplicateMatch { id: x.id, score: x.score, exact: x.exact }).collect()
}

/// Semantic deduplication. Reports candidates; never deletes.
#[pyclass(name = "DedupEngine", module = "cashvec")]
struct DedupEngine(core::DedupEngine);

#[pymethods]
impl DedupEngine {
    #[new]
    #[pyo3(signature = (dim, threshold=0.95, num_tables=16, bits_per_table=12, seed=None))]
    fn new(dim: usize, threshold: f32, num_tables: usize, bits_per_table: usize, seed: Option<u64>) -> PyResult<Self> {
        let cfg = core::DedupConfig::new(dim)
            .with_index(make_config(dim, num_tables, bits_per_table, seed))
            .with_threshold(threshold);
        core::DedupEngine::new(cfg).map(Self).map_err(err)
    }
    #[getter]
    fn threshold(&self) -> f32 {
        self.0.threshold()
    }
    fn __len__(&self) -> usize {
        self.0.len()
    }
    fn __contains__(&self, id: &str) -> bool {
        self.0.contains(id)
    }
    fn insert(&mut self, id: String, v: Embedding<'_>) -> PyResult<()> {
        self.0.insert(id, &v.get()).map_err(err)
    }
    /// Report duplicates at the default threshold, then index the embedding.
    fn check_and_insert(&mut self, id: String, v: Embedding<'_>) -> PyResult<Vec<DuplicateMatch>> {
        self.0.check_and_insert(id, &v.get()).map(matches).map_err(err)
    }
    fn find_exact(&self, v: Embedding<'_>) -> PyResult<Vec<String>> {
        self.0.find_exact(&v.get()).map_err(err)
    }
    fn is_exact_duplicate(&self, v: Embedding<'_>) -> PyResult<bool> {
        self.0.is_exact_duplicate(&v.get()).map_err(err)
    }
    /// Matches with cosine >= `threshold` (default: the engine's threshold).
    #[pyo3(signature = (v, threshold=None))]
    fn find_similar(&self, py: Python<'_>, v: Embedding<'_>, threshold: Option<f32>) -> PyResult<Vec<DuplicateMatch>> {
        let q = v.get();
        let t = threshold.unwrap_or_else(|| self.0.threshold());
        let e = &self.0;
        py.detach(|| e.find_similar(&q, t)).map(matches).map_err(err)
    }
    fn remove(&mut self, id: &str) -> bool {
        self.0.remove(id)
    }
    fn get<'py>(&self, py: Python<'py>, id: &str) -> Option<Bound<'py, PyArray1<f32>>> {
        self.0.get(id).map(|v| PyArray1::from_slice(py, v))
    }
}

#[pymodule]
fn _cashvec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("CashVecError", m.py().get_type::<CashVecError>())?;
    m.add_class::<Fingerprint>()?;
    m.add_class::<VectorHasher>()?;
    m.add_class::<SimHasher>()?;
    m.add_class::<VectorIndex>()?;
    m.add_class::<DuplicateMatch>()?;
    m.add_class::<DedupEngine>()?;
    m.add_function(wrap_pyfunction!(exact_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_similarity, m)?)?;
    Ok(())
}
