use std::{ptr::{self, NonNull}, io, ffi::{CStr, CString}, path::{Path, PathBuf}, fs::File, mem, ffi, intrinsics::copy_nonoverlapping, error, borrow, fmt};
use libc;
use bitflags::bitflags;
use libvmaf_sys::*;
use os_str_bytes::OsStrBytes;

pub mod y4m;

/// Represents the different levels of logging for VMAF.
///
/// Each log level includes all the levels above it. For instance, `Warning` includes warnings, errors, and none.
/// The log levels are used to control the verbosity of log output.
///
/// # Variants
///
/// - `None`: No log output.
/// - `Error`: Logs errors that occur during the VMAF computation process.
/// - `Warning`: Logs both errors and warnings that might not prevent computation but are anomalous or potentially problematic.
/// - `Info`: Includes error, warning, and informational messages that provide insights into the ongoing computation process.
/// - `Debug`: Provides detailed log output for debugging purposes, including all error, warning, and informational messages.
///
/// # Example
///
/// ```rust
/// use vmaf::*;
///
/// let log_level = LogLevel::Debug;
/// let score = collect_score("videos/foo.y4m", "videos/bar.y4m", CollectScoreOpts { log_level, ..Default::default() })?;
///
/// print!("Vmaf score: {}", score);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum LogLevel {
    None,
    Error,
    Warning,
    Info,
    Debug,
}

impl Into<VmafLogLevel> for LogLevel {
    /// This function converts `LogLevel` into `libvmaf_sys::VmafLogLevel`.
    fn into(self) -> VmafLogLevel {
        match self {
            Self::None => VmafLogLevel::VMAF_LOG_LEVEL_NONE,
            Self::Error => VmafLogLevel::VMAF_LOG_LEVEL_ERROR,
            Self::Warning => VmafLogLevel::VMAF_LOG_LEVEL_WARNING,
            Self::Info => VmafLogLevel::VMAF_LOG_LEVEL_INFO,
            Self::Debug => VmafLogLevel::VMAF_LOG_LEVEL_DEBUG,
        }
    }
}

impl Default for LogLevel {
    /// This function provides a default value for `LogLevel`, which is `None`.
    ///
    /// # Example
    /// ```rust
    /// use vmaf::*;
    /// assert_eq!(LogLevel::default(), LogLevel::None);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn default() -> Self {
        Self::None
    }
}

/// `OutputFormat` represents the various formats for outputting the VMAF computation results.
/// This enum is used to save the computation results in different formats such as XML, JSON, CSV, and SUB.
///
/// # Variants
///
/// - `None`: No output.
/// - `Xml`: Output in XML format.
/// - `Json`: Output in JSON format.
/// - `Csv`: Output in CSV format.
/// - `Sub`: Output in SUB format.
///
/// # Example
///
/// ```rust
/// use vmaf::*;
/// use std::path::Path;
///
/// let output_format = OutputFormat::Json;
/// # let _ = std::fs::remove_file(output_format.default_path());
/// # assert!(!Path::new("vmaf_output.json").exists());
/// let score = collect_score("videos/foo.y4m", "videos/bar.y4m", CollectScoreOpts {
///     output_format: Some(output_format),
///     ..Default::default()
/// })?;
///
/// assert!(Path::new("vmaf_output.json").exists());
/// println!("Score: {}", score);
/// # std::fs::remove_file(output_format.default_path())?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Override default output path
/// ```rust
/// use vmaf::*;
/// use std::path::{Path, PathBuf};
///
/// let output_format = OutputFormat::Json;
/// let output_path = Path::new("custom_name.json");
/// # let _ = std::fs::remove_file(output_path);
/// # assert!(!output_path.exists());
/// let score = collect_score("videos/foo.y4m", "videos/bar.y4m", CollectScoreOpts {
///     output_format: Some(output_format),
///     output_path: Some(PathBuf::from(output_path)),
///     ..Default::default()
/// })?;
///
/// assert!(output_path.exists());
/// println!("Score: {}", score);
/// # std::fs::remove_file(output_path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum OutputFormat {
    None,
    Xml,
    Json,
    Csv,
    Sub,
}

impl OutputFormat {
    fn from_path(path: impl AsRef<Path>) -> Self {
        let path = path.as_ref();
        match path.extension() {
            None => Self::default(),
            Some(os_str) => match os_str.to_str() {
                None => Self::default(),
                Some("xml") => Self::Xml,
                Some("json") => Self::Json,
                Some("csv") => Self::Csv,
                Some("sub") => Self::Sub,
                Some(_) => Self::default(),
            },
        }
    }

    fn extension(&self) -> String {
        match self {
            Self::None => String::from("dat"),
            Self::Xml => String::from("xml"),
            Self::Json => String::from("json"),
            Self::Csv => String::from("csv"),
            Self::Sub => String::from("sub"),
        }
    }

    /// Generates a default output path based on the current output format.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use vmaf::*;
    /// use std::path::Path;
    ///
    /// let format = OutputFormat::Csv;
    /// assert_eq!(format.default_path(), Path::new("vmaf_output.csv"));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn default_path(&self) -> PathBuf {
        PathBuf::from(&format!("vmaf_output.{}", self.extension()))
    }
}

impl Into<VmafOutputFormat> for OutputFormat {
    /// This function converts `OutputFormat` into `libvmaf_sys::VmafOutputFormat`.
    fn into(self) -> VmafOutputFormat {
        match self {
            Self::None => VmafOutputFormat::VMAF_OUTPUT_FORMAT_NONE,
            Self::Xml => VmafOutputFormat::VMAF_OUTPUT_FORMAT_XML,
            Self::Json => VmafOutputFormat::VMAF_OUTPUT_FORMAT_JSON,
            Self::Csv => VmafOutputFormat::VMAF_OUTPUT_FORMAT_CSV,
            Self::Sub => VmafOutputFormat::VMAF_OUTPUT_FORMAT_SUB,
        }
    }
}

impl Default for OutputFormat {
    /// This function provides a default value for `OutputFormat`, which is `Xml`.
    ///
    /// # Example
    /// ```rust
    /// use vmaf::*;
    ///
    /// assert_eq!(OutputFormat::default(), OutputFormat::Xml);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn default() -> Self {
        Self::Xml
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PoolingMethod {
    Unknown,
    Min,
    Max,
    Mean,
    HarmonicMean,
    Nb
}

impl Into<VmafPoolingMethod> for PoolingMethod {
    fn into(self) -> VmafPoolingMethod {
        match self {
            Self::Unknown => VmafPoolingMethod::VMAF_POOL_METHOD_UNKNOWN,
            Self::Min => VmafPoolingMethod::VMAF_POOL_METHOD_MIN,
            Self::Max => VmafPoolingMethod::VMAF_POOL_METHOD_MAX,
            Self::Mean => VmafPoolingMethod::VMAF_POOL_METHOD_MEAN,
            Self::HarmonicMean => VmafPoolingMethod::VMAF_POOL_METHOD_HARMONIC_MEAN,
            Self::Nb => VmafPoolingMethod::VMAF_POOL_METHOD_NB
        }
    }
}

impl Default for PoolingMethod {
    fn default() -> Self {
        Self::HarmonicMean
    }
}

bitflags! {
    #[derive(Debug)]
    pub struct ModelFlags: u64 {
        const DEFAULT = VmafModelFlags::VMAF_MODEL_FLAGS_DEFAULT as u64;
        const DISABLE_CLIP = VmafModelFlags::VMAF_MODEL_FLAG_DISABLE_CLIP as u64;
        const ENABLE_TRANSFORM = VmafModelFlags::VMAF_MODEL_FLAG_ENABLE_TRANSFORM as u64;
        const DISABLE_TRANSFORM = VmafModelFlags::VMAF_MODEL_FLAG_DISABLE_TRANSFORM as u64;
    }
}

impl Default for ModelFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Debug)]
pub struct BootstrappedScore {
    pub bagging_score: f64,
    pub stddev: f64,
    pub ci_p95_lo: f64,
    pub ci_p95_hi: f64,
}

#[derive(Debug)]
pub struct FeatureDictionary {
    ptr: *mut VmafFeatureDictionary,
}

impl FeatureDictionary {
    pub fn new() -> Self {
        Self {
            ptr: ptr::null_mut(),
        }
    }

    pub fn set(&mut self, key: impl AsRef<str>, value: impl AsRef<str>) -> Result<(), Error> {
        let key = CString::new(key.as_ref()).map_err(|_| Error::InvalidArgument)?;
        let value = CString::new(value.as_ref()).map_err(|_| Error::InvalidArgument)?;
        let r = unsafe { vmaf_feature_dictionary_set(&mut self.ptr, (&key).as_ptr(), (&value).as_ptr()) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        assert_ne!(self.ptr, ptr::null_mut());

        Ok(())
    }
}

impl Drop for FeatureDictionary {
    fn drop(&mut self) {
        unsafe { vmaf_feature_dictionary_free(&mut self.ptr) };
    }
}


#[derive(Debug)]
pub struct ModelConfig {
    _name: Option<Box<CString>>, // have CString in heap for reference from pointer
    resource: VmafModelConfig,
}

impl ModelConfig {
    pub fn builder() -> ModelConfigBuilder {
        ModelConfigBuilder::new()
    }
}

#[derive(Debug)]
pub struct ModelConfigBuilder {
    name: Option<String>,
    flags: ModelFlags,
}

impl ModelConfigBuilder {
    fn new() -> Self {
        Self {
            name: None,
            flags: ModelFlags::DEFAULT,
        }
    }

    pub fn name(mut self, name: Option<String>) -> Self {
        self.name = name;
        self
    }

    pub fn flags(mut self, flags: ModelFlags) -> Self {
        self.flags = flags;
        self
    }

    pub fn build(self) -> Result<ModelConfig, Error> {
        let (name, name_ptr) = if let Some(name) = self.name {
            let name = Box::new(CString::new(name).map_err(|_| Error::InvalidArgument)?);
            let name_ptr = (&*name).as_ptr();
            (Some(name), name_ptr)
        } else {
            (None, ptr::null())
        };
        Ok(ModelConfig {
            _name: name,
            resource: VmafModelConfig {
                name: name_ptr,
                flags: self.flags.bits(),
            }
        })
    }
}

#[derive(Debug)]
pub struct Configuration {
    resource: VmafConfiguration,
}

impl Configuration {
    pub fn builder() -> ConfigurationBuilder {
        ConfigurationBuilder::new()
    }
}

#[derive(Debug)]
pub struct ConfigurationBuilder {
    log_level: LogLevel,
    n_threads: usize,
    n_subsample: usize,
    cpumask: u64,
}

impl ConfigurationBuilder {
    fn new() -> Self {
        Self {
            log_level: LogLevel::None,
            n_threads: 0, // 0 not use thread pool
            n_subsample: 0,
            cpumask: 0,
        }
    }

    pub fn log_level(mut self, log_level: LogLevel) -> Self {
        self.log_level = log_level;
        self
    }

    pub fn n_threads(mut self, n_threads: usize) -> Self {
        self.n_threads = n_threads;
        self
    }

    pub fn n_subsample(mut self, n_subsample: usize) -> Self {
        self.n_subsample = n_subsample;
        self
    }

    pub fn cpumask(mut self, cpumask: u64) -> Self {
        self.cpumask = cpumask;
        self
    }

    pub fn build(self) -> Result<Configuration, Error> {
        Ok(Configuration {
            resource: VmafConfiguration {
                log_level: self.log_level.into(),
                n_threads: to_c_uint(self.n_threads)?,
                n_subsample: to_c_uint(self.n_subsample)?,
                cpumask: self.cpumask,
            }
        })
    }
}

#[derive(Debug)]
pub enum Error {
    OutOfMemory, // ENOMEM
    InvalidArgument, // EINVAL
    UnknownErrorNumber(i32), // Other libc::EXXXXX
    InvalidErrorNumber,
    IoError(io::Error),
    FailedInCustomPictureStreamBuilder(Box<dyn error::Error>),
    InvalidVideoFrameFormat(String),
}

impl Error {
    fn from_sys(r: libc::c_int) -> Self {
        match -r {
            libc::ENOMEM => Self::OutOfMemory,
            libc::EINVAL => Self::InvalidArgument,
            r => match r.try_into() {
                Ok(r) => Self::UnknownErrorNumber(r),
                Err(_) => Self::InvalidErrorNumber,
            },
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error { }

pub fn version() -> &'static str {
    let version_cstr = unsafe { CStr::from_ptr(vmaf_version()) };
    version_cstr.to_str().unwrap()
}

#[derive(Debug)]
pub struct CollectScoreOpts {
    pub name: Option<String>,
    pub flags: ModelFlags,
    pub model_version: ModelVersion,
    pub log_level: LogLevel,
    pub n_threads: usize,
    pub n_subsample: usize,
    pub cpumask: u64,
    pub pool_method: PoolingMethod,
    pub output_path: Option<PathBuf>,
    pub output_format: Option<OutputFormat>,
}

impl Default for CollectScoreOpts {
    fn default() -> Self {
        let model_default_opts = ModelCollectScoreOpts::default();
        Self {
            name: None,
            flags: ModelFlags::default(),
            model_version: ModelVersion::default(),
            log_level: model_default_opts.log_level,
            n_threads: model_default_opts.n_threads,
            n_subsample: model_default_opts.n_subsample,
            cpumask: model_default_opts.cpumask,
            pool_method: model_default_opts.pool_method,
            output_path: model_default_opts.output_path,
            output_format: model_default_opts.output_format,
        }
    }
}

pub fn collect_score(ref_path: impl TryInto<AutoPictureStream, Error=Error>, dist_path: impl TryInto<AutoPictureStream, Error=Error>, opts: CollectScoreOpts) -> Result<f64, Error> {
    let ref_stream = ref_path.try_into()?;
    let dist_stream = dist_path.try_into()?;

    collect_score_from_stream_pair(ref_stream, dist_stream, opts)
}

pub fn collect_score_from_stream_pair(ref_stream: impl PictureStream, dist_stream: impl PictureStream, opts: CollectScoreOpts) -> Result<f64, Error> {
    let model_cfg = ModelConfig::builder()
        .name(opts.name)
        .flags(opts.flags)
        .build()?;

    let model = Model::new(&model_cfg, &opts.model_version)?;
    model.collect_score_from_stream_pair(ref_stream, dist_stream, ModelCollectScoreOpts {
        log_level: opts.log_level,
        n_threads: opts.n_threads,
        n_subsample: opts.n_subsample,
        cpumask: opts.cpumask,
        pool_method: opts.pool_method,
        output_path: opts.output_path,
        output_format: opts.output_format,
    })
}

pub fn collect_bootstrapped_score(ref_path: impl TryInto<AutoPictureStream, Error=Error>, dist_path: impl TryInto<AutoPictureStream, Error=Error>, opts: CollectScoreOpts) -> Result<BootstrappedScore, Error> {
    let ref_stream = ref_path.try_into()?;
    let dist_stream = dist_path.try_into()?;

    collect_bootstrapped_score_from_stream_pair(ref_stream, dist_stream, opts)
}

pub fn collect_bootstrapped_score_from_stream_pair(ref_stream: impl PictureStream, dist_stream: impl PictureStream, opts: CollectScoreOpts) -> Result<BootstrappedScore, Error> {
    let model_cfg = ModelConfig::builder()
        .name(opts.name)
        .flags(opts.flags)
        .build()?;

    let model = Model::new(&model_cfg, &opts.model_version)?;
    model.collect_bootstrapped_score_from_stream_pair(ref_stream, dist_stream, ModelCollectScoreOpts {
        log_level: opts.log_level,
        n_threads: opts.n_threads,
        n_subsample: opts.n_subsample,
        cpumask: opts.cpumask,
        pool_method: opts.pool_method,
        output_path: opts.output_path,
        output_format: opts.output_format,
    })
}

#[derive(Debug)]
pub struct Context {
    ptr: NonNull<VmafContext>,
}

impl Context {
    /**
     * Allocate and open a VMAF instance.
     *
     * @param vmaf The VMAF instance to open.
     *             To be used in further libvmaf api calls.
     *             $vmaf will be set to the allocated context.
     *             Context should be cleaned up with `vmaf_close()` when finished.
     *
     * @param cfg  Configuration parameters.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn new(cfg: Configuration) -> Result<Self, Error> {
        let mut ptr: *mut VmafContext = ptr::null_mut();
        let r = unsafe { vmaf_init(&mut ptr, cfg.resource) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }

        let ptr = NonNull::new(ptr).unwrap();
        Ok(Self { ptr })
    }

    /**
     * Register feature extractors required by a specific `VmafModel`.
     * This may be called multiple times using different models.
     * In this case, the registered feature extractors will form a set, and any
     * features required by multiple models will only be extracted once.
     *
     * @param vmaf  The VMAF context allocated with `vmaf_init()`.
     *
     * @param model Opaque model context.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn use_simple_model_feature(&mut self, model: &Model) -> Result<(), Error> {
        let model_ptr = model.ptr;
        let r = unsafe { vmaf_use_features_from_model(self.ptr.as_ptr(), model_ptr.as_ptr()) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(())
    }

    /**
     * Register feature extractors required by a specific `VmafModelCollection`
     * Like `vmaf_use_features_from_model()`, this function may be called
     * multiple times using different model collections.
     *
     * @param vmaf             The VMAF context allocated with `vmaf_init()`.
     *
     * @param model_collection Opaque model collection context.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn use_collection_model_feature(&mut self, model: &Model) -> Result<(), Error> {
        let Some(model_collection_ptr) = model.collection_ptr else {
            return Err(Error::InvalidArgument);
        };
        let r = unsafe { vmaf_use_features_from_model_collection(self.ptr.as_ptr(), model_collection_ptr.as_ptr()) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(())
    }

    /**
     * Register specific feature extractor.
     * Useful when a specific/additional feature is required, usually one which
     * is not already provided by a model via `vmaf_use_features_from_model()`.
     * This may be called multiple times. `VmafContext` will take ownership of the
     * `VmafFeatureDictionary` (`opts_dict`). Use `vmaf_feature_dictionary_free()`
     * only in the case of failure.
     *
     * @param vmaf         The VMAF context allocated with `vmaf_init()`.
     *
     * @param feature_name Name of feature.
     *
     * @param opts_dict    Feature extractor options set via
     *                     `vmaf_feature_dictionary_set()`. If no special options
     *                     are required this parameter can be set to NULL.
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn use_features(&mut self, feature_name: impl AsRef<str>, dict: &FeatureDictionary) -> Result<(), Error> {
        let feature_name = CString::new(feature_name.as_ref()).map_err(|_| Error::InvalidArgument)?;
        let r = unsafe { vmaf_use_feature(self.ptr.as_ptr(), (&feature_name).as_ptr(), dict.ptr) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(())
    }

    /**
     * Import an external feature score.
     * Useful when pre-computed feature scores are available.
     * Also useful in the case where there is no libvmaf feature extractor
     * implementation for a required feature.
     *
     * @param vmaf         The VMAF context allocated with `vmaf_init()`.
     *
     * @param feature_name Name of feature.
     *
     * @param value        Score.
     *
     * @param index        Picture index.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn import_feature(&mut self, feature_name: impl AsRef<str>, value: f64, index: usize) -> Result<(), Error> {
        let feature_name = CString::new(feature_name.as_ref()).map_err(|_| Error::InvalidArgument)?;
        let r = unsafe {
            vmaf_import_feature_score(
                self.ptr.as_ptr(),
                (&feature_name).as_ptr(),
                value,
                index.try_into().map_err(|_| Error::InvalidArgument)?,
            )
        };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(())
    }

    /**
     * Read a pair of pictures and queue them for eventual feature extraction.
     * This should be called after feature extractors are registered via
     * `vmaf_use_features_from_model()` and/or `vmaf_use_feature()`.
     * `VmafContext` will take ownership of both `VmafPicture`s (`ref` and `dist`)
     * and `vmaf_picture_unref()`.
     *
     * When you're done reading pictures call this function again with both `ref`
     * and `dist` set to NULL to flush all feature extractors.
     *
     * @param vmaf  The VMAF context allocated with `vmaf_init()`.
     *
     * @param ref   Reference picture.
     *
     * @param dist  Distorted picture.
     *
     * @param index Picture index.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn read_pictures(&mut self, ref_pic: Picture, dist_pic: Picture, index: usize) -> Result<(), Error> {
        let mut ref_ptr = ref_pic.consume()?;
        let mut dist_ptr = dist_pic.consume()?;
        let r = unsafe { vmaf_read_pictures(self.ptr.as_ptr(), ref_ptr.as_mut(), dist_ptr.as_mut(), to_c_uint(index)?) };
        if r != 0 {
            unsafe {
                vmaf_picture_unref(ref_ptr.as_mut());
                vmaf_picture_unref(dist_ptr.as_mut());
            };
            return Err(Error::from_sys(r));
        }
        Ok(())
    }

    pub fn wait_for_all_pictures_flushed(&mut self) -> Result<(), Error> {
        let r = unsafe { vmaf_read_pictures(self.ptr.as_ptr(), ptr::null_mut(), ptr::null_mut(), 0) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(())
    }

    /**
     * Predict VMAF score at specific index.
     *
     * @param vmaf   The VMAF context allocated with `vmaf_init()`.
     *
     * @param model  Opaque model context.
     *
     * @param index  Picture index.
     *
     * @param score  Predicted score.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn score_at_index(&self, model: &Model, index: usize) -> Result<f64, Error> {
        let mut score = 0.0f64;
        let r = unsafe { vmaf_score_at_index(self.ptr.as_ptr(), model.ptr.as_ptr(), &mut score, to_c_uint(index)?) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(score)
    }

    /**
     * Predict VMAF score at specific index, using a model collection.
     *
     * @param vmaf              The VMAF context allocated with `vmaf_init()`.
     *
     * @param model_collection  Opaque model collection context.
     *
     * @param index             Picture index.
     *
     * @param score             Predicted score.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn bootstrapped_score_at_index(&self, model: &Model, index: usize) -> Result<BootstrappedScore, Error> {
        let mut score: VmafModelCollectionScore = unsafe { mem::zeroed() };
        let Some(model_collection_ptr) = model.collection_ptr else {
            return Err(Error::InvalidArgument);
        };
        let r = unsafe { vmaf_score_at_index_model_collection(self.ptr.as_ptr(), model_collection_ptr.as_ptr(), &mut score, to_c_uint(index)?) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        assert!(score.type_ == VmafModelCollectionScoreType::VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP);
        Ok(BootstrappedScore {
            bagging_score: score.bootstrap.bagging_score,
            stddev: score.bootstrap.stddev,
            ci_p95_lo: score.bootstrap.ci.p95.lo,
            ci_p95_hi: score.bootstrap.ci.p95.hi,
        })
    }

    /**
     * Fetch feature score at specific index.
     *
     * @param vmaf          The VMAF context allocated with `vmaf_init()`.
     *
     * @param feature_name  Name of the feature to fetch.
     *
     * @param index         Picture index.
     *
     * @param score         Score.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn feature_score_at_index(&self, feature_name: impl AsRef<str>, index: usize) -> Result<f64, Error> {
        let feature_name = CString::new(feature_name.as_ref()).map_err(|_| Error::InvalidArgument)?;

        let mut score = 0.0f64;
        let r = unsafe { vmaf_feature_score_at_index(self.ptr.as_ptr(), (&feature_name).as_ptr(), &mut score, to_c_uint(index)?) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(score)
    }

    /**
     * Pooled VMAF score for a specific interval.
     *
     * @param vmaf         The VMAF context allocated with `vmaf_init()`.
     *
     * @param model        Opaque model context.
     *
     * @param pool_method  Temporal pooling method to use.
     *
     * @param score        Pooled score.
     *
     * @param index_low    Low picture index of pooling interval.
     *
     * @param index_high   High picture index of pooling interval.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn pooled_score(&self, model: &Model, pool_method: PoolingMethod, start_index: usize, end_index: usize) -> Result<f64, Error> {
        assert!(start_index < end_index);
        let mut score = 0.0f64;
        let r = unsafe {
            vmaf_score_pooled(
                self.ptr.as_ptr(),
                model.ptr.as_ptr(),
                pool_method.into(),
                &mut score,
                to_c_uint(start_index)?,
                to_c_uint(end_index - 1)?,
            )
        };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(score)
    }

    /**
     * Pooled VMAF score for a specific interval, using a model collection.
     *
     * @param vmaf              The VMAF context allocated with `vmaf_init()`.
     *
     * @param model_collection  Opaque model collection context.
     *
     * @param pool_method       Temporal pooling method to use.
     *
     * @param score             Pooled score.
     *
     * @param index_low         Low picture index of pooling interval.
     *
     * @param index_high        High picture index of pooling interval.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn pooled_bootstrapped_score(&self, model: &Model, pool_method: PoolingMethod, start_index: usize, end_index: usize) -> Result<BootstrappedScore, Error> {
        assert!(start_index < end_index);
        let mut score: VmafModelCollectionScore = unsafe { mem::zeroed() };
        let Some(model_collection_ptr) = model.collection_ptr else {
            return Err(Error::InvalidArgument);
        };
        let r = unsafe {
            vmaf_score_pooled_model_collection(
                self.ptr.as_ptr(),
                model_collection_ptr.as_ptr(),
                pool_method.into(),
                &mut score,
                to_c_uint(start_index)?,
                to_c_uint(end_index - 1)?,
            )
        };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        assert!(score.type_ == VmafModelCollectionScoreType::VMAF_MODEL_COLLECTION_SCORE_BOOTSTRAP);
        Ok(BootstrappedScore {
            bagging_score: score.bootstrap.bagging_score,
            stddev: score.bootstrap.stddev,
            ci_p95_lo: score.bootstrap.ci.p95.lo,
            ci_p95_hi: score.bootstrap.ci.p95.hi,
        })
    }

    /**
     * Pooled feature score for a specific interval.
     *
     * @param vmaf          The VMAF context allocated with `vmaf_init()`.
     *
     * @param feature_name  Name of the feature to fetch.
     *
     * @param pool_method   Temporal pooling method to use.
     *
     * @param score         Pooled score.
     *
     * @param index_low     Low picture index of pooling interval.
     *
     * @param index_high    High picture index of pooling interval.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn pooled_feature_score(&self, feature_name: impl AsRef<str>, pool_method: PoolingMethod, start_index: usize, end_index: usize) -> Result<f64, Error> {
        assert!(start_index < end_index);
        let feature_name = CString::new(feature_name.as_ref()).map_err(|_| Error::InvalidArgument)?;
        let mut score = 0.0f64;
        let r = unsafe {
            vmaf_feature_score_pooled(
                self.ptr.as_ptr(),
                (&feature_name).as_ptr(),
                pool_method.into(),
                &mut score,
                to_c_uint(start_index)?,
                to_c_uint(end_index - 1)?,
            )
        };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(score)
    }


    /**
     * Write VMAF stats to an output file.
     *
     * @param vmaf         The VMAF context allocated with `vmaf_init()`.
     *
     * @param output_path  Output file path.
     *
     * @param fmt          Output file format.
     *                     See `enum VmafOutputFormat` for options.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    pub fn write_score_to_path(&self, output_path: impl AsRef<Path>, fmt: OutputFormat) -> Result<(), Error> {
        let output_path = CString::new(output_path.as_ref().as_os_str().to_raw_bytes()).map_err(|_| Error::InvalidArgument)?;

        let r = unsafe { vmaf_write_output(self.ptr.as_ptr(), (&output_path).as_ptr(), fmt.into()) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }
        Ok(())
    }
}

impl Drop for Context {
    /**
     * Close a VMAF instance and free all associated memory.
     *
     * @param vmaf The VMAF instance to close.
     *
     *
     * @return 0 on success, or < 0 (a negative errno code) on error.
     */
    fn drop(&mut self) {
        unsafe { vmaf_close(self.ptr.as_ptr()) };
    }
}

#[derive(Debug, Clone)]
pub enum ModelVersion {
    String(String),
    Path(PathBuf),
}

impl Default for ModelVersion {
    fn default() -> Self {
        Self::String("vmaf_b_v0.6.3".into())
    }
}

#[derive(Debug)] 
pub struct ModelCollectScoreOpts {
    pub log_level: LogLevel,
    pub n_threads: usize,
    pub n_subsample: usize,
    pub cpumask: u64,
    pub pool_method: PoolingMethod,
    pub output_path: Option<PathBuf>,
    pub output_format: Option<OutputFormat>,
}

impl Default for ModelCollectScoreOpts {
    fn default() -> Self {
        Self {
            log_level: LogLevel::default(),
            n_threads: 0,
            n_subsample: 0,
            cpumask: 0,
            pool_method: PoolingMethod::default(),
            output_path: None,
            output_format: None,
        }
    }
}


#[derive(Debug)]
pub struct Model {
    ptr: NonNull<VmafModel>,
    collection_ptr: Option<NonNull<VmafModelCollection>>,
}

impl Model {
    pub fn new(cfg: &ModelConfig, version: &ModelVersion) -> Result<Self, Error> {
        match version {
            ModelVersion::String(version) => Self::from_version(cfg, version),
            ModelVersion::Path(path) => Self::from_path(cfg, path),
        }
    }

    fn from_version(cfg: &ModelConfig, version: impl AsRef<str>) -> Result<Self, Error> {
        let version = CString::new(version.as_ref()).map_err(|_| Error::InvalidArgument)?;

        let load_simple_model = || -> Result<NonNull<VmafModel>, Error> {
            let mut ptr: *mut VmafModel = ptr::null_mut();
            let r = unsafe { vmaf_model_load(&mut ptr, &cfg.resource as *const _ as *mut _, (&version).as_ptr()) };
            if r != 0 {
                return Err(Error::from_sys(r));
            }
            Ok(NonNull::new(ptr).unwrap())
        };

        let load_colletion_model = || -> Result<(NonNull<VmafModel>, NonNull<VmafModelCollection>), Error> {
            let mut ptr: *mut VmafModel = ptr::null_mut();
            let mut collection_ptr: *mut VmafModelCollection = ptr::null_mut();

            let r = unsafe { vmaf_model_collection_load(&mut ptr, &mut collection_ptr, &cfg.resource as *const _ as *mut _, (&version).as_ptr()) };
            if r != 0 {
                return Err(Error::from_sys(r));
            }
            Ok((NonNull::new(ptr).unwrap(), NonNull::new(collection_ptr).unwrap()))
        };

        Self::load_simple_or_collection_model(load_simple_model, load_colletion_model)
    }

    fn from_path(cfg: &ModelConfig, path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = CString::new(path.as_ref().as_os_str().to_raw_bytes()).map_err(|_| Error::InvalidArgument)?;

        let load_simple_model = || -> Result<NonNull<VmafModel>, Error> {
            let mut ptr: *mut VmafModel = ptr::null_mut();
            let r = unsafe { vmaf_model_load_from_path(&mut ptr, &cfg.resource as *const _ as *mut _, (&path).as_ptr()) };
            if r != 0 {
                return Err(Error::from_sys(r));
            }
            Ok(NonNull::new(ptr).unwrap())
        };

        let load_colletion_model = || -> Result<(NonNull<VmafModel>, NonNull<VmafModelCollection>), Error> {
            let mut ptr: *mut VmafModel = ptr::null_mut();
            let mut collection_ptr: *mut VmafModelCollection = ptr::null_mut();

            let r = unsafe { vmaf_model_collection_load_from_path(&mut ptr, &mut collection_ptr, &cfg.resource as *const _ as *mut _, (&path).as_ptr()) };
            if r != 0 {
                return Err(Error::from_sys(r));
            }
            Ok((NonNull::new(ptr).unwrap(), NonNull::new(collection_ptr).unwrap()))
        };

        Self::load_simple_or_collection_model(load_simple_model, load_colletion_model)
    }

    fn load_simple_or_collection_model(load_simple_model: impl FnOnce() -> Result<NonNull<VmafModel>, Error>, load_colletion_model: impl FnOnce() -> Result<(NonNull<VmafModel>, NonNull<VmafModelCollection>), Error>) -> Result<Self, Error> {
        let (ptr, collection_ptr) = match load_simple_model() {
            Ok(ptr) => (ptr, None),
            Err(Error::InvalidArgument) => {
                let (ptr, collection_ptr) = load_colletion_model()?;
                (ptr, Some(collection_ptr))
            },
            Err(err) => return Err(err),
        };
        Ok(Self { ptr, collection_ptr })
    }

    pub fn feature_overload(&mut self, feature_name: impl AsRef<str>, dict: &FeatureDictionary) -> Result<(), Error> {
        let feature_name = CString::new(feature_name.as_ref()).map_err(|_| Error::InvalidArgument)?;

        let ptr = unsafe { self.ptr.as_mut() };
        let r = unsafe { vmaf_model_feature_overload(ptr, (&feature_name).as_ptr(), dict.ptr) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }

        if let Some(mut collection_ptr) = self.collection_ptr {
            let mut collection_ptr = unsafe { collection_ptr.as_mut() as *mut _ };
            let collection_ptr = &mut collection_ptr as *mut _;
            let r = unsafe { vmaf_model_collection_feature_overload(ptr, collection_ptr, (&feature_name).as_ptr(), dict.ptr) };
            if r != 0 {
                return Err(Error::from_sys(r));
            }
        }

        Ok(())
    }

    pub fn collect_score(self, ref_path: impl TryInto<AutoPictureStream, Error=Error>, dist_path: impl TryInto<AutoPictureStream, Error=Error>, opts: ModelCollectScoreOpts) -> Result<f64, Error> {
        let ref_stream = ref_path.try_into()?;
        let dist_stream = dist_path.try_into()?;

        self.collect_score_from_stream_pair(ref_stream, dist_stream, opts)
    }

    pub fn collect_score_from_stream_pair(&self, mut ref_stream: impl PictureStream, mut dist_stream: impl PictureStream, opts: ModelCollectScoreOpts) -> Result<f64, Error> {
        let cfg = Configuration::builder()
            .log_level(opts.log_level)
            .n_threads(opts.n_threads)
            .n_subsample(opts.n_subsample)
            .cpumask(opts.cpumask)
            .build()?;

        let mut ctx = Context::new(cfg)?;
        ctx.use_simple_model_feature(self)?;

        let mut index = 0;
        while let (Some(ref_pic), Some(dist_pic)) = (ref_stream.next_pic(), dist_stream.next_pic()) {
            let ref_pic = ref_pic?;
            let dist_pic = dist_pic?;
            ctx.read_pictures(ref_pic, dist_pic, index)?;
            index += 1;
        }

        ctx.wait_for_all_pictures_flushed()?;
        let score = ctx.pooled_score(self, opts.pool_method, 0, index)?;

        match (opts.output_path, opts.output_format) {
            (Some(output_path), Some(output_format)) => ctx.write_score_to_path(output_path, output_format)?,
            (Some(output_path), None) => {
                let output_format = OutputFormat::from_path(&output_path);
                ctx.write_score_to_path(output_path, output_format)?;
            },
            (None, Some(output_format)) => {
                ctx.write_score_to_path(output_format.default_path(), output_format)?;
            },
            (None, None) => (),
        }

        Ok(score)
    }

    pub fn collect_bootstrapped_score(self, ref_path: impl TryInto<AutoPictureStream, Error=Error>, dist_path: impl TryInto<AutoPictureStream, Error=Error>, opts: ModelCollectScoreOpts) -> Result<BootstrappedScore, Error> {
        let ref_stream = ref_path.try_into()?;
        let dist_stream = dist_path.try_into()?;

        self.collect_bootstrapped_score_from_stream_pair(ref_stream, dist_stream, opts)
    }

    pub fn collect_bootstrapped_score_from_stream_pair(&self, mut ref_stream: impl PictureStream, mut dist_stream: impl PictureStream, opts: ModelCollectScoreOpts) -> Result<BootstrappedScore, Error> {
        let cfg = Configuration::builder()
            .log_level(opts.log_level)
            .n_threads(opts.n_threads)
            .n_subsample(opts.n_subsample)
            .cpumask(opts.cpumask)
            .build()?;

        let mut ctx = Context::new(cfg)?;
        ctx.use_collection_model_feature(self)?;

        let mut index = 0;
        while let (Some(ref_pic), Some(dist_pic)) = (ref_stream.next_pic(), dist_stream.next_pic()) {
            let ref_pic = ref_pic?;
            let dist_pic = dist_pic?;
            ctx.read_pictures(ref_pic, dist_pic, index)?;
            index += 1;
        }

        ctx.wait_for_all_pictures_flushed()?;
        let score = ctx.pooled_bootstrapped_score(self, opts.pool_method, 0, index)?;

        match (opts.output_path, opts.output_format) {
            (Some(output_path), Some(output_format)) => ctx.write_score_to_path(output_path, output_format)?,
            (Some(output_path), None) => {
                let output_format = OutputFormat::from_path(&output_path);
                ctx.write_score_to_path(output_path, output_format)?;
            },
            (None, Some(output_format)) => {
                ctx.write_score_to_path(output_format.default_path(), output_format)?;
            },
            (None, None) => (),
        }

        Ok(score)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { vmaf_model_destroy(self.ptr.as_ptr()) };
        if let Some(collection_ptr) = self.collection_ptr {
            unsafe { vmaf_model_collection_destroy(collection_ptr.as_ptr()) };
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        let cfg = ModelConfig::builder().build().expect("Failed to make empty config");
        Self::new(&cfg, &ModelVersion::default()).expect("Failed to load default model")
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum PixelFormat {
    Unknown,
    Yuv400p,
    Yuv420p,
    Yuv422p,
    Yuv444p,
}

impl Into<VmafPixelFormat> for PixelFormat {
    fn into(self) -> VmafPixelFormat {
        match self {
            Self::Unknown => VmafPixelFormat::VMAF_PIX_FMT_UNKNOWN,
            Self::Yuv420p => VmafPixelFormat::VMAF_PIX_FMT_YUV420P,
            Self::Yuv422p => VmafPixelFormat::VMAF_PIX_FMT_YUV422P,
            Self::Yuv444p => VmafPixelFormat::VMAF_PIX_FMT_YUV444P,
            Self::Yuv400p => VmafPixelFormat::VMAF_PIX_FMT_YUV400P,
        }
    }
}

impl From<VmafPixelFormat> for PixelFormat {
    fn from(format: VmafPixelFormat) -> Self {
        match format {
            VmafPixelFormat::VMAF_PIX_FMT_UNKNOWN => Self::Unknown,
            VmafPixelFormat::VMAF_PIX_FMT_YUV400P => Self::Yuv400p,
            VmafPixelFormat::VMAF_PIX_FMT_YUV420P => Self::Yuv420p,
            VmafPixelFormat::VMAF_PIX_FMT_YUV422P => Self::Yuv422p,
            VmafPixelFormat::VMAF_PIX_FMT_YUV444P => Self::Yuv444p,
        }
    }
}

pub trait Component {
    fn data(&self) -> &[u8];
    fn width(&self) -> usize;
    fn height(&self) -> usize;

    fn stride(&self) -> usize {
        self.width()
    }
}

pub trait Frame<C: Component> {
    fn pixel_format(&self) -> PixelFormat;
    fn width(&self) -> usize;
    fn height(&self) -> usize;
    fn depth(&self) -> usize;
    fn y_component(&self) -> &C;
    fn u_component(&self) -> Option<&C>;
    fn v_component(&self) -> Option<&C>;

    fn components(&self) -> Vec<&C> {
        if self.pixel_format() == PixelFormat::Yuv400p {
            vec![self.y_component()]
        } else {
            vec![
                self.y_component(),
                self.u_component().expect("must have chroma"),
                self.v_component().expect("must have chroma"),
            ]
        }
    }

    fn pixel_stride(&self) -> usize {
        self.depth().div_ceil(8)
    }
}

pub trait PictureStream {
    fn next_pic(&mut self) -> Option<Result<Picture, Error>>;
}

pub enum AutoPictureStream {
    Y4m(y4m::PictureStream<File>),
    Dynamic(Box<dyn PictureStream>),
}

impl PictureStream for AutoPictureStream {
    fn next_pic(&mut self) -> Option<Result<Picture, Error>> {
        match self {
            Self::Y4m(stream) => stream.next_pic(),
            Self::Dynamic(stream) => stream.next_pic(),
        }
    }
}

impl AutoPictureStream {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        match path.extension() {
            None => Err(Error::InvalidArgument),
            Some(ext) => match ext.to_str() {
                Some("y4m") => Ok(Self::Y4m(y4m::PictureStream::from_path(path)?)),
                _ => Err(Error::InvalidArgument),
            },
        }
    }

    pub fn from_path_with_custom_builder(path: impl AsRef<Path>, custom_builder: impl FnOnce(&Path) -> Option<Result<Box<dyn PictureStream>, Box<dyn error::Error>>>) -> Result<Self, Error> {
        let path = path.as_ref();
        match custom_builder(path) {
            Some(stream) => match stream {
                Ok(stream) => Ok(Self::Dynamic(stream)),
                Err(err) => Err(Error::FailedInCustomPictureStreamBuilder(err)),
            },
            None => Self::from_path(path),
        }
    }
}

// XXX I don't known how to impl TryFrom<AsRef<Path>>, so used macro.
macro_rules! auto_picture_stream_try_from_path {
    ($($t:ty),*) => {
        $(
            impl TryFrom<$t> for AutoPictureStream {
                type Error = Error;
                fn try_from(path: $t) -> Result<Self, Self::Error> {
                    Self::from_path(path)
                }
            }
        )*
    };
}
auto_picture_stream_try_from_path!(&Path, &ffi::OsStr, &borrow::Cow<'_, ffi::OsStr>, &ffi::OsString, &str, &String, &PathBuf);

#[derive(Debug)]
pub struct Picture {
    ptr: Option<Box<VmafPicture>>,
}

impl Picture {
    pub fn new<F: Frame<C>, C: Component>(frame: &F) -> Result<Self, Error>
    {
        let mut ptr: Box<VmafPicture> = Box::new(unsafe { mem::zeroed() });

        let r = unsafe { vmaf_picture_alloc(ptr.as_mut(), frame.pixel_format().into(), to_c_uint(frame.depth())?, to_c_uint(frame.width())?, to_c_uint(frame.height())?) };
        if r != 0 {
            return Err(Error::from_sys(r));
        }

        let depth = frame.depth();
        assert_eq!(from_c_uint(ptr.bpc)?, depth);

        let pixel_stride = frame.pixel_stride();
        assert_eq!(from_c_uint(ptr.bpc.div_ceil(8))?, pixel_stride);

        let frame_components = frame.components();
        for (component_index, frame_component) in frame_components.iter().enumerate() {
            let component_width = frame_component.width();
            assert_eq!(from_c_uint(ptr.w[component_index])?, component_width);

            let component_height = frame_component.height();
            assert_eq!(from_c_uint(ptr.h[component_index])?, component_height);

            let src_row_stride = frame_component.stride();
            assert!(component_width * pixel_stride <= src_row_stride);
            assert!(src_row_stride * component_height <= frame_component.data().len());

            // error when negative pointer diff
            let dst_row_stride = ptr.stride[component_index].try_into().map_err(|_| Error::InvalidArgument)?;
            assert!(component_width * pixel_stride <= dst_row_stride);

            let src_data = frame_component.data().as_ptr() as *const u8;
            let dst_data = ptr.data[component_index] as *mut u8;

            for y in 0..component_height {
                let n_copy_bytes = component_width * pixel_stride;
                let src_offset = y * src_row_stride;
                let dst_offset = y * dst_row_stride;

                unsafe {
                    copy_nonoverlapping(src_data.wrapping_add(src_offset), dst_data.wrapping_add(dst_offset), n_copy_bytes);
                }
            }
        };

        Ok(Picture { ptr: Some(ptr) })
    }

    pub fn consume(mut self) -> Result<Box<VmafPicture>, Error> {
        if let Some(ptr) = self.ptr.take() {
            Ok(ptr)
        } else {
            Err(Error::InvalidArgument)
        }
    }
}

impl Drop for Picture {
    fn drop(&mut self) {
        if let Some(ref mut ptr) = &mut self.ptr {
            unsafe { vmaf_picture_unref(ptr.as_mut()) };
        }
    }
}

fn to_c_uint(u: usize) -> Result<ffi::c_uint, Error> {
    match u.try_into() {
        Ok(u) => Ok(u),
        Err(_) => Err(Error::InvalidArgument),
    }
}

fn from_c_uint(u: ffi::c_uint) -> Result<usize, Error> {
    match u.try_into() {
        Ok(u) => Ok(u),
        Err(_) => Err(Error::InvalidArgument),
    }
}

#[cfg(test)]
mod tests {
    use std::{error, path::{Path, PathBuf}};
    use super::*;

    #[test]
    fn log_level_general_usage() -> Result<(), Box<dyn error::Error>> {
        // general usage
        let log_level = LogLevel::Debug;
        let score = collect_score("videos/short_original.y4m", "videos/short_high_quality.y4m", CollectScoreOpts { log_level, ..Default::default() })?;
        assert!(score >= 90.0);
        assert!(score <= 100.0);

        // default
        assert_eq!(LogLevel::default(), LogLevel::None);

        Ok(())
    }

    #[test]
    fn output_format_general_usage() -> Result<(), Box<dyn error::Error>> {
        // general usage
        let output_format = OutputFormat::Json;
        let _ = std::fs::remove_file(output_format.default_path());
        assert!(!Path::new("vmaf_output.json").exists());
        let score = collect_score("videos/short_original.y4m", "videos/short_high_quality.y4m", CollectScoreOpts {
            output_format: Some(output_format),
            ..Default::default()
        })?;
        assert!(Path::new("vmaf_output.json").exists());
        assert!(score >= 90.0);
        assert!(score <= 100.0);

        // override default path
        let output_format = OutputFormat::Json;
        let output_path = Path::new("custom_name.json");
        let _ = std::fs::remove_file(output_path);
        assert!(!output_path.exists());
        let score = collect_score("videos/short_original.y4m", "videos/short_low_quality.y4m", CollectScoreOpts {
            output_format: Some(output_format),
            output_path: Some(PathBuf::from(output_path)),
            ..Default::default()
        })?;
        assert!(output_path.exists());
        assert!(score >= 0.0);
        assert!(score <= 80.0);

        // default path
        assert_eq!(OutputFormat::None.default_path(), Path::new("vmaf_output.dat"));
        assert_eq!(OutputFormat::Xml.default_path(), Path::new("vmaf_output.xml"));
        assert_eq!(OutputFormat::Json.default_path(), Path::new("vmaf_output.json"));
        assert_eq!(OutputFormat::Csv.default_path(), Path::new("vmaf_output.csv"));
        assert_eq!(OutputFormat::Sub.default_path(), Path::new("vmaf_output.sub"));

        // default
        assert_eq!(OutputFormat::default(), OutputFormat::Xml);

        Ok(())
    }
}

