use std::{io::{self, BufReader}, path::{Path, PathBuf}, fs::File, error, borrow, fmt, ffi, marker::PhantomData, sync::atomic::{AtomicUsize, Ordering}};

pub mod libvmaf;
pub mod y4m;

#[cfg(feature="gst")]
pub mod gst;

#[cfg(not(feature="gst"))]
mod gst {
    pub fn supported() -> bool { false }
    #[derive(Debug)]
    pub enum Error {
    }
    #[derive(Debug)]
    pub struct PictureStream<R: std::io::Read + std::io::Seek + Send + Sync + 'static> {
        _marker: std::marker::PhantomData<R>,
    }
    impl PictureStream<std::io::BufReader<std::fs::File>>  {
        pub fn from_path(_path: impl AsRef<std::path::Path>, _opts: gst::PictureStreamOpts) -> Result<Self, super::Error> {
            panic!("need gst feature");
        }
    }
    impl<R: std::io::Read + std::io::Seek + Send + Sync + 'static> super::PictureStream for PictureStream<R> {
        fn next_pic(&mut self) -> Result<Option<super::Picture>, super::Error> {
            panic!("need gst feature");
        }
    }
}

pub use libvmaf::version;
pub use libvmaf::LogLevel;
pub use libvmaf::OutputFormat;
pub use libvmaf::PoolingMethod;
pub use libvmaf::BootstrappedScore;
pub use libvmaf::ModelFlags;
pub use libvmaf::PixelFormat;
pub use libvmaf::Component;
pub use libvmaf::Frame;


#[derive(Debug)]
pub enum Error {
    LibvmafError(String, libvmaf::Error),
    IoError(io::Error),
    FailedInCustomPictureStreamBuilder(Box<dyn error::Error>),
    InvalidVideoFrameFormat(String),
    UnsupportedVideoFile(String),
    Y4mError(y4m::Error),
    GstreamerError(gst::Error),
}

impl From<y4m::Error> for Error {
    fn from(err: y4m::Error) -> Self {
        Self::Y4mError(err)
    }
}

impl From<gst::Error> for Error {
    fn from(err: gst::Error) -> Self {
        Self::GstreamerError(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error { }

impl Into<libvmaf::ModelVersion> for ModelVersion {
    fn into(self) -> libvmaf::ModelVersion {
        match self {
            Self::Version063B => libvmaf::ModelVersion::String("vmaf_b_v0.6.3".into()),
            Self::Path(path) => libvmaf::ModelVersion::Path(path),
        }
    }
}

impl Default for ModelVersion {
    fn default() -> Self {
        Self::Version063B
    }
}

macro_rules! liberr {
    ($($arg:tt)*) => {
        liberr_converter(format!($($arg)*))
    };
}

fn liberr_converter(description: String) -> impl FnOnce(libvmaf::Error) -> Error {
    |err| {
        Error::LibvmafError(description, err)
    }
}


#[derive(Debug)] 
pub struct ScoreCollectorOpts {
    pub log_level: LogLevel,
    pub n_threads: usize,
    pub n_subsample: usize,
    pub cpumask: u64,
}

impl Default for ScoreCollectorOpts {
    fn default() -> Self {
        Self {
            log_level: LogLevel::default(),
            n_threads: 0,
            n_subsample: 0,
            cpumask: 0,
        }
    }
}


#[derive(Debug)]
pub struct ScoreCollectorCollectScoreOpts {
    pub pool_method: PoolingMethod,
    pub output_path: Option<PathBuf>,
    pub output_format: Option<OutputFormat>,
}

impl Default for ScoreCollectorCollectScoreOpts {
    fn default() -> Self {
        Self {
            pool_method: PoolingMethod::default(),
            output_path: None,
            output_format: None,
        }
    }
}

#[derive(Debug)]
pub struct ScoreCollector<Score> {
    context: libvmaf::Context,
    model: Model,
    n_scores: AtomicUsize,
    _marker: PhantomData<Score>,
}

impl<Score> ScoreCollector<Score> {
    fn new_impl(model: Model, opts: ScoreCollectorOpts) -> Result<Self, Error> {
        let cfg = libvmaf::Configuration::builder()
            .log_level(opts.log_level)
            .n_threads(opts.n_threads)
            .n_subsample(opts.n_subsample)
            .cpumask(opts.cpumask)
            .build().map_err(liberr!("Failed to build libvmaf Context Configuration"))?;

        let context = libvmaf::Context::new(cfg).map_err(liberr!("Failed to build libvmaf context"))?;
        let n_scores = AtomicUsize::new(0);

        Ok(Self { context, model, n_scores, _marker: PhantomData })
    }

    pub fn read_pictures(&mut self, ref_pic: Picture, dist_pic: Picture) -> Result<(), Error> {
        let index = self.next_index();
        self.context.read_pictures(ref_pic.picture, dist_pic.picture, index).map_err(liberr!("Failed to read pictures"))
    }

    pub fn n_scores(&self) -> usize {
        self.n_scores.load(Ordering::Acquire)
    }

    fn write_score_if_needed(&self, output_path: Option<PathBuf>, output_format: Option<OutputFormat>) -> Result<(), Error> {
        if (output_path.as_ref(), output_format) != (None, None) {
            let (output_path, output_format) = match (output_path.clone(), output_format) {
                (Some(output_path), Some(output_format)) => (output_path, output_format),
                (Some(output_path), None) => {
                    let output_format = OutputFormat::from_path(&output_path);
                    (output_path, output_format)
                },
                (None, Some(output_format)) => (output_format.default_path(), output_format),
                (None, None) => unreachable!(),
            };
            self.context.write_score_to_path(&output_path, output_format).map_err(liberr!("Failed to write vmaf scores to file: {:?}", output_path))?;
        }
        Ok(())
    }

    fn next_index(&mut self) -> usize {
        let mut n_scores = self.n_scores.load(Ordering::Relaxed);
        loop {
            let next_n_scores = n_scores + 1;
            match self.n_scores.compare_exchange_weak(n_scores, next_n_scores, Ordering::SeqCst, Ordering::Relaxed) {
                Ok(_) => break n_scores,
                Err(next_n_scores) => {
                    n_scores = next_n_scores;
                },
            }
        }
    }
}

impl ScoreCollector<f64> {
    pub fn new(model: Model, opts: ScoreCollectorOpts) -> Result<Self, Error> {
        let mut collector = Self::new_impl(model, opts)?;
        collector.context.use_simple_model_feature(&collector.model.model).map_err(liberr!("Failed to set model feature to use"))?;
        Ok(collector)
    }

    pub fn collect_score(&mut self, opts: ScoreCollectorCollectScoreOpts) -> Result<f64, Error> {
        self.context.wait_for_all_pictures_flushed().map_err(liberr!("Failed to flush scores"))?;
        let score = self.context.pooled_score(
            &self.model.model,
            opts.pool_method,
            0, self.n_scores()
        ).map_err(liberr!("Failed to pool score"))?;

        self.write_score_if_needed(opts.output_path, opts.output_format)?;

        Ok(score)
    }
}

impl ScoreCollector<BootstrappedScore> {
    pub fn new(model: Model, opts: ScoreCollectorOpts) -> Result<Self, Error> {
        let mut collector = Self::new_impl(model, opts)?;
        collector.context.use_collection_model_feature(&collector.model.model).map_err(liberr!("Failed to set model feature to use"))?;
        Ok(collector)
    }

    pub fn collect_score(&mut self, opts: ScoreCollectorCollectScoreOpts) -> Result<BootstrappedScore, Error> {
        self.context.wait_for_all_pictures_flushed().map_err(liberr!("Failed to flush scores"))?;
        let score = self.context.pooled_bootstrapped_score(
            &self.model.model,
            opts.pool_method,
            0, self.n_scores()
        ).map_err(liberr!("Failed to pool bootstrapped score"))?;

        self.write_score_if_needed(opts.output_path, opts.output_format)?;

        Ok(score)
    }
}

#[derive(Debug, Clone)]
pub enum ModelVersion {
    // TODO more versions
    Version063B,
    Path(PathBuf),
}

#[derive(Debug)]
pub struct ModelOpts {
    pub flags: ModelFlags,
    pub name_to_override: Option<String>,
    pub version: ModelVersion,
}

impl Default for ModelOpts {
    fn default() -> Self {
        Self {
            flags: ModelFlags::default(),
            name_to_override: None,
            version: ModelVersion::default(),
        }
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
        let score_collector_opts = ScoreCollectorOpts::default();
        let collect_score_opts = ScoreCollectorCollectScoreOpts::default();
        Self {
            log_level: score_collector_opts.log_level,
            n_threads: score_collector_opts.n_threads,
            n_subsample: score_collector_opts.n_subsample,
            cpumask: score_collector_opts.cpumask,
            pool_method: collect_score_opts.pool_method,
            output_path: collect_score_opts.output_path,
            output_format: collect_score_opts.output_format,
        }
    }
}


#[derive(Debug)]
pub struct Model {
    model: libvmaf::Model,
}

impl Model {
    pub fn new(opts: ModelOpts) -> Result<Self, Error> {
        let cfg = libvmaf::ModelConfig::builder()
            .name(opts.name_to_override)
            .flags(opts.flags)
            .build().map_err(liberr!("Failed to build ModelConfig"))?;

        let model = libvmaf::Model::new(&cfg, opts.version.clone().into())
            .map_err(liberr!("Failed to load model version: {:?}", opts.version))?;

        Ok(Self { model })
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::new(ModelOpts::default()).expect("Failed to load default model")
    }
}

#[derive(Debug)]
pub struct CollectScoreOpts {
    pub model_flags: ModelFlags,
    pub model_version: ModelVersion,
    pub model_name_to_override: Option<String>,
    pub log_level: LogLevel,
    pub n_threads: usize,
    pub n_subsample: usize,
    pub cpumask: u64,
    pub pool_method: PoolingMethod,
    pub output_path: Option<PathBuf>,
    pub output_format: Option<OutputFormat>,
}

impl CollectScoreOpts {
    pub fn into_score_collector_opts(self) -> (ModelOpts, ScoreCollectorOpts, ScoreCollectorCollectScoreOpts) {
        let Self { 
            model_flags, model_version, model_name_to_override,
            log_level, n_threads, n_subsample, cpumask,
            pool_method, output_path, output_format,
        } = self;
        (
            ModelOpts { flags: model_flags, version: model_version, name_to_override: model_name_to_override },
            ScoreCollectorOpts { log_level, n_threads, n_subsample, cpumask },
            ScoreCollectorCollectScoreOpts { pool_method, output_path, output_format },
        )
    }

    pub fn into_model_opts(self) -> (ModelOpts, ModelCollectScoreOpts) {
        let Self { 
            model_flags, model_version, model_name_to_override,
            log_level, n_threads, n_subsample, cpumask,
            pool_method, output_path, output_format,
        } = self;
        (
            ModelOpts { flags: model_flags, version: model_version, name_to_override: model_name_to_override },
            ModelCollectScoreOpts { log_level, n_threads, n_subsample, cpumask, pool_method, output_path, output_format },
        )
    }
}

impl Default for CollectScoreOpts {
    fn default() -> Self {
        let model_opts = ModelOpts::default();
        let score_collector_opts = ScoreCollectorOpts::default();
        let collect_score_opts = ScoreCollectorCollectScoreOpts::default();
        Self {
            model_flags: model_opts.flags,
            model_version: model_opts.version,
            model_name_to_override: model_opts.name_to_override,
            log_level: score_collector_opts.log_level,
            n_threads: score_collector_opts.n_threads,
            n_subsample: score_collector_opts.n_subsample,
            cpumask: score_collector_opts.cpumask,
            pool_method: collect_score_opts.pool_method,
            output_path: collect_score_opts.output_path,
            output_format: collect_score_opts.output_format,
        }
    }
}

pub fn collect_score(ref_path: impl TryInto<AutoPictureStream, Error=Error>, dist_path: impl TryInto<AutoPictureStream, Error=Error>, opts: CollectScoreOpts) -> Result<f64, Error> {
    let ref_stream = ref_path.try_into()?;
    let dist_stream = dist_path.try_into()?;

    collect_score_from_stream_pair(ref_stream, dist_stream, opts)
}

pub fn collect_score_from_stream_pair(mut ref_stream: impl PictureStream, mut dist_stream: impl PictureStream, opts: CollectScoreOpts) -> Result<f64, Error> {
    let (model_opts, score_collector_opts, collect_score_opts) = opts.into_score_collector_opts();
    let model = Model::new(model_opts)?;
    let mut score_collector = ScoreCollector::<f64>::new(model, score_collector_opts)?;

    while let (Some(ref_pic), Some(dist_pic)) = (ref_stream.next_pic()?, dist_stream.next_pic()?) {
        score_collector.read_pictures(ref_pic, dist_pic)?;
    }
    score_collector.collect_score(collect_score_opts)
}

pub fn collect_bootstrapped_score(ref_path: impl TryInto<AutoPictureStream, Error=Error>, dist_path: impl TryInto<AutoPictureStream, Error=Error>, opts: CollectScoreOpts) -> Result<BootstrappedScore, Error> {
    let ref_stream = ref_path.try_into()?;
    let dist_stream = dist_path.try_into()?;

    collect_bootstrapped_score_from_stream_pair(ref_stream, dist_stream, opts)
}

pub fn collect_bootstrapped_score_from_stream_pair(mut ref_stream: impl PictureStream, mut dist_stream: impl PictureStream, opts: CollectScoreOpts) -> Result<BootstrappedScore, Error> {
    let (model_opts, score_collector_opts, collect_score_opts) = opts.into_score_collector_opts();
    let model = Model::new(model_opts)?;
    let mut score_collector = ScoreCollector::<BootstrappedScore>::new(model, score_collector_opts)?;

    while let (Some(ref_pic), Some(dist_pic)) = (ref_stream.next_pic()?, dist_stream.next_pic()?) {
        score_collector.read_pictures(ref_pic, dist_pic)?;
    }
    score_collector.collect_score(collect_score_opts)
}

#[derive(Debug)]
pub struct Picture {
    picture: libvmaf::Picture,
}

impl Picture {
    pub fn new<F: Frame<C>, C: Component>(frame: &F) -> Result<Self, Error> {
        let picture = libvmaf::Picture::new(frame).map_err(liberr!("Failed to create picture"))?;
        Ok(Self { picture })
    }
}

pub trait PictureStream {
    fn next_pic(&mut self) -> Result<Option<Picture>, Error>;
}

pub enum AutoPictureStream {
    Y4m(y4m::PictureStream<BufReader<File>>),
    Gstreamer(gst::PictureStream<BufReader<File>>),
    Dynamic(Box<dyn PictureStream>),
}

impl PictureStream for AutoPictureStream {
    fn next_pic(&mut self) -> Result<Option<Picture>, Error> {
        match self {
            Self::Y4m(stream) => stream.next_pic(),
            Self::Gstreamer(stream) => stream.next_pic(),
            Self::Dynamic(stream) => stream.next_pic(),
        }
    }
}

impl AutoPictureStream {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        match path.extension() {
            None => Err(Error::UnsupportedVideoFile(format!("{:?}", path))),
            Some(ext) => match ext.to_str() {
                Some("y4m") => Ok(Self::Y4m(y4m::PictureStream::from_path(path)?)),
                _ => if gst::supported() {
                    Ok(Self::Gstreamer(gst::PictureStream::from_path(path, gst::PictureStreamOpts::default())?))
                } else {
                    Err(Error::UnsupportedVideoFile(format!("{:?}", path)))
                },
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

