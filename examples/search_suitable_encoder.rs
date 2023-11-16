use std::{fs, error, cmp::Ordering, time::{Instant, Duration}, path::{Path, PathBuf}, collections::VecDeque, fmt::{self, Display}, sync::{Mutex, Arc, atomic::{self, AtomicUsize}}, ops::Deref};
use clap::Parser;
use gstreamer as gst;
use gstreamer_video as gst_video;
use gst::prelude::*;
use path_to_unicode_filename::to_filename;
use tinytemplate::TinyTemplate;
use rand::distributions::{Uniform, Distribution as RandDistribution};
use rand::thread_rng;
use statrs::statistics::{self, Distribution as StatDistribution};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[arg()]
    path: PathBuf,

    #[arg()]
    target_vmaf: f64,

    #[arg(short, long, default_value_t=10)]
    fps: usize,

    #[arg(short, long, default_value_t=60)]
    segment_size: usize,

    #[arg(short, long, default_value="tmp")]
    tmp_dir: PathBuf,
}

#[derive(Debug)]
enum Error {
    CouldntGetVideoDuration(String),
    PathConversionError(String),
    PipelineSentErrorMessage(String),
    FailedToSeekPipeline(String),
    FailedToChangePipelineState(String),
    VmafError(vmaf::Error),
    CapsCouldntChangedInSingleStream,
    FailedToGetPipelineBus,
    CapsNotFound,
    UnexpectedEos,
    CapsStructureNotFound,
    TooManyCapsStructure(String),
    InvalidCapStructureName(String),
    FrameRateNotFound(String),
    PixelAspectRatioNotFound(String),
    CouldntGetVideoInfoFromCaps(String),
}

impl From<vmaf::Error> for Error {
    fn from(err: vmaf::Error) -> Self {
        Self::VmafError(err)
    }
}

impl From<vmaf::gst::Error> for Error {
    fn from(err: vmaf::gst::Error) -> Self {
        Self::VmafError(err.into())
    }
}

impl From<path_to_unicode_filename::Error> for Error {
    fn from(err: path_to_unicode_filename::Error) -> Self {
        Self::PathConversionError(format!("{:?}", err))
    }
}

impl From<&gst::message::Error> for Error {
    fn from(err: &gst::message::Error) -> Self {
        Self::PipelineSentErrorMessage(format!("{:?}", err))
    }
}

impl Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

unsafe impl Send for Error { }
unsafe impl Sync for Error { }
impl error::Error for Error { }

fn main() -> Result<(), Box<dyn error::Error>> {
    let encoder_candidates: Vec<(&str, isize, (isize, isize))> = vec![
        ("svtav1enc preset=10 parameters-string=keyint=300:crf={@root}", 35, (63, 1)),
        ("svtav1enc preset=12 parameters-string=keyint=300:crf={@root}", 35, (63, 1)),
        ("x265enc option-string=crf={@root} preset=medium key-int-max=300", 23, (51, 0)),
        ("x265enc option-string=crf={@root} preset=veryfast key-int-max=300", 23, (51, 0)),
    ];

    let mut tt = TinyTemplate::new();
    let args = Args::parse();

    fs::create_dir_all(&args.tmp_dir)?;

    for (enc_template, initial_value, (worst_value, best_value)) in encoder_candidates {
        tt.add_template(enc_template, enc_template)?;
        let value_worst_best_diff = best_value - worst_value;
        let value_range_length = value_worst_best_diff.abs();
        let value_direction = value_worst_best_diff / value_range_length;
        let mut worst_acceptable_value_result_pair = None;
        let mut cur_worst_value = worst_value;
        let mut cur_best_value = best_value;
        let mut value = initial_value;

        let value_result_pair = loop {
            let (ordering, result) = compare_with_target_vmaf(
                &args.path, &args.tmp_dir, args.target_vmaf,
                tt.render(enc_template, &value)?,
                CompareWithTargetVmafOpts {
                    segment_size: gst::ClockTime::from_seconds(args.segment_size as u64),
                    fps: Some(args.fps),
                    ..Default::default()
                },
            )?;
            match ordering {
                Ordering::Equal => break Some((value, result)),
                Ordering::Greater => {
                    // acceptable
                    match &worst_acceptable_value_result_pair {
                        None => {
                            worst_acceptable_value_result_pair = Some((value, result));
                        },
                        Some((old_value, _old_result)) => {
                            if value * value_direction < old_value * value_direction {
                                worst_acceptable_value_result_pair = Some((value, result));
                            }
                        },
                    }
                    cur_best_value = value;
                },
                Ordering::Less => {
                    cur_worst_value = value;
                }
            };
            if (cur_worst_value - cur_best_value).abs() <= 1 {
                break worst_acceptable_value_result_pair;
            }

            value = (cur_worst_value + cur_best_value) / 2;
        };

        if let Some((value, result)) = value_result_pair {
            println!("-- Encoder: {} --", tt.render(enc_template, &value)?);
            println!("Compressed rate: {}", (result.n_total_decoded_bytes as f64) / (result.n_total_original_bytes as f64));
            println!("Compressed bytes: {}", (result.n_total_original_bytes - result.n_total_decoded_bytes));
            println!("Approx process time: {:?}", result.total_process_time);
            println!("Compressed bytes per sec: {}", (result.n_total_original_bytes - result.n_total_decoded_bytes) as f64 / result.total_process_time.as_secs_f64());
        } else {
            println!("-- Not found: {} --", enc_template);
        };
    }

    Ok(())
}

#[derive(Debug)]
struct CompareWithTargetVmafOpts {
    segment_size: gst::ClockTime,
    fps: Option<usize>,
}

impl Default for CompareWithTargetVmafOpts {
    fn default() -> Self {
        Self {
            segment_size: gst::ClockTime::from_seconds(300),
            fps: None,
        }
    }
}

#[derive(Debug)]
struct ComparisonResult {
    n_total_original_bytes: usize,
    n_total_decoded_bytes: usize,
    total_process_time: Duration,
    scores: Vec<f64>,
}

impl ComparisonResult {
    fn new() -> Self {
        Self {
            n_total_original_bytes: 0,
            n_total_decoded_bytes: 0,
            total_process_time: Duration::ZERO,
            scores: Vec::new(),
        }
    }

    fn bootstrapped_score(&self) -> vmaf::BootstrappedScore {
        let mut rng = thread_rng();
        let n_bootstrap = 10000;
        let mut bootstrap_means = Vec::with_capacity(n_bootstrap);
        let uniform = Uniform::from(0..self.scores.len());
        for _ in 0..n_bootstrap {
            let scores: Vec<f64> = (0..self.scores.len()).map(|_| self.scores[uniform.sample(&mut rng)]).collect();
            let harmonic_mean = Self::harmonic_mean_impl(&scores);
            bootstrap_means.push(harmonic_mean);
        }

        let data = statistics::Data::new(bootstrap_means);
        let mean = data.mean().unwrap();
        let stddev = data.std_dev().unwrap();
        let stderr = stddev / (data.len() as f64).sqrt();
        let lower = mean - 1.96 * stderr;
        let higher = mean + 1.96 * stderr;

        vmaf::BootstrappedScore {
            bagging_score: mean,
            stddev: stddev,
            ci_p95_lo: lower,
            ci_p95_hi: higher,
        }
    }

    fn harmonic_mean(&self) -> f64 {
        Self::harmonic_mean_impl(&self.scores)
    }

    fn harmonic_mean_impl(scores: &Vec<f64>) -> f64 {
        let sum: f64 = scores.iter().map(|&x| x.recip()).sum();
        scores.len() as f64 / sum
    }
}

#[derive(Debug)]
struct SegmentIterator {
    segment_size: gst::ClockTime,
    remaining_segments: VecDeque<(gst::ClockTime, gst::ClockTime)>
}

impl Iterator for SegmentIterator {
    type Item = (gst::ClockTime, gst::ClockTime);

    fn next(&mut self) -> Option<Self::Item> {
        let Some((start, end)) = self.remaining_segments.pop_front() else {
            return None;
        };
        let duration = end - start;
        if duration <= self.segment_size {
            return Some((start, duration));
        }

        let segment_start = duration / 2 - self.segment_size / 2;
        let segment_end = segment_start + self.segment_size;
        self.remaining_segments.push_back((start, segment_start));
        self.remaining_segments.push_back((segment_end, end));
        Some((segment_start, self.segment_size))
    }
}

impl SegmentIterator {
    fn new(duration: gst::ClockTime, segment_size: gst::ClockTime) -> Self {
        Self {
            segment_size,
            remaining_segments: [(gst::ClockTime::ZERO, duration)].into(),
        }
    }
}

fn compare_with_target_vmaf(path: impl AsRef<Path>, save_dir: impl AsRef<Path>, target_vmaf: f64, enc: impl AsRef<str>, opts: CompareWithTargetVmafOpts) -> Result<(Ordering, ComparisonResult), Error> {
    let path = path.as_ref();
    let save_dir = save_dir.as_ref();
    let enc = enc.as_ref();
    let mut ordering = None;
    let mut result = ComparisonResult::new();

    let video_duration = video_duration(path)?;
    println!("Video duration: {}", video_duration);

    for (start, duration) in SegmentIterator::new(video_duration, opts.segment_size) {
        let (raw_path, n_original_bytes, decoding_time, caps) = create_segment_raw_file(path, &save_dir, start, duration)?;
        println!("Original video stream size: {}", n_original_bytes);
        println!("Decoding time: {:?}", decoding_time);
        println!("Caps: {}", caps);

        // decode the raw file
        let (decoded_path, n_decoded_bytes, encoding_time) = encode_raw_file(&raw_path, &save_dir, &enc, caps)?;
        println!("Encoding time: {:?}", encoding_time);
        println!("Decoded video stream size: {}", n_decoded_bytes);

        // collect vmaf score for segment
        let fps = opts.fps;
        let mut ref_stream = vmaf::gst::PictureStream::from_path(&raw_path, vmaf::gst::PictureStreamOpts { fps, ..Default::default() })?;
        let mut dist_stream = vmaf::gst::PictureStream::from_path(decoded_path, vmaf::gst::PictureStreamOpts { fps, ..Default::default() })?;
        let score = vmaf::collect_typed_score_from_stream_pair::<vmaf::BootstrappedScore>(&mut ref_stream, &mut dist_stream, vmaf::CollectScoreOpts::default())?;

        result.n_total_original_bytes += n_original_bytes;
        result.n_total_decoded_bytes += n_decoded_bytes;
        result.total_process_time += decoding_time + encoding_time;
        result.scores.push(score.bagging_score);

        if 3 <= result.scores.len() {
            let bootstrapped_score = result.bootstrapped_score();
            if target_vmaf <= bootstrapped_score.ci_p95_lo {
                ordering = Some(Ordering::Greater);
                break;
            } else if bootstrapped_score.ci_p95_hi <= target_vmaf {
                ordering = Some(Ordering::Greater);
                break;
            } else if bootstrapped_score.ci_p95_hi - bootstrapped_score.ci_p95_lo < 1.0 {
                ordering = Some(Ordering::Equal);
                break;
            } else {
                continue;
            }
        }
    }

    Ok(if let Some(ordering) = ordering {
        (ordering, result)
    } else {
        (target_vmaf.total_cmp(&result.harmonic_mean()), result)
    })
}

#[derive(Debug)]
struct ProgressBar {
    pb: indicatif::ProgressBar,
}

impl ProgressBar {
    fn new(message: impl Into<String>) -> Self {
        let message = message.into();
        let pb = indicatif::ProgressBar::new(0);
        let style = indicatif::ProgressStyle::with_template("{message} {spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% (ETA: {eta})").expect("hardcoded")
            .with_key("eta", |state: &indicatif::ProgressState, w: &mut dyn fmt::Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
            .with_key("message", move |_state: &indicatif::ProgressState, w: &mut dyn fmt::Write| write!(w, "{}", message).unwrap())
            .progress_chars("#>-");
        pb.set_style(style);
        Self { pb }
    }

    fn update(&self, current: u64, total: u64) {
        self.pb.set_position(current);
        self.pb.set_length(total);
    }
}

impl Drop for ProgressBar {
    fn drop(&mut self) {
        self.pb.finish();
    }
}

fn video_duration(path: impl AsRef<Path>) -> Result<gst::ClockTime, Error> {
    let path = path.as_ref();
    let pipeline = make_pipeline("filesrc name=src ! decodebin force-sw-decoders=true ! fakesink sync=false");

    let filesrc = element_by_name(&*pipeline, "src");
    filesrc.set_property("location", path);

    pipeline_wait_paused(&*pipeline)?;

    let Some(duration) = pipeline.query_duration::<gst::ClockTime>() else {
        return Err(Error::CouldntGetVideoDuration(format!("{}", path.display())));
    };

    Ok(duration)
}

fn create_segment_raw_file(path: impl AsRef<Path>, save_dir: impl AsRef<Path>, start: gst::ClockTime, duration: gst::ClockTime) -> Result<(PathBuf, usize, Duration, gst::Caps), Error> {
    let path = path.as_ref();
    let save_dir = save_dir.as_ref();
    let pipeline = make_pipeline("filesrc name=src ! parsebin ! decodebin force-sw-decoders=true name=dec ! capsfilter caps=video/x-raw ! progressreport update-freq=1 silent=true name=progress ! filesink sync=false name=sink");

    let filesrc = element_by_name(&*pipeline, "src");
    filesrc.set_property("location", path);

    let filesink = element_by_name(&*pipeline, "sink");
    let sink_path = save_dir.join(format!("{}.yuv", to_filename(path)?));
    filesink.set_property("location", &sink_path);
    let sink_pad = filesink.static_pad("sink").expect("hardcoded pad name");
    let caps = probe_unique_caps(&sink_pad);

    println!("Yuv path: {}", sink_path.display());

    let decodebin = element_by_name(&pipeline, "dec");
    let sink_pad = decodebin.static_pad("sink").expect("hardcoded pad name");

    let n_total_buffer_bytes = probe_buffer_n_bytes(&sink_pad);

    let start_time = Instant::now();
    pipeline_wait_paused(&*pipeline)?;

    pipeline.seek(1.0, gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE, gst::SeekType::Set, start, gst::SeekType::Set, start + duration).map_err(|err| Error::FailedToSeekPipeline(format!("{:?}", err)))?;

    let pb = ProgressBar::new(&format!("Creating raw file {} ({})", start, duration));
    playback_pipeline_to_eos(&*pipeline, pb)?;

    let caps = Arc::into_inner(caps).unwrap();
    let caps = caps.into_inner().unwrap();
    let Some(caps) = caps else {
        return Err(Error::CapsNotFound);
    };
    let caps = caps?;

    Ok((sink_path, n_total_buffer_bytes.load(atomic::Ordering::SeqCst), start_time.elapsed(), caps))
}

fn encode_raw_file(path: impl AsRef<Path>, save_dir: impl AsRef<Path>, enc: impl AsRef<str>, caps: gst::Caps) -> Result<(PathBuf, usize, Duration), Error> {
    let path = path.as_ref();
    let save_dir = save_dir.as_ref();
    let enc = enc.as_ref();
    let pipeline = make_pipeline(format!("filesrc name=src ! rawvideoparse name=parse ! progressreport update-freq=1 silent=true name=progress ! {} name=enc ! matroskamux name=mux ! filesink sync=false name=sink", enc));

    let filesrc = element_by_name(&*pipeline, "src");
    filesrc.set_property("location", path);

    let rawvideoparse = element_by_name(&*pipeline, "parse");
    let Some(caps_structure) = caps.structure(0) else {
        return Err(Error::CapsStructureNotFound);
    };
    if caps.structure(1).is_some() {
        return Err(Error::TooManyCapsStructure(format!("{:?}", caps.structure(1))));
    };
    if caps_structure.name() != "video/x-raw" {
        return Err(Error::InvalidCapStructureName(format!("{}", caps_structure.name())));
    };
    let info = match gst_video::VideoInfo::from_caps(&caps) {
        Ok(info) => info,
        Err(err) => return Err(Error::CouldntGetVideoInfoFromCaps(format!("{:?}", err))),
    };
    rawvideoparse.set_property("format", info.format());
    rawvideoparse.set_property("colorimetry", info.colorimetry().to_string());
    match caps_structure.value("framerate") {
        Ok(frame_rate) => rawvideoparse.set_property("framerate", frame_rate),
        Err(err) => return Err(Error::FrameRateNotFound(format!("{:?}", err))),
    };
    rawvideoparse.set_property("width", info.width() as i32);
    rawvideoparse.set_property("height", info.height() as i32);
    rawvideoparse.set_property("interlaced", info.is_interlaced());
    match caps_structure.value("pixel-aspect-ratio") {
        Ok(pixel_aspect_ratio) => rawvideoparse.set_property("pixel-aspect-ratio", pixel_aspect_ratio),
        Err(err) => return Err(Error::PixelAspectRatioNotFound(format!("{:?}", err))),
    };

    let filesink = element_by_name(&*pipeline, "sink");
    let sink_path = save_dir.join(format!("{}.mkv", to_filename(path)?));
    filesink.set_property("location", &sink_path);

    println!("Encoded file path: {}", sink_path.display());

    let matroskamux = element_by_name(&*pipeline, "enc");
    let src_pad = matroskamux.static_pad("src").expect("hardcoded pad name");

    let n_total_buffer_bytes = probe_buffer_n_bytes(&src_pad);

    let start_time = Instant::now();

    let pb = ProgressBar::new("Encoding raw file");
    playback_pipeline_to_eos(&*pipeline, pb)?;

    Ok((sink_path, n_total_buffer_bytes.load(atomic::Ordering::SeqCst), start_time.elapsed()))
}

#[derive(Debug)]
struct PipelineGuard(gst::Pipeline);

impl Drop for PipelineGuard {
    fn drop(&mut self) {
        let _ = self.0.set_state(gst::State::Null);
    }
}

impl Deref for PipelineGuard {
    type Target = gst::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn make_pipeline(desc: impl AsRef<str>) -> PipelineGuard {
    vmaf::gst::init();
    PipelineGuard(gst::parse_launch(desc.as_ref()).expect("must be valid pipeline description").downcast::<gst::Pipeline>().expect("must be able to cast to pipeline"))
}

fn pipeline_wait_paused(pipeline: &gst::Pipeline) -> Result<(), Error> {
    let bus = pipeline.bus().ok_or(Error::FailedToGetPipelineBus)?;
    pipeline.set_state(gst::State::Paused).map_err(|err| Error::FailedToChangePipelineState(format!("{:?}", err)))?;
    for message in bus.iter_timed(None) {
        match message.view() {
            gst::MessageView::Error(err) => return Err(Error::from(err)),
            gst::MessageView::Eos(_) => return Err(Error::UnexpectedEos),
            gst::MessageView::StateChanged(state_changed) => {
                if state_changed.src() == Some(pipeline.upcast_ref()) {
                    if state_changed.current() == gst::State::Paused && state_changed.pending() == gst::State::VoidPending {
                        break;
                    }
                }
            },
            _ => (),
        }
    }
    Ok(())
}

fn playback_pipeline_to_eos(pipeline: &gst::Pipeline, pb: ProgressBar) -> Result<(), Error> {
    let bus = pipeline.bus().ok_or(Error::FailedToGetPipelineBus)?;
    pipeline.set_state(gst::State::Playing).map_err(|err| Error::FailedToChangePipelineState(format!("{:?}", err)))?;

    for message in bus.iter_timed(None) {
        match message.view() {
            gst::MessageView::Error(err) => return Err(Error::from(err)),
            gst::MessageView::Eos(_) => break,
            gst::MessageView::Element(element) => {
                if element.src().map(|el| el.name() == "progress").unwrap_or(false) {
                    if let Some(structure) = message.structure() {
                        if let (Ok(current), Ok(total)) = (structure.get::<i64>("current"), structure.get::<i64>("total")) {
                            pb.update(current as u64, total as u64);
                        }
                    }
                }
            },
            _ => (),
        }
    }
    Ok(())
}

fn element_by_name(pipeline: &gst::Pipeline, name: &str) -> gst::Element {
    pipeline.by_name(name).expect(&format!("element must be added to pipeline"))
}

fn probe_buffer_n_bytes(pad: &gst::Pad) -> Arc<AtomicUsize> {
    let n_total_buffer_bytes = Arc::new(AtomicUsize::new(0));
    let n_total_buffer_bytes_weak = Arc::downgrade(&n_total_buffer_bytes);

    pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, info| {
        let Some(n_total_buffer_bytes) = n_total_buffer_bytes_weak.upgrade() else {
            return gst::PadProbeReturn::Ok;
        };

        let buffer = match &info.data {
            Some(gst::PadProbeData::Buffer(buffer)) => buffer,
            _ => panic!("pad probe info must have buffer"),
        };
        let n_buffer_bytes = buffer.size();

        let mut old_n_bytes = n_total_buffer_bytes.load(atomic::Ordering::Relaxed);
        loop {
            let new_n_bytes = old_n_bytes + n_buffer_bytes;
            match n_total_buffer_bytes.compare_exchange_weak(old_n_bytes, new_n_bytes, atomic::Ordering::SeqCst, atomic::Ordering::Relaxed) {
                Ok(_) => break,
                Err(cur_n_bytes) => {
                    old_n_bytes = cur_n_bytes;
                },
            }
        }
        gst::PadProbeReturn::Ok
    });

    n_total_buffer_bytes
}

fn probe_unique_caps(pad: &gst::Pad) -> Arc<Mutex<Option<Result<gst::Caps, Error>>>> {
    let caps = Arc::new(Mutex::new(None));
    let caps_weak = Arc::downgrade(&caps);

    pad.add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
        let Some(caps) = caps_weak.upgrade() else {
            return gst::PadProbeReturn::Ok;
        };

        match &info.data {
            Some(gst::PadProbeData::Event(event)) => match event.view() { // TODO as_ref?
                gst::EventView::Caps(caps_event) => {
                    let new_caps = caps_event.caps_owned();
                    let mut caps = caps.lock().unwrap();
                    if caps.is_some() {
                        *caps = Some(Err(Error::CapsCouldntChangedInSingleStream))
                    } else {
                        *caps = Some(Ok(new_caps));
                    }
                },
                _ => (),
            },
            _ => panic!("pad probe info must have event"),
        };
        gst::PadProbeReturn::Ok
    });

    caps
}

