use std::{io, env, fs, error, ffi::OsStr, cmp::Ordering, time::{Instant, Duration}, path::{Path, PathBuf}, collections::VecDeque, fmt::{self, Display}, sync::{Mutex, Weak, Arc, atomic::{self, AtomicUsize}}, ops::Deref};
use clap::Parser;
use gstreamer as gst;
use gstreamer_video as gst_video;
use gst::prelude::*;
use vmaf::PictureStream;
use path_to_unicode_filename::to_filename;
use tinytemplate::TinyTemplate;
use rand::distributions::{Uniform, Distribution as RandDistribution};
use rand::thread_rng;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[arg()]
    path: PathBuf,

    #[arg()]
    target_vmaf: f64,

    #[arg(short, long, default_value_t=30)]
    segment_size: usize,

    #[arg(short, long, default_value="output.csv")]
    output_path: PathBuf,

    #[arg(short, long, default_value="tmp")]
    tmp_dir: PathBuf,
}

#[derive(Debug)]
enum Error {
    MultipleVideoInfoInSegment(gst_video::VideoInfo, gst_video::VideoInfo), // recoverable

    CouldntGetVideoDuration(String),
    PathConversionError(String),
    PipelineSentErrorMessage(String),
    FailedToSeekPipeline(String),
    FailedToChangePipelineState(String),
    VmafError(vmaf::Error),
    CouldntGetVideoInfoFromCaps(String),
    FailedToGetPipelineBus,
    VideoInfoNotFound,
    UnexpectedEos,
    RawFileStreamShorter,
    EncodedFileStreamShorter,
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
        Self::PipelineSentErrorMessage(format!("{}", err))
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

fn main() {
    env::set_var("SVT_LOG", "2"); // SVT_LOG=warn
    env_logger::init();
    
    let Args { target_vmaf, path, tmp_dir, segment_size, output_path } = Args::parse();

    fs::create_dir_all(&tmp_dir).unwrap();

    let mut csv_writer = csv::Writer::from_path(&output_path).unwrap();

    if path.is_dir() {
        for entry in fs::read_dir(&path).unwrap() {
            let entry = entry.unwrap();
            let sub_path = entry.path();
            process_file(&sub_path, target_vmaf, segment_size, &mut csv_writer, &tmp_dir);
        }
    } else {
        process_file(&path, target_vmaf, segment_size, &mut csv_writer, &tmp_dir);
    }

    fn process_file<W: io::Write>(path: impl AsRef<Path>, target_vmaf: f64, segment_size: usize, csv_writer: &mut csv::Writer<W>, tmp_dir: impl AsRef<Path>) {
        let path = path.as_ref();
        let video_info = match video_info(&path) {
            Err(err) => {
                writre_csv(csv_writer, path, "Error", Some(format!("CouldntGetVideoInfo:{:?}", err)), None, None, None, None, None);
                return;
            },
            Ok(video_info) => video_info,
        };

        let width = video_info.width() as usize;
        let height = video_info.height() as usize;

        let results = match search_encoders(&path, target_vmaf, segment_size, &tmp_dir) {
            Err(err) => {
                writre_csv(csv_writer, path, "Error", Some(format!("ProcessError:{:?}", err)), Some(width), Some(height), None, None, None);
                return;
            },
            Ok(results) => results,
        };
        for (enc, result) in results {
            if let Some((value, result)) = result {
                writre_csv(csv_writer, path, "Ok", None, Some(width), Some(height), Some(enc), Some(value), Some(result));
            } else {
                writre_csv(csv_writer, path, "NotFound", None, Some(width), Some(height), Some(enc), None, None);
            }
        }

        for entry in fs::read_dir(&tmp_dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            match path.extension().and_then(OsStr::to_str) {
                Some("yuv") | Some("mkv") => fs::remove_file(path).unwrap(),
                _ => (),
            }
        }
    }

    // TODO rough
    fn writre_csv<W: io::Write>(
        csv_writer: &mut csv::Writer<W>,
        path: &Path,
        kind: &str,
        err_message: Option<String>,
        width: Option<usize>,
        height: Option<usize>,
        enc: Option<String>,
        value: Option<isize>,
        result: Option<ComparisonResult>
    ) {
        let mut record = vec![
            format!("{}", path.display()),
            String::from(kind),
            err_message.unwrap_or(String::from("")),
            width.map(|u| u.to_string()).unwrap_or(String::from("")),
            height.map(|u| u.to_string()).unwrap_or(String::from("")),
            enc.unwrap_or(String::from("")),
            value.map(|i| i.to_string()).unwrap_or(String::from("")),
        ];
        if let Some(result) = result {
            let bootstrapped_score = result.bootstrapped_score();

            record.extend_from_slice(&[
                result.n_total_original_bytes.to_string(),
                result.n_total_decoded_bytes.to_string(),
                result.total_process_time.as_secs_f64().to_string(),
                bootstrapped_score.bagging_score.to_string(),
                bootstrapped_score.ci_p95_lo.to_string(),
                bootstrapped_score.ci_p95_hi.to_string(),
                bootstrapped_score.stddev.to_string(),
            ]);
        } else {
            record.extend_from_slice(&[
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
                String::new(),
            ]);
        }
        log::info!("Write score to csv: {:?}", record);
        csv_writer.write_record(record).unwrap();
        csv_writer.flush().unwrap();
    }
}

fn search_encoders(path: impl AsRef<Path>, target_vmaf: f64, segment_size: usize, tmp_dir: impl AsRef<Path>) -> Result<Vec<(String, Option<(isize, ComparisonResult)>)>, Box<dyn error::Error>> {
    let path = path.as_ref();
    let encoder_candidates: Vec<(&str, usize, (usize, usize))> = vec![
        // ("x265enc option-string=crf={@root} speed-preset=medium key-int-max=300 ! h265parse", 23, (51, 0)),
        // ("x265enc option-string=crf={@root} speed-preset=veryfast key-int-max=300 ! h265parse", 23, (51, 0)),
        ("svtav1enc preset=10 parameters-string=keyint=10s:crf={@root}", 35, (55, 20)),
        // ("svtav1enc preset=12 parameters-string=keyint=10s:crf={@root}", 35, (63, 1)),
    ];

    let mut tt = TinyTemplate::new();
    let mut result = Vec::new();

    for (enc_template, initial_value, (worst_value, best_value)) in encoder_candidates {
        log::info!("== Start encoder besec search ==");
        log::info!("Path: {}", path.display());
        log::info!("Encoder: {}", enc_template);
        log::info!("================================");

        let initial_value = initial_value as isize;
        let worst_value = worst_value as isize;
        let best_value = best_value as isize;

        tt.add_template(enc_template, enc_template)?;
        let value_worst_best_diff = best_value - worst_value;
        let value_range_length = value_worst_best_diff.abs();
        let value_direction = value_worst_best_diff / value_range_length;
        let mut worst_acceptable_value_result_pair = None;
        let mut cur_worst_value = worst_value;
        let mut cur_best_value = best_value;
        let mut value = initial_value;

        let value_result_pair = loop {
            log::info!("Current value: {}", value);
            let (ordering, result) = compare_with_target_vmaf(
                &path, &tmp_dir, target_vmaf,
                tt.render(enc_template, &value)?,
                CompareWithTargetVmafOpts {
                    segment_size: gst::ClockTime::from_seconds(segment_size as u64),
                    ..Default::default()
                },
            )?;
            log::trace!("Segment vmaf: {:?} {:?}", ordering, result);
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
                    cur_worst_value = value + value_direction;
                }
            };


            if cur_worst_value == cur_best_value {
                break worst_acceptable_value_result_pair;
            }


            if value_direction < 0 {
                value = (cur_worst_value + cur_best_value) / 2 + (cur_worst_value + cur_best_value) % 2;
            } else {
                value = (cur_worst_value + cur_best_value) / 2;
            }
            log::info!("Next value: {} ({} - {})", value, cur_worst_value, cur_best_value);
        };

        result.push((enc_template.to_string(), value_result_pair));
    }

    Ok(result)
}

#[derive(Debug)]
struct CompareWithTargetVmafOpts {
    segment_size: gst::ClockTime,
}

impl Default for CompareWithTargetVmafOpts {
    fn default() -> Self {
        Self {
            segment_size: gst::ClockTime::from_seconds(300),
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
        let n_bootstrap = 1000;
        let mut bootstrap_means = Vec::with_capacity(n_bootstrap);
        let uniform = Uniform::from(0..self.scores.len());
        for _ in 0..n_bootstrap {
            let scores: Vec<f64> = (0..self.scores.len()).map(|_| self.scores[uniform.sample(&mut rng)]).collect();
            let harmonic_mean = Self::harmonic_mean_impl(&scores);
            bootstrap_means.push(harmonic_mean);
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = bootstrap_means.iter().sum::<f64>() / bootstrap_means.len() as f64;
        let lower = bootstrap_means[(bootstrap_means.len() as f64 * 0.025) as usize];
        let higher = bootstrap_means[(bootstrap_means.len() as f64 * 0.975) as usize];

        vmaf::BootstrappedScore {
            bagging_score: mean,
            stddev: 0.0,
            ci_p95_lo: lower,
            ci_p95_hi: higher,
        }
    }

    fn harmonic_mean(&self) -> f64 {
        Self::harmonic_mean_impl(&self.scores)
    }

    // the same way as libvmaf
    fn harmonic_mean_impl(scores: &Vec<f64>) -> f64 {
        let sum: f64 = scores.iter().map(|&x| 1.0 / (x + 1.0)).sum();
        scores.len() as f64 / sum - 1.0
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

        let segment_start = start + duration / 2 - self.segment_size / 2;
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
    log::trace!("Video duration: {}", video_duration);

    let mut segment_count = 0;

    for (start, duration) in SegmentIterator::new(video_duration, opts.segment_size) {
        log::info!("Start calculating vmaf for segment: {} ({})", start, duration);
        log::trace!("Start segment: {} - {}", start, duration);
        let (raw_path, n_original_bytes, decoding_time, video_info) = match create_segment_raw_file(path, &save_dir, start, duration) {
            Ok(raw_file_info) => raw_file_info,
            Err(Error::MultipleVideoInfoInSegment(video_info, second_video_info)) => {
                log::warn!("Multiple video info found. Skip the segment: {:?} != {:?}", video_info, second_video_info);
                continue;
            },
            Err(err) => return Err(err),
        };
        log::trace!("Original video stream size: {}", n_original_bytes);
        log::trace!("Decoding time: {:?}", decoding_time);
        log::trace!("Video info: {:?}", video_info);

        // decode the raw file
        let (decoded_path, n_decoded_bytes, encoding_time) = encode_raw_file(&raw_path, &save_dir, &enc, &video_info)?;
        log::trace!("Encoding time: {:?}", encoding_time);
        log::trace!("Decoded video stream size: {}", n_decoded_bytes);

        // collect vmaf score for segment
        let mut ref_stream = vmaf::gst::PictureStream::from_raw_path(&raw_path, &video_info, vmaf::gst::PictureStreamOpts::default())?;
        let mut dist_stream = vmaf::gst::PictureStream::from_path(decoded_path, vmaf::gst::PictureStreamOpts::default())?;

        let mut score_collector = vmaf::ScoreCollector::<vmaf::BootstrappedScore>::new(vmaf::Model::default(), vmaf::ScoreCollectorOpts {
            n_threads: num_cpus::get(),
            ..Default::default()
        })?;

        {
            let pb = ProgressBar::new("Comparing streams");
            let mut progress_updated_at = Instant::now();
            loop {
                let (ref_pic, dist_pic) = match (ref_stream.next_pic()?, dist_stream.next_pic()?) {
                    (Some(ref_pic), Some(dist_pic)) => (ref_pic, dist_pic),
                    (None, None) => break,
                    (None, Some(_)) => return Err(Error::RawFileStreamShorter),
                    (Some(_), None) => return Err(Error::EncodedFileStreamShorter),
                };
                if Duration::from_millis(100) < progress_updated_at.elapsed() {
                    if let (Some(pos), Some(dur)) = (dist_stream.position(), dist_stream.duration()) {
                        pb.update(pos.nseconds(), dur.nseconds());
                        progress_updated_at = Instant::now();
                    }
                }
                score_collector.read_pictures(ref_pic, dist_pic)?;
            };
            // drop progress bar
        }


        result.n_total_original_bytes += n_original_bytes;
        result.n_total_decoded_bytes += n_decoded_bytes;
        result.total_process_time += decoding_time + encoding_time;
        for score_index in 0..score_collector.n_scores() {
            let score = score_collector.score_at_index(score_index)?;
            result.scores.push(score.bagging_score);
        }

        let bootstrapped_score = result.bootstrapped_score();

        segment_count += 1;

        log::info!("Encoding rote: {:.2} MB -> {:.2} MB ({:.2} %) [n={}][seg={}]",
            result.n_total_original_bytes as f64 / 1000_000.0,
            result.n_total_decoded_bytes as f64 / 1000_000.0,
            result.n_total_decoded_bytes as f64 / result.n_total_original_bytes as f64 * 100.0,
            result.scores.len(),
            segment_count,
        );

        log::info!("Current vmaf score: {:.2} ({:.2} - {:.2}) [n={}][seg={}]",
            bootstrapped_score.bagging_score,
            bootstrapped_score.ci_p95_lo,
            bootstrapped_score.ci_p95_hi,
            result.scores.len(),
            segment_count,
        );

        if 3 <= segment_count {
            log::trace!("Enc {} Score: {:?}", &enc, bootstrapped_score);
            if target_vmaf <= bootstrapped_score.ci_p95_lo {
                if bootstrapped_score.ci_p95_hi < target_vmaf + 0.5 {
                    ordering = Some(Ordering::Equal);
                } else {
                    ordering = Some(Ordering::Greater);
                }
                break;
            } else if bootstrapped_score.ci_p95_hi <= target_vmaf {
                ordering = Some(Ordering::Less);
                break;
            } else if bootstrapped_score.ci_p95_hi - bootstrapped_score.ci_p95_lo < 0.5 {
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

// TODO only rough impl
fn video_info(path: impl AsRef<Path>) -> Result<gst_video::VideoInfo, Error> {
    let path = path.as_ref();
    let pipeline = make_pipeline("filesrc name=src ! decodebin force-sw-decoders=true ! fakesink sync=false name=sink");

    let filesrc = element_by_name(&*pipeline, "src");
    filesrc.set_property("location", path);

    pipeline_wait_paused(&*pipeline)?;

    let fakesink = element_by_name(&*pipeline, "sink");
    let sink_pad = fakesink.static_pad("sink").expect("hardcoded");
    let caps = sink_pad.caps().expect("sink of paused pipeline must have caps");
    let video_info = gst_video::VideoInfo::from_caps(&caps).map_err(|err| Error::CouldntGetVideoInfoFromCaps(format!("{:?}", err)))?;

    Ok(video_info)
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

fn create_segment_raw_file(path: impl AsRef<Path>, save_dir: impl AsRef<Path>, start: gst::ClockTime, duration: gst::ClockTime) -> Result<(PathBuf, usize, Duration, gst_video::VideoInfo), Error> {
    let path = path.as_ref();
    let save_dir = save_dir.as_ref();
    let pipeline = make_pipeline("filesrc name=src ! parsebin ! decodebin force-sw-decoders=true name=dec ! capsfilter caps=video/x-raw ! progressreport update-freq=1 silent=true name=progress ! filesink sync=false name=sink");

    let filesrc = element_by_name(&*pipeline, "src");
    filesrc.set_property("location", path);

    let filesink = element_by_name(&*pipeline, "sink");
    let sink_path = save_dir.join(format!("{}.yuv", to_filename(path)?));
    filesink.set_property("location", &sink_path);
    log::trace!("Yuv path: {}", sink_path.display());
    let sink_pad = filesink.static_pad("sink").expect("hardcoded pad name");
    let video_info_data = probe_video_info(&sink_pad);

    let decodebin = element_by_name(&pipeline, "dec");
    let sink_pad = decodebin.static_pad("sink").expect("hardcoded pad name");

    let n_total_buffer_bytes = probe_buffer_n_bytes(&sink_pad, start, Some(duration));

    let start_time = Instant::now();

    pipeline_wait_paused(&*pipeline)?;
    pipeline.seek(1.0, gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE, gst::SeekType::Set, start, gst::SeekType::Set, start + duration).map_err(|err| Error::FailedToSeekPipeline(format!("{:?}", err)))?;

    let pb = ProgressBar::new(&format!("Creating raw file {} ({})", start, duration));
    playback_pipeline_to_eos(&*pipeline, pb)?;

    let video_info_data = Arc::into_inner(video_info_data).unwrap();
    let video_info_data = video_info_data.into_inner().unwrap();
    let video_info = video_info_data.into_video_info()?;

    Ok((sink_path, n_total_buffer_bytes.load(atomic::Ordering::SeqCst), start_time.elapsed(), video_info))
}

fn encode_raw_file(path: impl AsRef<Path>, save_dir: impl AsRef<Path>, enc: impl AsRef<str>, video_info: &gst_video::VideoInfo) -> Result<(PathBuf, usize, Duration), Error> {
    let path = path.as_ref();
    let save_dir = save_dir.as_ref();
    let enc = enc.as_ref();
    let pipeline = make_pipeline(format!("filesrc name=src ! rawvideoparse name=parse ! progressreport update-freq=1 silent=true name=progress ! {} name=enc ! matroskamux name=mux ! filesink sync=false name=sink", enc));

    let filesrc = element_by_name(&*pipeline, "src");
    filesrc.set_property("location", path);

    let rawvideoparse = element_by_name(&*pipeline, "parse");
    rawvideoparse.set_property("width", video_info.width() as i32);
    rawvideoparse.set_property("height", video_info.height() as i32);
    rawvideoparse.set_property("format", video_info.format());
    let framerate = video_info.fps();
    if framerate.numer() != 0 {
        rawvideoparse.set_property("framerate", framerate);
    }
    rawvideoparse.set_property("pixel-aspect-ratio", video_info.par());
    if video_info.is_interlaced() {
        rawvideoparse.set_property("interlaced", true);
        rawvideoparse.set_property("top-field-first", video_info.field_order() == gst_video::VideoFieldOrder::TopFieldFirst);
    } else {
        rawvideoparse.set_property("interlaced", false);
    }
    rawvideoparse.set_property("plane-strides", gst::Array::new(video_info.stride().into_iter().map(|n| *n as i32)).to_value());
    rawvideoparse.set_property("plane-offsets", gst::Array::new(video_info.offset().into_iter().map(|n| *n as i32)).to_value());
    rawvideoparse.set_property("frame-size", video_info.size() as u32);
    rawvideoparse.set_property("colorimetry", video_info.colorimetry().to_string());

    // XXX chroma site cannot be set to rawvideoparse
    // is it ignorable?

    let filesink = element_by_name(&*pipeline, "sink");
    let sink_path = save_dir.join(format!("{}.mkv", to_filename(path)?));
    filesink.set_property("location", &sink_path);

    log::trace!("Encoded file path: {}", sink_path.display());

    let matroskamux = element_by_name(&*pipeline, "enc");
    let src_pad = matroskamux.static_pad("src").expect("hardcoded pad name");

    let n_total_buffer_bytes = probe_buffer_n_bytes(&src_pad, gst::ClockTime::ZERO, None);

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

fn probe_buffer_n_bytes(pad: &gst::Pad, start: gst::ClockTime, duration: Option<gst::ClockTime>) -> Arc<AtomicUsize> {
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

        let Some(pts) = buffer.pts() else {
            return gst::PadProbeReturn::Ok;
        };

        if pts < start {
            return gst::PadProbeReturn::Ok;
        }

        if let Some(duration) = duration {
            if start + duration <= pts {
                return gst::PadProbeReturn::Ok;
            }
        }

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

#[derive(Debug)]
struct VideoInfoData {
    video_info: Option<gst_video::VideoInfo>,
    first_buffer_arrived: bool,
    error: Option<Error>,
}

impl VideoInfoData {
    fn new() -> Self {
        Self {
            video_info: None,
            first_buffer_arrived: false,
            error: None,
        }
    }

    fn into_video_info(self) -> Result<gst_video::VideoInfo, Error> {
        if let Some(err) = self.error {
            Err(err)
        } else if let Some(video_info) = self.video_info {
            Ok(video_info)
        } else {
            Err(Error::VideoInfoNotFound)
        }
    }
}

fn probe_video_info(pad: &gst::Pad) -> Arc<Mutex<VideoInfoData>> {
    let video_info_data: Arc<Mutex<VideoInfoData>> = Arc::new(Mutex::new(VideoInfoData::new()));
    let video_info_data_weak_for_flush_probe = Arc::downgrade(&video_info_data);
    let video_info_data_weak_for_event_probe = Arc::downgrade(&video_info_data);

    fn detect_first_buffer(pad: &gst::Pad, video_info_data_weak: Weak<Mutex<VideoInfoData>>) {
        pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, _info| {
            log::trace!("Filesink detect first buffer in segment");
            let Some(video_info_data) = video_info_data_weak.upgrade() else {
                return gst::PadProbeReturn::Remove;
            };
            let video_info_data = video_info_data.deref();
            let mut video_info_data = video_info_data.lock().unwrap();
            video_info_data.first_buffer_arrived = true;
            return gst::PadProbeReturn::Remove;
        });
    }

    detect_first_buffer(pad, Arc::downgrade(&video_info_data));

    pad.add_probe(gst::PadProbeType::EVENT_FLUSH, move |pad, info| {
        let Some(video_info_data) = video_info_data_weak_for_flush_probe.upgrade() else {
            return gst::PadProbeReturn::Remove;
        };

        match &info.data {
            Some(gst::PadProbeData::Event(event)) => match event.view() {
                gst::EventView::FlushStop(_) => {
                    log::trace!("Filesink FLUSH_STOP event recieved");
                    {
                        let video_info_data = video_info_data.deref();
                        let mut video_info_data = video_info_data.lock().unwrap();
                        video_info_data.first_buffer_arrived = false; // not seen buffer yet in new segment
                    }

                    detect_first_buffer(pad, Arc::downgrade(&video_info_data));
                },
                _ => (),
            },
            _ => unreachable!(),
        };
        gst::PadProbeReturn::Ok
    });

    pad.add_probe(gst::PadProbeType::EVENT_DOWNSTREAM, move |_pad, info| {
        let Some(video_info_data) = video_info_data_weak_for_event_probe.upgrade() else {
            return gst::PadProbeReturn::Remove;
        };

        match &info.data {
            Some(gst::PadProbeData::Event(event)) => match event.view() {
                gst::EventView::Caps(caps) => {
                    log::trace!("Filesink CAPS event recieved");
                    let caps = caps.caps_owned();
                    let video_info_data = video_info_data.deref();
                    let mut video_info_data = video_info_data.lock().unwrap();
                    match gst_video::VideoInfo::from_caps(&caps) {
                        Ok(new_video_info) => {
                            // if first buffer arrived, not allowed change video info
                            if video_info_data.first_buffer_arrived {
                                if let Some(old_video_info) = &video_info_data.video_info {
                                    if new_video_info != *old_video_info {
                                        video_info_data.error = Some(Error::MultipleVideoInfoInSegment(new_video_info.clone(), old_video_info.clone()));
                                    }
                                }
                            }
                            if video_info_data.error.is_none() {
                                video_info_data.video_info = Some(new_video_info);
                            }
                        },
                        Err(err) => {
                            video_info_data.error = Some(Error::CouldntGetVideoInfoFromCaps(format!("{} ({})", caps, err)));
                        },
                    };
                },
                _ => (),
            },
            _ => panic!("pad probe info must have event"),
        };
        gst::PadProbeReturn::Ok
    });

    video_info_data
}



