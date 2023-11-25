use std::{error, path::PathBuf, fmt::Write, time::{Duration, Instant}};
use clap::Parser;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use vmaf::*;
use gstreamer::ClockTime;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[arg()]
    path: PathBuf,

    #[arg(short, long)]
    start_sec: Option<f64>,

    #[arg(short, long)]
    duration_sec: Option<f64>,
}

fn main() -> Result<(), Box<dyn error::Error>> {
    let args = Args::parse();
    let start = if let Some(s) = args.start_sec {
        Some(ClockTime::from_nseconds((s * 1000_000_000.0).trunc() as u64))
    } else {
        None
    };
    let duration = if let Some(d) = args.duration_sec {
        Some(ClockTime::from_nseconds((d * 1000_000_000.0).trunc() as u64))
    } else {
        None
    };
    let mut stream = gst::PictureStream::from_path(args.path, gst::PictureStreamOpts { start, duration, ..Default::default() })?;

    let pb = ProgressBar::new(0);
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% (ETA: {eta})")?
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#>-"));

    let started_at = Instant::now();

    let mut pic_count = 0;
    let progress_update_interval = Duration::from_millis(100);
    let mut progress_updated_at = Instant::now();

    let start = start.unwrap_or(ClockTime::ZERO);
    while let Some(_pic) = stream.next_pic()? {
        if progress_update_interval < progress_updated_at.elapsed() {
            if let (Some(pos), Some(dur)) = (stream.position(), stream.duration()) {
                if let Some(duration) = duration {
                    pb.set_length(duration.nseconds() + start.nseconds());
                } else {
                    pb.set_length(dur.nseconds());
                }
                pb.set_position(pos.nseconds() - start.nseconds());
            }
            progress_updated_at = Instant::now();
        }
        pic_count += 1;
    }
    pb.finish();

    println!("Elapsed Time: {:?}, Picture Count: {}", started_at.elapsed(), pic_count);

    Ok(())
}

