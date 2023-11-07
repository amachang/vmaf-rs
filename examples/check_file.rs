use std::{error, path::PathBuf, fmt::Write, time::{Duration, Instant}};
use clap::Parser;
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use vmaf::*;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[arg()]
    path: PathBuf,
}

fn main() -> Result<(), Box<dyn error::Error>>{
    let args = Args::parse();
    let mut stream = gst::PictureStream::from_path(args.path, false)?;

    let pb = ProgressBar::new(0);
    pb.set_style(ProgressStyle::with_template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {percent}% (ETA: {eta})")?
        .with_key("eta", |state: &ProgressState, w: &mut dyn Write| write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap())
        .progress_chars("#>-"));

    let started_at = Instant::now();

    let mut pic_count = 0;
    let progress_update_interval = Duration::from_millis(100);
    let mut progress_updated_at = Instant::now();

    while let Some(_pic) = stream.next_pic()? {
        if progress_update_interval < progress_updated_at.elapsed() {
            if let (Some(pos), Some(dur)) = (stream.position_nanos(), stream.duration_nanos()) {
                pb.set_length(dur);
                pb.set_position(pos);
            }
            progress_updated_at = Instant::now();
        }
        pic_count += 1;
    }
    pb.finish();

    println!("Elapsed Time: {:?}, Picture Count: {}", started_at.elapsed(), pic_count);

    Ok(())
}

