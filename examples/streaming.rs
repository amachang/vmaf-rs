use std::{error, path::PathBuf};
use clap::Parser;
use vmaf::*;

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    ref_path: PathBuf,

    #[arg(short, long)]
    dist_path: PathBuf,
}

fn main() -> Result<(), Box<dyn error::Error>>{
    let args = Args::parse();

    let mut ref_stream = AutoPictureStream::from_path(args.ref_path)?;
    let mut dist_stream = AutoPictureStream::from_path(args.dist_path)?;
    let mut score_collector = ScoreCollector::<BootstrappedScore>::new(Model::default(), ScoreCollectorOpts::default())?;

    let collection_interval = 100;
    while let (Some(ref_pic), Some(dist_pic)) = (ref_stream.next_pic()?, dist_stream.next_pic()?) {
        score_collector.read_pictures(ref_pic, dist_pic)?;
        let n_scores = score_collector.n_scores();
        if n_scores % collection_interval == 0 {
            let score = score_collector.collect_score(ScoreCollectorCollectScoreOpts::default())?;
            println!("[{} Frames] Score: {:?}", n_scores, score);
        }
    }

    Ok(())
}


