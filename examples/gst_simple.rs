use std::error;
use vmaf::*;

fn main() -> Result<(), Box<dyn error::Error>>{
    let score = collect_score("videos/low_quality.mp4", "videos/high_quality.mp4", CollectScoreOpts::default())?;

    println!("Score: {}", score);
    Ok(())
}

