use std::error;
use vmaf::*;

fn main() -> Result<(), Box<dyn error::Error>>{
    let score = collect_bootstrapped_score("videos/original.y4m", "videos/low_quality.y4m", CollectScoreOpts::default())?;

    println!("Score: {:?}", score);
    Ok(())
}

