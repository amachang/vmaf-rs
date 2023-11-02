use std::error;
use vmaf::*;

fn main() -> Result<(), Box<dyn error::Error>>{
    let score = collect_score("videos/original.y4m", "videos/high_quality.y4m", CollectScoreOpts::default())?;

    println!("Score: {}", score);
    Ok(())
}
