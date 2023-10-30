use std::error;
use vmaf::*;

fn main() -> Result<(), Box<dyn error::Error>>{
    let score = collect_score("videos/foo.y4m", "videos/bar.y4m", CollectScoreOpts::default())?;

    println!("Score: {}", score);
    Ok(())
}
