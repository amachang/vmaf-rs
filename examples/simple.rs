fn main() {
    let score = vmaf::collect_score("videos/foo.y4m", "videos/bar.y4m", vmaf::CollectScoreOpts::default()).unwrap();

    println!("Score: {}", score);
}
