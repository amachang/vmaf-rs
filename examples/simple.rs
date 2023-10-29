fn main() {
    let score = vmaf::collect_score("tests/original.y4m", "tests/high_quality.y4m", vmaf::CollectScoreOpts::default()).unwrap();

    println!("Score: {}", score);
}
