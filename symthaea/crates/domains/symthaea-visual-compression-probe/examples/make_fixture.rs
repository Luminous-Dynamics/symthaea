use symthaea_visual_compression_probe::GrayImage;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 64;
    let height = 64;
    let mut pixels = vec![0.0; width * height];
    for y in 0..height {
        for x in 0..width {
            let gradient = (x as f32 / (width - 1) as f32) * 0.45;
            let pipe = if (28..=35).contains(&y) { 0.35 } else { 0.0 };
            let crack = if x == y || x + y == 63 { 0.25 } else { 0.0 };
            pixels[y * width + x] = (gradient + pipe + crack).clamp(0.0, 1.0);
        }
    }
    let image = GrayImage::new(width, height, pixels)?;
    image.write_pgm("fixture_pump_scan.pgm")?;
    println!("wrote fixture_pump_scan.pgm");
    Ok(())
}
