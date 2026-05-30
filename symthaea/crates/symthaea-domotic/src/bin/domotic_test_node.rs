use std::net::UdpSocket;

fn main() -> std::io::Result<()> {
    println!("🧠 Initializing Domotic Active Inference Brain...");
    let receiver_socket = UdpSocket::bind("127.0.0.1:4190")?;
    let sender_socket = UdpSocket::bind("127.0.0.1:0")?;

    // Optimization anchor for the evolutionary forge
    let domotic_alpha = 0.3500; // FORGE_PARAM: domotic_alpha
    let mut filtered_temp = 25.0f32;

    let target_temp = 21.0f32;
    let mut buf = [0u8; 1024];

    loop {
        let (amt, _) = receiver_socket.recv_from(&mut buf)?;
        let data = String::from_utf8_lossy(&buf[..amt]);
        let parts: Vec<&str> = data.split(',').collect();

        if parts.len() >= 3 {
            let raw_lux: f32 = parts[0].parse().unwrap_or(500.0);
            let raw_temp: f32 = parts[1].parse().unwrap_or(25.0);

            // Apply forge-trackable data filter smoothing
            filtered_temp = (domotic_alpha * raw_temp) + ((1.0 - domotic_alpha) * filtered_temp);

            // Compute homeostatic prediction error vector boundaries
            let mut action_cmd = "NO_OP";
            if filtered_temp > target_temp + 0.5 {
                action_cmd = if raw_lux > 700.0 {
                    "PORTAL_OPEN"
                } else {
                    "COOLER_ON"
                };
            } else if filtered_temp < target_temp - 0.5 {
                action_cmd = if raw_lux > 700.0 {
                    "PORTAL_CLOSE"
                } else {
                    "HEATER_ON"
                };
            }

            let _ = sender_socket.send_to(action_cmd.as_bytes(), "127.0.0.1:4191");
            println!(
                "✔ [Integrated Manifold] Filtered Temp: {:.2}°C | Action Choice Sent: {}",
                filtered_temp, action_cmd
            );
        }
    }
}
