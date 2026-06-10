use std::net::UdpSocket;
use std::thread;
use std::time::{Duration, Instant};

fn main() -> std::io::Result<()> {
    println!("🚀 Closed-Loop Domotic Telemetry Simulator Engine Active.");
    let sender_socket = UdpSocket::bind("127.0.0.1:0")?;
    let receiver_socket = UdpSocket::bind("127.0.0.1:4191")?;
    receiver_socket.set_nonblocking(true)?;

    println!("📡 Streaming to 127.0.0.1:4190 | Listening for actuation on 127.0.0.1:4191");

    let mut lux = 500.0f32;
    let mut temp = 25.0f32;
    let mut portal_open = false;
    let mut cycle = 0.0f32;

    loop {
        // Process incoming active inference commands from Symthaea
        let mut buf = [0u8; 64];
        if let Ok((amt, _)) = receiver_socket.recv_from(&mut buf) {
            let msg = String::from_utf8_lossy(&buf[..amt]);
            if msg.contains("PORTAL_OPEN") {
                portal_open = true;
            }
            if msg.contains("PORTAL_CLOSE") {
                portal_open = false;
            }
            if msg.contains("HEATER_ON") {
                temp += 0.4;
            }
            if msg.contains("COOLER_ON") {
                temp -= 0.4;
            }
        }

        // Simulating environmental drift mechanics
        cycle += 0.05;
        lux = 500.0 + (cycle.sin() * 500.0);

        if portal_open {
            temp += (15.0 - temp) * 0.05; // Portal vents toward external 15°C air
        } else {
            temp += (22.0 - temp) * 0.01; // Internal ambient insulation crawl
        }

        let packet = format!(
            "{:.1},{:.1},{}",
            lux,
            temp,
            if portal_open { 1.0 } else { 0.0 }
        );
        let _ = sender_socket.send_to(packet.as_bytes(), "127.0.0.1:4190");

        println!(
            "⚡ Env State -> Lux: {:.1}, Temp: {:.1}°C, Portal Open: {}",
            lux, temp, portal_open
        );
        thread::sleep(Duration::from_millis(200));
    }
}
