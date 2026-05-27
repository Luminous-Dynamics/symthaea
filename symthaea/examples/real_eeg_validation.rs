// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Validate Cincinnati-LTC and Hybrid Ensemble on Real EEG Data
//!
//! Uses CHB-MIT Scalp EEG Database (PhysioNet)
//! This is real clinical EEG from pediatric patients
//!
//! Expected: 75%+ accuracy (better than random) on real biosignals

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use symthaea::hdc::reservoir::HybridEnsemblePredictor;

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     REAL EEG VALIDATION - CHB-MIT PhysioNet Database         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  Testing Cincinnati-LTC and Hybrid Ensemble on clinical EEG  ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Load EDF file
    let edf_path = "data/eeg/chb01_01.edf";
    println!("Loading: {}", edf_path);

    match read_edf_channel(edf_path, 0) {
        Ok((data, header)) => {
            println!("\n=== EEG RECORDING INFO ===\n");
            println!("  Patient:           {}", header.patient_id.trim());
            println!("  Recording:         {}", header.recording_id.trim());
            println!("  Start date:        {}", header.start_date);
            println!("  Duration:          {} seconds", header.duration_sec);
            println!("  Channels:          {}", header.num_channels);
            println!("  Samples/channel:   {}", data.len());
            println!(
                "  Sample rate:       ~{:.1} Hz",
                data.len() as f64 / header.duration_sec
            );

            // Normalize data to [0, 1]
            let min_val = data.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let range = max_val - min_val;
            let normalized: Vec<f64> = data.iter().map(|v| (v - min_val) / range).collect();

            println!("\n  Signal range:      {:.2} to {:.2}", min_val, max_val);
            println!("  Normalized:        0.0 to 1.0\n");

            // Test Hybrid Ensemble
            println!("=== HYBRID ENSEMBLE PREDICTION ===\n");
            test_on_eeg(&normalized, "Hybrid Ensemble");

            // Summary
            println!("\n╔══════════════════════════════════════════════════════════════╗");
            println!("║                     VALIDATION COMPLETE                      ║");
            println!("╚══════════════════════════════════════════════════════════════╝");
        }
        Err(e) => {
            println!("Error loading EDF: {}", e);
            println!("\nPlease ensure data/eeg/chb01_01.edf exists.");
            println!("Download from: https://physionet.org/content/chbmit/1.0.0/");
        }
    }
}

fn test_on_eeg(data: &[f64], name: &str) {
    // Use first 80% for training, last 20% for testing
    let train_size = (data.len() * 8) / 10;
    let test_size = data.len() - train_size - 1;

    println!("  Method:            {}", name);
    println!("  Training samples:  {}", train_size);
    println!("  Test samples:      {}", test_size);

    // Subsample for faster processing (every 10th sample)
    let subsample = 10;
    let subsampled: Vec<f64> = data.iter().step_by(subsample).cloned().collect();
    let sub_train = (subsampled.len() * 8) / 10;
    let sub_test = subsampled.len() - sub_train - 1;

    println!(
        "  Subsampled by:     {}x (for faster processing)",
        subsample
    );
    println!("  Effective train:   {}", sub_train);
    println!("  Effective test:    {}", sub_test);

    // Create predictor
    let mut ensemble = HybridEnsemblePredictor::new(42);

    // Training phase
    for sample in subsampled.iter().take(sub_train) {
        ensemble.observe(*sample);
    }

    // Test phase
    let mut correct = 0;
    let mut total = 0;
    let threshold = 0.5;

    for i in sub_train..(subsampled.len() - 1) {
        let pred_binary = ensemble.predict() > threshold;
        let actual_binary = subsampled[i + 1] > threshold;

        if pred_binary == actual_binary {
            correct += 1;
        }
        total += 1;
        ensemble.observe(subsampled[i]);
    }

    let accuracy = 100.0 * correct as f64 / total as f64;
    println!("\n  Correct:           {}", correct);
    println!("  Accuracy:          {:.1}%", accuracy);
    println!("  (Random baseline:  50.0%)");
    println!("  Signal type:       {:?}", ensemble.get_signal_type());
    println!("  Diagnostics:       {}", ensemble.diagnostics());

    if accuracy > 60.0 {
        println!("  Result:            ✅ SIGNIFICANTLY BETTER THAN RANDOM");
    } else if accuracy > 55.0 {
        println!("  Result:            ✅ BETTER THAN RANDOM");
    } else if accuracy > 52.0 {
        println!("  Result:            🔄 MARGINALLY BETTER");
    } else {
        println!("  Result:            ⚠️ Near random (EEG is complex!)");
    }
}

// Simple EDF header parser
#[derive(Debug)]
#[allow(dead_code)]
struct EdfHeader {
    patient_id: String,
    recording_id: String,
    start_date: String,
    num_channels: usize,
    duration_sec: f64,
    samples_per_record: Vec<usize>,
}

fn read_edf_channel(path: &str, channel: usize) -> Result<(Vec<f64>, EdfHeader), String> {
    let file = File::open(path).map_err(|e| format!("Cannot open file: {}", e))?;
    let mut reader = BufReader::new(file);

    // Read header (256 bytes fixed + 256 bytes per channel)
    let mut header_buf = [0u8; 256];
    reader
        .read_exact(&mut header_buf)
        .map_err(|e| format!("Cannot read header: {}", e))?;

    // Parse header fields
    let patient_id = String::from_utf8_lossy(&header_buf[8..88]).to_string();
    let recording_id = String::from_utf8_lossy(&header_buf[88..168]).to_string();
    let start_date = String::from_utf8_lossy(&header_buf[168..184]).to_string();

    // Number of bytes in header
    let header_bytes: usize = String::from_utf8_lossy(&header_buf[184..192])
        .trim()
        .parse()
        .unwrap_or(256);

    // Number of data records
    let num_records: i64 = String::from_utf8_lossy(&header_buf[236..244])
        .trim()
        .parse()
        .unwrap_or(1);

    // Duration of each record in seconds
    let record_duration: f64 = String::from_utf8_lossy(&header_buf[244..252])
        .trim()
        .parse()
        .unwrap_or(1.0);

    // Number of channels
    let num_channels: usize = String::from_utf8_lossy(&header_buf[252..256])
        .trim()
        .parse()
        .unwrap_or(1);

    let total_duration = num_records as f64 * record_duration;

    // Read channel-specific headers
    let channel_header_size = 256 * num_channels;
    let mut channel_headers = vec![0u8; channel_header_size];
    reader
        .read_exact(&mut channel_headers)
        .map_err(|e| format!("Cannot read channel headers: {}", e))?;

    // Parse samples per record for each channel (at offset 216-224 per channel block)
    let mut samples_per_record = Vec::new();
    for ch in 0..num_channels {
        let offset = ch * 256 + 216; // Position of "nr of samples" field
        if offset + 8 <= channel_headers.len() {
            // Field is at bytes 216-224 of each 256-byte channel block
            // But we need to look at the ns field which is after the other fields
            let samples_offset = 216 * num_channels + ch * 8;
            if samples_offset + 8 <= channel_headers.len() {
                let samples_str =
                    String::from_utf8_lossy(&channel_headers[samples_offset..samples_offset + 8]);
                let samples: usize = samples_str.trim().parse().unwrap_or(256);
                samples_per_record.push(samples);
            } else {
                samples_per_record.push(256); // default
            }
        } else {
            samples_per_record.push(256);
        }
    }

    // If parsing failed, use a reasonable default based on typical 256 Hz sampling
    if samples_per_record.is_empty() || samples_per_record[0] == 0 {
        let default_samples = (256.0 * record_duration) as usize;
        samples_per_record = vec![default_samples; num_channels];
    }

    let header = EdfHeader {
        patient_id,
        recording_id,
        start_date,
        num_channels,
        duration_sec: total_duration,
        samples_per_record: samples_per_record.clone(),
    };

    // Seek to data start
    reader
        .seek(SeekFrom::Start(header_bytes as u64))
        .map_err(|e| format!("Cannot seek: {}", e))?;

    // Read data for the specified channel
    let samples_per_ch = samples_per_record.get(channel).copied().unwrap_or(256);
    let _total_samples = samples_per_ch * num_records as usize;

    // Each sample is 2 bytes (16-bit signed integer)
    let mut all_data = Vec::new();
    // Calculate bytes per record for all channels
    let bytes_per_record: usize = samples_per_record.iter().map(|s| s * 2).sum();
    let channel_offset: usize = samples_per_record[..channel].iter().map(|s| s * 2).sum();

    for _record in 0..num_records {
        // Seek to channel data within this record
        let mut record_data = vec![0u8; bytes_per_record];
        if reader.read_exact(&mut record_data).is_err() {
            break; // End of file
        }

        // Extract samples for our channel
        for sample in 0..samples_per_ch {
            let offset = channel_offset + sample * 2;
            if offset + 2 <= record_data.len() {
                let value = i16::from_le_bytes([record_data[offset], record_data[offset + 1]]);
                all_data.push(value as f64);
            }
        }
    }

    Ok((all_data, header))
}