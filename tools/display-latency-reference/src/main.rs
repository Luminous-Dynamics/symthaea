// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use display_latency_reference::{
    FrameprobeHeader, LatencyConfig, LatencyResult, TimingReference, analyze_capture,
};
use std::env;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const OUTPUT_HEADER: &str = "row_index,status,reference,baseline_adc,settled_adc,swing_adc,noise_peak_to_peak_adc,separation,estimated_sample_interval_us,t10_us,t50_us,t90_us,response_t10_t90_us,reason";

struct Options {
    input: Option<PathBuf>,
    reference: TimingReference,
}

fn main() {
    let options = match parse_args() {
        Ok(Some(options)) => options,
        Ok(None) => return,
        Err(error) => fail(&error, 2),
    };

    let input: Box<dyn BufRead> = match options.input {
        Some(path) => match open_input(&path) {
            Ok(reader) => Box::new(reader),
            Err(error) => fail(&error, 2),
        },
        None => Box::new(BufReader::new(io::stdin())),
    };

    let stdout = io::stdout();
    let mut output = stdout.lock();
    if let Err(error) = process(input, &mut output, options.reference) {
        fail(&error, 2);
    }
}

fn parse_args() -> Result<Option<Options>, String> {
    let mut input = None;
    let mut reference = TimingReference::Press;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" => {
                let path = args
                    .next()
                    .ok_or_else(|| "--input requires a file path".to_owned())?;
                input = Some(PathBuf::from(path));
            }
            "--reference" => {
                reference = match args.next().as_deref() {
                    Some("press") => TimingReference::Press,
                    Some("delivery") => TimingReference::HostDelivery,
                    _ => {
                        return Err(
                            "--reference must be either press or delivery".to_owned()
                        );
                    }
                };
            }
            "--help" | "-h" => {
                print_help();
                return Ok(None);
            }
            other => return Err(format!("unknown argument: {other}")),
        }
    }
    Ok(Some(Options { input, reference }))
}

fn open_input(path: &Path) -> Result<BufReader<File>, String> {
    File::open(path)
        .map(BufReader::new)
        .map_err(|error| format!("cannot open {}: {error}", path.display()))
}

fn process<R: BufRead, W: Write>(
    reader: R,
    output: &mut W,
    reference: TimingReference,
) -> Result<(), String> {
    let config = LatencyConfig::default();
    config.validate().map_err(|error| error.to_string())?;
    writeln!(output, "{OUTPUT_HEADER}").map_err(|error| error.to_string())?;
    let mut header: Option<FrameprobeHeader> = None;
    let mut row_index = 0_usize;
    for line in reader.lines() {
        let line = line.map_err(|error| format!("cannot read input: {error}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if header.is_none() {
            header = Some(FrameprobeHeader::parse(trimmed)?);
            continue;
        }

        row_index += 1;
        let result = match header.as_ref().expect("header was set").parse_row(trimmed) {
            Ok(row) => analyze_capture(&row, reference, config)
                .map_err(|error| error.to_string())?,
            Err(reason) => LatencyResult {
                status: display_latency_reference::LatencyStatus::MalformedRow,
                reference,
                reason: Some(reason),
                baseline_adc: None,
                settled_adc: None,
                swing_adc: None,
                noise_peak_to_peak_adc: None,
                signal_noise_separation: None,
                estimated_sample_interval_us: None,
                t10_us: None,
                t50_us: None,
                t90_us: None,
                response_t10_t90_us: None,
            },
        };
        write_result(output, row_index, &result)
            .map_err(|error| format!("cannot write row {row_index}: {error}"))?;
    }
    if header.is_none() {
        return Err("input contains no supported CSV header".to_owned());
    }
    Ok(())
}

fn write_result<W: Write>(
    output: &mut W,
    row_index: usize,
    result: &LatencyResult,
) -> io::Result<()> {
    let reason = result.reason.as_deref().unwrap_or("");
    writeln!(
        output,
        "{row_index},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        result.status.as_str(),
        result.reference.as_str(),
        number(result.baseline_adc),
        number(result.settled_adc),
        number(result.swing_adc),
        number(result.noise_peak_to_peak_adc),
        number(result.signal_noise_separation),
        number(result.estimated_sample_interval_us),
        number(result.t10_us),
        number(result.t50_us),
        number(result.t90_us),
        number(result.response_t10_t90_us),
        csv_cell(reason),
    )
}

fn number(value: Option<f64>) -> String {
    value.map_or_else(String::new, |value| format!("{value:.3}"))
}

fn csv_cell(value: &str) -> String {
    if value.contains(',')
        || value.contains('"')
        || value.contains('\n')
        || value.contains('\r')
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn print_help() {
    println!(
        "Analyze frameprobe-compatible photodiode captures without physical qualification.\n\
         Usage: display-latency-reference [--input FILE] [--reference press|delivery]\n\
         Default input is standard input; default reference is press.\n\
         Output is one CSV row per capture, including explicit skip/status fields.\n\
         v0 does not compensate for temporal light modulation and reports only a\n\
         normalized-crossing proxy, not absolute display quality or panel-only response."
    );
}

fn fail(message: &str, exit_code: i32) -> ! {
    eprintln!("{message}");
    std::process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::{process, OUTPUT_HEADER};
    use display_latency_reference::TimingReference;
    use std::io::Cursor;

    #[test]
    fn cli_emits_one_deterministic_result_row_per_capture() {
        let mut samples = vec![1_000; 1_200];
        samples.extend((0..100).map(|value| 1_000 + value as u16));
        samples.extend(vec![1_100; 500]);
        let row = format!(
            "80,43400,1800,1200,{};,200\n",
            samples
                .iter()
                .map(|sample| sample.to_string())
                .collect::<Vec<_>>()
                .join(";")
        );
        let input = format!(
            "# META,mode=click\nclickTime,timeTaken,sampleCount,preClickSamples,samples,deliveryTime\n{row}"
        );
        let mut first = Vec::new();
        process(Cursor::new(input.as_bytes()), &mut first, TimingReference::Press)
            .expect("fixture input is valid");
        let mut second = Vec::new();
        process(Cursor::new(input.as_bytes()), &mut second, TimingReference::Press)
            .expect("same input is valid");
        assert_eq!(first, second);
        let output = String::from_utf8(first).expect("output is UTF-8");
        assert_eq!(output.lines().count(), 2);
        assert!(output.starts_with(OUTPUT_HEADER));
        assert!(output.contains("MEASURED_UNQUALIFIED"));
        assert!(output.contains(",1400.000,"));
    }

    #[test]
    fn cli_retains_malformed_rows_as_explicit_statuses() {
        let input = concat!(
            "clickTime,timeTaken,sampleCount,preClickSamples,samples\n",
            "80,44200,3,2,1000;bad;1100;\n"
        );
        let mut output = Vec::new();
        process(Cursor::new(input), &mut output, TimingReference::Press)
            .expect("one malformed row is recorded, not fatal");
        let text = String::from_utf8(output).expect("output is UTF-8");
        assert!(text.contains("MALFORMED_ROW"));
    }
}
