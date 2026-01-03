#![allow(missing_docs, clippy::use_debug)]

//! This example demonstrates FSST compression on a business email template.
//!
//! It shows how FSST can efficiently compress repeated patterns in structured
//! business documents like email templates, especially those with repetitive
//! formatting and terminology.
//!
//! Example:
//!
//! ```
//! cargo run --example email_template_compression
//! ```

use std::{fs::File, io::Read, path::Path};

use fsst::Compressor;

fn main() {
    let template_path = Path::new("templates/cost_application_email.txt");

    // Read the email template
    let mut template = String::new();
    {
        let mut f = File::open(template_path).expect(
            "Failed to open template file. Make sure templates/cost_application_email.txt exists",
        );
        f.read_to_string(&mut template)
            .expect("Failed to read template file");
    }

    let original_size = template.len();
    println!("Original email template size: {} bytes", original_size);
    println!("\n{}", "=".repeat(60));

    // Split template into lines for training
    let lines: Vec<&[u8]> = template.lines().map(|line| line.as_bytes()).collect();

    // Train the compressor on the email template
    println!("Training FSST compressor on email template...");
    let start = std::time::Instant::now();
    let compressor = Compressor::train(&lines);
    let train_duration = std::time::Instant::now().duration_since(start);
    println!("Training completed in {}µs", train_duration.as_micros());

    // Compress each line
    let mut total_compressed_size = 0;
    let mut buffer = Vec::with_capacity(1024);

    let start = std::time::Instant::now();
    for line in &lines {
        buffer.clear();
        unsafe { compressor.compress_into(line, &mut buffer) };
        total_compressed_size += buffer.len();
    }
    let compress_duration = std::time::Instant::now().duration_since(start);

    println!("\n{}", "=".repeat(60));
    println!("Compression Results:");
    println!("{}", "-".repeat(60));
    println!("Original size:    {} bytes", original_size);
    println!("Compressed size:  {} bytes", total_compressed_size);
    println!(
        "Compression ratio: {:.2}%",
        100.0 * (total_compressed_size as f64) / (original_size as f64)
    );
    println!(
        "Space saved:      {} bytes ({:.2}%)",
        original_size - total_compressed_size,
        100.0 * (1.0 - (total_compressed_size as f64) / (original_size as f64))
    );
    println!("Compression time: {}µs", compress_duration.as_micros());

    // Verify decompression works correctly
    println!("\n{}", "=".repeat(60));
    println!("Verifying decompression...");

    let decompressor = compressor.decompressor();
    let mut decompressed_template = String::new();

    for (i, line) in lines.iter().enumerate() {
        buffer.clear();
        unsafe { compressor.compress_into(line, &mut buffer) };
        let decompressed = decompressor.decompress(&buffer);
        let decompressed_str =
            std::str::from_utf8(&decompressed).expect("Failed to decode decompressed data");
        decompressed_template.push_str(decompressed_str);
        // Add newline after each line
        if i < lines.len() - 1 {
            decompressed_template.push('\n');
        }
    }

    // Add final newline if original had one
    if template.ends_with('\n') {
        decompressed_template.push('\n');
    }

    if template == decompressed_template {
        println!("✓ Decompression successful! Template matches original.");
    } else {
        println!("✗ Decompression failed! Template does not match.");
        println!("Original length: {}", template.len());
        println!("Decompressed length: {}", decompressed_template.len());
    }

    println!("{}", "=".repeat(60));
}
