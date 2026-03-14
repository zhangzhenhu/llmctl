#![allow(dead_code)]
use colored::Colorize;
use std::io::Write;

pub fn stream_output(content: &str, model: &str) {
    print!("{}: {}", "Model".green(), model.green());
    println!(" {}", "(streaming...)".dimmed());
    print!("");
    print!("{}", "─".repeat(50).dimmed());
    println!("");
    print!("");
    print!("{}", content);
    std::io::stdout().flush().ok();
}

pub fn stream_end(duration_ms: u64) {
    println!("");
    println!("{}", "─".repeat(50).dimmed());
    println!("{}: {} ms", "Duration".yellow(), duration_ms);
}
