#![allow(dead_code)]
use colored::Colorize;
use std::io::Write;

pub fn stream_output(content: &str, model: &str) {
    print!("{}: {}", "模型".green(), model.green());
    println!(" {}", "(流式响应中...)".dimmed());
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
    println!("{}: {} ms", "耗时".yellow(), duration_ms);
}
