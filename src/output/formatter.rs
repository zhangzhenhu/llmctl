use crate::provider::ChatResponse;
use colored::*;

pub fn format_chat_response(response: &ChatResponse) {
    if let Some(ref reasoning) = response.reasoning_content {
        println!("{}:", "Thinking".cyan());
        println!("{}", "─".repeat(50).dimmed());
        println!("{}", reasoning);
        println!("{}", "─".repeat(50).dimmed());
    }

    if let Some(ref content) = response.content {
        println!("{}:", "Response".cyan());
        println!("{}", "─".repeat(50).dimmed());
        println!("{}", content);
    }
    println!("{}", "─".repeat(50).dimmed());

    if let (Some(input), Some(output)) = (response.input_tokens, response.output_tokens) {
        println!("{}: Input {}, Output {}", "Token".dimmed(), input, output);
    }
    println!(
        "{}: ({}){}",
        "Model".green(),
        response.provider.green(),
        response.model.green()
    );
    println!("{}: {} ms", "Duration".yellow(), response.duration_ms);
}

pub fn format_model_list(models: &[crate::provider::ModelInfo]) {
    println!("{}", "─".repeat(50).dimmed());
    println!("{}", "Available Models:".green());
    println!("{}", "─".repeat(50).dimmed());

    for model in models {
        println!("  - {}", model.name);
    }

    println!("{}", "─".repeat(50).dimmed());
}

pub fn print_error(error: &str) {
    eprintln!("{}: {}", "Error".red(), error.red());
}

pub fn print_success(message: &str) {
    println!("{}", message.green());
}

pub fn print_info(message: &str) {
    println!("{}", message);
}
