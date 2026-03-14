use crate::provider::ChatResponse;
use colored::*;

pub fn format_chat_response(response: &ChatResponse) {
    // println!("{}", "─".repeat(50).dimmed());

    if let Some(ref reasoning) = response.reasoning_content {
        println!("{}:", "思考过程".cyan());
        println!("{}", "─".repeat(50).dimmed());
        println!("{}", reasoning);
        println!("{}", "─".repeat(50).dimmed());
    }

    if let Some(ref content) = response.content {
        println!("{}:", "回答".cyan());
        println!("{}", "─".repeat(50).dimmed());
        println!("{}", content);
    }
    println!("{}", "─".repeat(50).dimmed());

    if let (Some(input), Some(output)) = (response.input_tokens, response.output_tokens) {
        println!("{}: 输入 {}, 输出 {}", "Token".dimmed(), input, output);
    }
    println!(
        "{}: ({}){}",
        "模型".green(),
        response.provider.green(),
        response.model.green()
    );
    println!("{}: {} ms", "耗时".yellow(), response.duration_ms);
}

pub fn format_model_list(models: &[crate::provider::ModelInfo]) {
    println!("{}", "─".repeat(50).dimmed());
    println!("{}", "可用模型列表:".green());
    println!("{}", "─".repeat(50).dimmed());

    for model in models {
        println!("  - {} ({})", model.name, model.provider);
    }

    println!("{}", "─".repeat(50).dimmed());
}

pub fn print_error(error: &str) {
    eprintln!("{}: {}", "错误".red(), error.red());
}

pub fn print_success(message: &str) {
    println!("{}", message.green());
}

pub fn print_info(message: &str) {
    println!("{}", message);
}
