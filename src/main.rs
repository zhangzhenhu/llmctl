#![allow(dead_code)]
mod backends;
mod builder;
mod config;
mod error;
mod output;
mod provider;
mod utils;
use clap::Parser;
use config::{
    convert_config, load_config, merge_configs, search_config_file, validate_config_with_list, Args,
};
use error::LlmProbeError;
use output::{format_chat_response, format_model_list, print_error, print_info, print_success};
use provider::{create_llm_backend, LLMClient};
// use std::io::Write;
use std::path::PathBuf;
// use std::time::Instant;
use utils::init_config_file;

use clap::CommandFactory;

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    if std::env::args().len() == 1 {
        let _ = Args::command().print_help();
        println!();
        std::process::exit(0);
    }

    let args = Args::parse();

    if args.version {
        println!("llmctl {}", Args::command().get_version().unwrap_or("0.1.6"));
        std::process::exit(0);
    }

    if let Err(e) = run(args) {
        print_error(&e.user_message());
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), LlmProbeError> {
    if args.init.is_some() || args.init_path.is_some() {
        return handle_init(&args);
    }

    if let Some(convert_paths) = &args.convert {
        return handle_convert(convert_paths);
    }

    let file_config = if let Some(config_path) = &args.config {
        Some(load_config(config_path)?)
    } else if let Some(auto_path) = search_config_file() {
        Some(load_config(&auto_path)?)
    } else {
        None
    };

    let runtime_config = merge_configs(file_config, &args);

    validate_config_with_list(&runtime_config, args.list)?;

    let backend = create_llm_backend(
        &runtime_config.provider,
        &runtime_config.api_key,
        Some(&runtime_config.base_url),
        &runtime_config.model,
        Some(&runtime_config),
    )?;

    if args.list {
        return handle_list(&backend);
    }

    handle_chat(&args, &runtime_config, backend)
}

fn handle_init(args: &Args) -> Result<(), LlmProbeError> {
    let format_input = args.init.as_deref().unwrap_or("yaml");

    let (output_path, format) = if let Some(path) = &args.init_path {
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("yaml");
        (path.clone(), ext.to_string())
    } else if format_input.contains('.') || (format_input != "yaml" && format_input != "json") {
        let path = PathBuf::from(format_input);
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("yaml")
            .to_string();
        (path, ext)
    } else {
        (PathBuf::from(format!("./llmctl.{}", format_input)), format_input.to_string())
    };

    init_config_file(&output_path, &format).map_err(|e| LlmProbeError::ApiError(e))?;

    print_success(&format!("Config file created: {}", output_path.display()));
    Ok(())
}

fn handle_convert(convert_paths: &Vec<PathBuf>) -> Result<(), LlmProbeError> {
    let input_path = &convert_paths[0];
    let output_path = convert_paths.get(1).map(|p| p.as_path());

    convert_config(input_path, output_path)
}

fn handle_list(backend: &LLMClient) -> Result<(), LlmProbeError> {
    print_info("Fetching model list...");

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|_| LlmProbeError::ApiError("Failed to create runtime".to_string()))?;
    let models = runtime.block_on(backend.list_models())?;

    format_model_list(&models);
    Ok(())
}

fn handle_chat(
    _args: &Args,
    config: &config::RuntimeConfig,
    backend: LLMClient,
) -> Result<(), LlmProbeError> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|_| LlmProbeError::ApiError("Failed to create runtime".to_string()))?;

    let messages = config.context.clone();

    if config.stream {
        runtime.block_on(backend.stream_chat(messages, &config.model))?;
        Ok(())
    } else {
        let response = runtime.block_on(backend.chat_completion(messages, &config.model))?;

        format_chat_response(&response);
        Ok(())
    }
}
