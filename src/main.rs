mod config;
mod error;
mod http;
mod output;
mod provider;
mod runtime;
mod utils;
use clap::Parser;
use config::{
    convert_config, list_builtin_adapters, load_app_config, resolve_runtime_config,
    search_config_file, validate_resolved_config, AppConfigV2, Args,
};
use error::LlmProbeError;
use output::{
    format_chat_response, format_model_list, print_adapter_list, print_doctor_report,
    print_dry_run, print_error, print_info, print_success,
};
use runtime::{merge_plan_diagnostics, plan_runtime, GenaiRuntime, RuntimeBackend};
use std::path::PathBuf;
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
        println!(
            "llmctl {}",
            Args::command()
                .get_version()
                .unwrap_or(env!("CARGO_PKG_VERSION"))
        );
        std::process::exit(0);
    }

    if let Err(e) = run(args) {
        print_error(&e.user_message());
        std::process::exit(1);
    }
}

fn run(args: Args) -> Result<(), LlmProbeError> {
    if args.list_adapters {
        print_adapter_list(&list_builtin_adapters());
        return Ok(());
    }

    if args.init.is_some() || args.init_path.is_some() {
        return handle_init(&args);
    }

    if let Some(convert_paths) = &args.convert {
        return handle_convert(convert_paths);
    }

    let app_config = if let Some(config_path) = &args.config {
        load_app_config(config_path)?
    } else if let Some(auto_path) = search_config_file() {
        load_app_config(&auto_path)?
    } else {
        AppConfigV2::default()
    };
    let resolved = resolve_runtime_config(app_config, &args)?;
    let plan = plan_runtime(&resolved);
    let mut report = validate_resolved_config(&resolved, &args);
    merge_plan_diagnostics(&mut report, &plan);

    if args.dry_run {
        print_dry_run(
            &resolved,
            &report,
            plan.backend_label(),
            plan.reason_label(),
            plan.extra_body_supported,
        );
    }
    if args.doctor_config {
        print_doctor_report(&report);
    }
    if report.has_errors() {
        let messages = report
            .errors()
            .into_iter()
            .map(|d| format!("[{}] {}", d.code, d.message))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(LlmProbeError::ConfigError(messages));
    }
    if args.dry_run || args.doctor_config {
        return Ok(());
    }

    if plan.backend != RuntimeBackend::Genai {
        return Err(LlmProbeError::ConfigError(
            "No executable runtime backend is available".to_string(),
        ));
    }

    let genai_runtime = GenaiRuntime::from_resolved(resolved)?;
    if args.list {
        return handle_list_genai(&genai_runtime);
    }
    handle_chat_genai(&genai_runtime)
}

fn handle_init(args: &Args) -> Result<(), LlmProbeError> {
    let format_input = args.init.as_deref().unwrap_or("yaml");

    let (output_path, format) = if let Some(path) = &args.init_path {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("yaml");
        (path.clone(), ext.to_string())
    } else if format_input.contains('.') || (format_input != "yaml" && format_input != "json") {
        let path = PathBuf::from(format_input);
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("yaml")
            .to_string();
        (path, ext)
    } else {
        (
            PathBuf::from(format!("./llmctl.{}", format_input)),
            format_input.to_string(),
        )
    };

    init_config_file(&output_path, &format).map_err(LlmProbeError::ApiError)?;

    print_success(&format!("Config file created: {}", output_path.display()));
    Ok(())
}

fn handle_convert(convert_paths: &[PathBuf]) -> Result<(), LlmProbeError> {
    let input_path = &convert_paths[0];
    let output_path = convert_paths.get(1).map(|p| p.as_path());

    convert_config(input_path, output_path)
}

fn handle_list_genai(runtime_client: &GenaiRuntime) -> Result<(), LlmProbeError> {
    print_info("Fetching model list...");

    let runtime = tokio::runtime::Runtime::new()
        .map_err(|_| LlmProbeError::ApiError("Failed to create runtime".to_string()))?;
    let result = runtime.block_on(runtime_client.list_models())?;
    print_info(&format!("Model list source: {}", result.source.as_label()));

    format_model_list(&result.models);
    Ok(())
}

fn handle_chat_genai(runtime_client: &GenaiRuntime) -> Result<(), LlmProbeError> {
    let runtime = tokio::runtime::Runtime::new()
        .map_err(|_| LlmProbeError::ApiError("Failed to create runtime".to_string()))?;

    if runtime_client.is_stream_enabled() {
        runtime.block_on(runtime_client.stream_chat())?;
        Ok(())
    } else {
        let response = runtime.block_on(runtime_client.chat_completion())?;
        format_chat_response(&response);
        Ok(())
    }
}
