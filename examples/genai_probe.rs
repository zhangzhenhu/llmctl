use std::env;

/// Lightweight scaffold for genai capability spike.
///
/// Keep this example minimal and compilable; wire actual provider calls in
/// follow-up commits after we lock the exact genai API surface.
fn main() {
    let mut provider: Option<String> = None;
    let mut model: Option<String> = None;
    let mut base_url: Option<String> = None;
    let mut stream = false;
    let mut message: Option<String> = None;

    let args: Vec<String> = env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--provider" if i + 1 < args.len() => {
                provider = Some(args[i + 1].clone());
                i += 2;
            }
            "--model" if i + 1 < args.len() => {
                model = Some(args[i + 1].clone());
                i += 2;
            }
            "--base-url" if i + 1 < args.len() => {
                base_url = Some(args[i + 1].clone());
                i += 2;
            }
            "--message" if i + 1 < args.len() => {
                message = Some(args[i + 1].clone());
                i += 2;
            }
            "--stream" => {
                stream = true;
                i += 1;
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(2);
            }
        }
    }

    let provider = provider.unwrap_or_else(|| "openai".to_string());
    let model = model.unwrap_or_else(|| "gpt-4.1-nano".to_string());
    let message = message.unwrap_or_else(|| "hello".to_string());

    // Keep output deterministic for docs/genai_spike.md collection.
    println!("provider={provider}");
    println!("model={model}");
    println!(
        "base_url={}",
        base_url.unwrap_or_else(|| "<default>".to_string())
    );
    println!("stream={stream}");
    println!("message={message}");
    println!("status=scaffold_only");
}
