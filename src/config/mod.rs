pub mod converter;
pub mod legacy;
pub mod loader;
pub mod resolver;
pub mod schema;
pub mod validate;

pub use converter::convert_config;
pub use legacy::load_app_config;
pub use loader::search_config_file;
pub use resolver::{
    list_builtin_provider_presets, resolve_runtime_config, BuiltinPresetInfo, ResolvedRuntimeConfig,
};
pub use schema::{AppConfigV2, Args, RuntimeConfig};
pub use validate::{
    validate_resolved_config, ConfigDiagnostic, DiagnosticSeverity, ValidationReport,
};
