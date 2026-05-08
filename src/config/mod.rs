pub mod converter;
pub mod loader;
pub mod resolver;
pub mod schema;
pub mod validate;

pub use converter::convert_config;
pub use loader::{load_app_config, search_config_file};
pub use resolver::{
    list_builtin_adapters, resolve_runtime_config, BuiltinAdapterInfo, ResolvedRuntimeConfig,
};
pub use schema::{AppConfigV2, Args};
pub use validate::{
    validate_resolved_config, ConfigDiagnostic, DiagnosticSeverity, ValidationReport,
};
