use crate::config::Args;
use crate::config::{
    ConfigDiagnostic, DiagnosticSeverity, ResolvedRuntimeConfig, ValidationReport,
};
use crate::runtime::GenaiRuntime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    Genai,
    LegacyLlm,
    Unsupported,
}

#[derive(Debug, Clone)]
pub struct RuntimePlan {
    pub backend: RuntimeBackend,
    pub reason: Option<String>,
    pub extra_body_supported: bool,
    pub diagnostics: Vec<ConfigDiagnostic>,
}

impl RuntimePlan {
    pub fn backend_label(&self) -> &'static str {
        match self.backend {
            RuntimeBackend::Genai => "genai",
            RuntimeBackend::LegacyLlm => "legacy_llm",
            RuntimeBackend::Unsupported => "unsupported",
        }
    }

    pub fn reason_label(&self) -> Option<&str> {
        self.reason.as_deref()
    }

    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error)
    }
}

pub fn plan_runtime(resolved: &ResolvedRuntimeConfig, args: &Args) -> RuntimePlan {
    let mut diagnostics = Vec::new();

    if args.legacy_runtime {
        diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Warning,
            code: "legacy_runtime_requested".to_string(),
            message: "Using legacy llm runtime; behavior may differ from genai runtime".to_string(),
        });
        return RuntimePlan {
            backend: RuntimeBackend::LegacyLlm,
            reason: Some("explicit_legacy_runtime".to_string()),
            extra_body_supported: true,
            diagnostics,
        };
    }

    if !GenaiRuntime::supports_resolved_config(resolved) {
        diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Error,
            code: "runtime_adapter_not_supported_by_genai".to_string(),
            message: format!(
                "Adapter '{}' is not supported by the genai runtime; use --legacy-runtime only as a temporary compatibility path",
                resolved.adapter
            ),
        });
        return RuntimePlan {
            backend: RuntimeBackend::Unsupported,
            reason: Some(format!(
                "adapter_not_supported_by_genai ({})",
                resolved.adapter
            )),
            extra_body_supported: false,
            diagnostics,
        };
    }

    RuntimePlan {
        backend: RuntimeBackend::Genai,
        reason: None,
        extra_body_supported: true,
        diagnostics,
    }
}

pub fn merge_plan_diagnostics(report: &mut ValidationReport, plan: &RuntimePlan) {
    report.diagnostics.extend(plan.diagnostics.clone());
}
