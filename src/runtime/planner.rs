use crate::config::{
    ConfigDiagnostic, DiagnosticSeverity, ResolvedRuntimeConfig, ValidationReport,
};
use crate::runtime::GenaiRuntime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeBackend {
    Genai,
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
            RuntimeBackend::Unsupported => "unsupported",
        }
    }

    pub fn reason_label(&self) -> Option<&str> {
        self.reason.as_deref()
    }
}

pub fn plan_runtime(resolved: &ResolvedRuntimeConfig) -> RuntimePlan {
    let mut diagnostics = Vec::new();

    if !GenaiRuntime::supports_resolved_config(resolved) {
        diagnostics.push(ConfigDiagnostic {
            severity: DiagnosticSeverity::Error,
            code: "runtime_adapter_not_supported_by_genai".to_string(),
            message: format!(
                "Adapter '{}' is not supported by the genai runtime",
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
