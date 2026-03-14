#![allow(dead_code)]
pub use crate::backends::openai::{
    OpenAIAPIChatRequest,
    OpenAICompatible,
    // OpenAIResponseFormat,
    OpenAITool,
};
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use llm::chat::{
    ChatMessage,
    // ChatProvider, ChatResponse, Tool,
    Usage,
};
use llm::error::LLMError;
// use llm::providers::openai_compatible::reqwest;
// use llm::models::{ModelListRequest, ModelListResponse, ModelsProvider};
use llm::providers::openai_compatible::OpenAIStreamOptions;
// use llm::LLMProvider;
use llm::{default_call_type, FunctionCall, LLMProvider, ToolCall};
// use reqwest::Client;
use serde::{Deserialize, Serialize};
// use serde_json::json;
// use std::pin::Pin;
#[derive(Debug, Clone, Deserialize)]
struct UsageParsed {
    #[serde(rename = "prompt_tokens")]
    prompt_tokens: u32,
    #[serde(rename = "completion_tokens")]
    completion_tokens: u32,
    #[serde(rename = "total_tokens")]
    total_tokens: u32,
}
/// Stream response chunk that mimics OpenAI's streaming response format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamResponseWithReasoning {
    /// Array of choices in the response
    pub choices: Vec<StreamChoiceWithReasoning>,
    /// Usage metadata, typically present in the final chunk
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Individual choice in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChoiceWithReasoning {
    /// Delta containing the incremental content
    pub delta: StreamDeltaWithReasoning,
}

/// Delta content in a streaming response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamDeltaWithReasoning {
    /// The incremental content, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,

    /// The incremental tool calls, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// Streaming response structures
#[derive(Deserialize, Debug)]
pub struct OpenAIStreamChunk {
    pub choices: Vec<OpenAIStreamChoice>,
    pub usage: Option<Usage>,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIStreamChoice {
    pub delta: OpenAIStreamDelta,
}

#[derive(Deserialize, Debug)]
pub struct OpenAIStreamDelta {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

/// Tool call represents a function call that an LLM wants to make.
/// This is a standardized structure used across all providers.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct OpenAIStreamToolCall {
    /// The ID of the tool call.
    pub id: Option<String>,
    /// The type of the tool call (defaults to "function" if not provided).
    #[serde(rename = "type", default = "default_call_type")]
    pub call_type: String,
    /// The function to call.
    pub function: OpenAIStreamFunctionCall,
}

/// FunctionCall contains details about which function to call and with what arguments.
#[derive(Debug, Deserialize, Serialize, Clone, Eq, PartialEq)]
pub struct OpenAIStreamFunctionCall {
    /// The name of the function to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The arguments to pass to the function, typically serialized as a JSON string.
    pub arguments: String,
}

#[async_trait]
pub trait OpenAIWithReasoning: LLMProvider {
    /// Stream chat responses as `ChatMessage` structured objects, including usage information
    async fn chat_stream_struct_with_reasoning(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponseWithReasoning, LLMError>> + Send>>,
        LLMError,
    >;
    // async fn chat_with_reasoning(
    //     &self,
    //     messages: &[ChatMessage],
    // ) -> Result<Box<dyn ChatResponse>, LLMError> {
    //     self.chat_with_tools_with_reasoning(messages, None).await
    // }
    // async fn chat_with_tools_with_reasoning(
    //     &self,
    //     messages: &[ChatMessage],
    //     tools: Option<&[Tool]>,
    // ) -> Result<Box<dyn ChatResponse>, LLMError>;
}

#[async_trait]
impl OpenAIWithReasoning for OpenAICompatible {
    /// Stream chat responses as `ChatMessage` structured objects, including usage information
    async fn chat_stream_struct_with_reasoning(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponseWithReasoning, LLMError>> + Send>>,
        LLMError,
    > {
        let openai_msgs = self.provider.prepare_messages(messages);
        // Convert regular tools to OpenAI format for streaming
        let openai_tools: Option<Vec<OpenAITool>> =
            self.provider.config.tools.as_deref().map(|tools| {
                tools
                    .iter()
                    .map(|tool| OpenAITool::Function {
                        tool_type: tool.tool_type.clone(),
                        function: tool.function.clone(),
                    })
                    .collect()
            });
        let body = OpenAIAPIChatRequest {
            model: &self.provider.config.model,
            messages: openai_msgs,
            input: None,
            max_completion_tokens: self.provider.config.max_tokens,
            max_output_tokens: None,
            temperature: self.provider.config.temperature,
            stream: true,
            top_p: self.provider.config.top_p,
            top_k: self.provider.config.top_k,
            tools: openai_tools,
            tool_choice: self.provider.config.tool_choice.as_ref().cloned(),
            reasoning_effort: self
                .provider
                .config
                .reasoning_effort
                .as_deref()
                .map(|s| s.to_owned()),
            response_format: None,
            stream_options: Some(OpenAIStreamOptions {
                include_usage: true,
            }),
            extra_body: self.provider.config.extra_body.clone(),
        };
        let url = self
            .provider
            .config
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self
            .provider
            .client
            .post(url)
            .bearer_auth(&self.provider.config.api_key)
            .json(&body);
        if let Some(timeout) = self.provider.config.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        Ok(create_sse_stream_with_reasoning(
            response,
            self.provider.config.normalize_response,
        ))
    }

    // async fn chat_with_tools_with_reasoning(
    //     &self,
    //     messages: &[ChatMessage],
    //     tools: Option<&[Tool]>,
    // ) -> Result<Box<dyn ChatResponse>, LLMError> {
    //     // Use the common prepare_messages method from the OpenAI-compatible provider
    //     let openai_msgs = self.provider.prepare_messages(messages);
    //     let response_format: Option<OpenAIResponseFormat> = self
    //         .provider
    //         .config
    //         .json_schema
    //         .as_ref()
    //         .cloned()
    //         .map(|s| s.into());
    //     // Convert regular tools to OpenAI format
    //     let tool_calls = tools
    //         .map(|t| t.to_vec())
    //         .or_else(|| self.provider.config.tools.as_deref().map(|t| t.to_vec()));
    //     let mut openai_tools: Vec<OpenAITool> = Vec::new();
    //     // Add regular function tools
    //     if let Some(tools) = &tool_calls {
    //         for tool in tools {
    //             openai_tools.push(OpenAITool::Function {
    //                 tool_type: tool.tool_type.clone(),
    //                 function: tool.function.clone(),
    //             });
    //         }
    //     }
    //     let final_tools = if openai_tools.is_empty() {
    //         None
    //     } else {
    //         Some(openai_tools)
    //     };
    //     let request_tool_choice = if final_tools.is_some() {
    //         self.provider.config.tool_choice.as_ref().cloned()
    //     } else {
    //         None
    //     };
    //     let body = OpenAIAPIChatRequest {
    //         model: &self.provider.config.model,
    //         messages: openai_msgs,
    //         input: None,
    //         max_completion_tokens: self.provider.config.max_tokens,
    //         max_output_tokens: None,
    //         temperature: self.provider.config.temperature,
    //         stream: false,
    //         top_p: self.provider.config.top_p,
    //         top_k: self.provider.config.top_k,
    //         tools: final_tools,
    //         tool_choice: request_tool_choice,
    //         reasoning_effort: self
    //             .provider
    //             .config
    //             .reasoning_effort
    //             .as_deref()
    //             .map(|s| s.to_owned()),
    //         response_format,
    //         stream_options: None,
    //         extra_body: self.provider.config.extra_body.clone(),
    //     };
    //     let url = self
    //         .provider
    //         .config
    //         .base_url
    //         .join("chat/completions")
    //         .map_err(|e| LLMError::HttpError(e.to_string()))?;
    //     let mut request = self
    //         .provider
    //         .client
    //         .post(url)
    //         .bearer_auth(&self.provider.config.api_key)
    //         .json(&body);
    //     if log::log_enabled!(log::Level::Trace) {
    //         if let Ok(json) = serde_json::to_string(&body) {
    //             log::trace!("OpenAI request payload: {}", json);
    //         }
    //     }
    //     if let Some(timeout) = self.provider.config.timeout_seconds {
    //         request = request.timeout(std::time::Duration::from_secs(timeout));
    //     }
    //     let response = request.send().await?;
    //     log::debug!("OpenAI HTTP status: {}", response.status());
    //     if !response.status().is_success() {
    //         let status = response.status();
    //         let error_text = response.text().await?;
    //         return Err(LLMError::ResponseFormatError {
    //             message: format!("OpenAI API returned error status: {status}"),
    //             raw_response: error_text,
    //         });
    //     }
    //     // Parse the successful response
    //     let resp_text = response.text().await?;
    //     let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
    //         serde_json::from_str(&resp_text);
    //     match json_resp {
    //         Ok(response) => Ok(Box::new(response)),
    //         Err(e) => Err(LLMError::ResponseFormatError {
    //             message: format!("Failed to decode OpenAI API response: {e}"),
    //             raw_response: resp_text,
    //         }),
    //     }
    // }
}
/// Creates a structured SSE stream that returns `StreamResponse` objects
///
/// Buffer required to accumulate JSON payload lines that are split across multiple SSE chunks
pub fn create_sse_stream_with_reasoning(
    response: reqwest::Response,
    normalize_response: bool,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<StreamResponseWithReasoning, LLMError>> + Send>> {
    struct SSEStreamParser {
        event_buffer: String,
        tool_buffer: ToolCall,
        usage: Option<Usage>,
        results: Vec<Result<StreamResponseWithReasoning, LLMError>>,
        normalize_response: bool,
    }
    impl SSEStreamParser {
        fn new(normalize_response: bool) -> Self {
            Self {
                event_buffer: String::new(),
                usage: None,
                results: Vec::new(),
                tool_buffer: ToolCall {
                    id: String::new(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: String::new(),
                        arguments: String::new(),
                    },
                },
                normalize_response,
            }
        }

        /// Push the current `tool_buffer` as a `StreamResponse` and reset it
        fn push_tool_call(&mut self) {
            if self.normalize_response && !self.tool_buffer.function.name.is_empty() {
                self.results.push(Ok(StreamResponseWithReasoning {
                    choices: vec![StreamChoiceWithReasoning {
                        delta: StreamDeltaWithReasoning {
                            content: None,
                            reasoning_content: None,
                            tool_calls: Some(vec![self.tool_buffer.clone()]),
                        },
                    }],
                    usage: None,
                }));
            }
            self.tool_buffer = ToolCall {
                id: String::new(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: String::new(),
                    arguments: String::new(),
                },
            };
        }

        /// Parse the accumulated event_buffer as one SSE event
        fn parse_event(&mut self) {
            let mut data_payload = String::new();
            for line in self.event_buffer.lines() {
                if let Some(data) = line.strip_prefix("data: ") {
                    if data == "[DONE]" {
                        self.push_tool_call();
                        if let Some(usage) = self.usage.clone() {
                            self.results.push(Ok(StreamResponseWithReasoning {
                                choices: vec![StreamChoiceWithReasoning {
                                    delta: StreamDeltaWithReasoning {
                                        content: None,
                                        reasoning_content: None,
                                        tool_calls: None,
                                    },
                                }],
                                usage: Some(usage),
                            }));
                        }
                        return;
                    }
                    data_payload.push_str(data);
                } else {
                    data_payload.push_str(line);
                }
            }
            if data_payload.is_empty() {
                return;
            }
            if let Ok(response) = serde_json::from_str::<OpenAIStreamChunk>(&data_payload) {
                if let Some(resp_usage) = response.usage.clone() {
                    self.usage = Some(resp_usage);
                }
                for choice in &response.choices {
                    let content = choice.delta.content.clone();
                    // Map StreamToolCall (some fields are optional) to ToolCall
                    let tool_calls: Option<Vec<ToolCall>> =
                        choice.delta.tool_calls.clone().map(|calls| {
                            calls
                                .into_iter()
                                .map(|c| ToolCall {
                                    id: c.id.unwrap_or_default(),
                                    call_type: c.call_type,
                                    function: FunctionCall {
                                        name: c.function.name.unwrap_or_default(),
                                        arguments: c.function.arguments,
                                    },
                                })
                                .collect::<Vec<ToolCall>>()
                        });
                    if content.is_some() || tool_calls.is_some() {
                        if self.normalize_response && tool_calls.is_some() {
                            // If normalize_response is enabled, accumulate tool call outputs
                            if let Some(calls) = &tool_calls {
                                for call in calls {
                                    // println!("Accumulating tool call: {:?}", call);
                                    if !call.function.name.is_empty() {
                                        self.push_tool_call();
                                        self.tool_buffer.function.name = call.function.name.clone();
                                    }
                                    if !call.function.arguments.is_empty() {
                                        self.tool_buffer
                                            .function
                                            .arguments
                                            .push_str(&call.function.arguments);
                                    }
                                    if !call.id.is_empty() {
                                        self.tool_buffer.id = call.id.clone();
                                    }
                                    if !call.call_type.is_empty() {
                                        self.tool_buffer.call_type = call.call_type.clone();
                                    }
                                }
                            }
                        } else {
                            self.push_tool_call();
                            self.results.push(Ok(StreamResponseWithReasoning {
                                choices: vec![StreamChoiceWithReasoning {
                                    delta: StreamDeltaWithReasoning {
                                        content,
                                        reasoning_content: None,
                                        tool_calls,
                                    },
                                }],
                                usage: None,
                            }));
                        }
                    }
                    let reasoning_content = choice.delta.reasoning_content.clone();
                    if reasoning_content.is_some() {
                        self.results.push(Ok(StreamResponseWithReasoning {
                            choices: vec![StreamChoiceWithReasoning {
                                delta: StreamDeltaWithReasoning {
                                    content: None,
                                    reasoning_content: reasoning_content,
                                    tool_calls: None,
                                },
                            }],
                            usage: None,
                        }));
                    }
                }
            }
        }
    }

    let bytes_stream = response.bytes_stream();
    let stream = bytes_stream
        .scan(SSEStreamParser::new(normalize_response), |parser, chunk| {
            let results = match chunk {
                Ok(bytes) => {
                    let text = String::from_utf8_lossy(&bytes);
                    for line in text.lines() {
                        let line = line.trim_end();
                        if line.is_empty() {
                            // Blank line: end of event, parse accumulated event_buffer
                            parser.parse_event();
                            parser.event_buffer.clear();
                        } else {
                            parser.event_buffer.push_str(line);
                            parser.event_buffer.push('\n');
                        }
                    }
                    parser.results.drain(..).collect::<Vec<_>>()
                }
                Err(e) => vec![Err(LLMError::HttpError(e.to_string()))],
            };
            futures::future::ready(Some(results))
        })
        .flat_map(futures::stream::iter);
    Box::pin(stream)
}

// 非流式

// /// Generic OpenAI-compatible chat response
// #[derive(Deserialize, Debug)]
// pub struct OpenAIChatResponse {
//     pub choices: Vec<OpenAIChatChoice>,
//     pub usage: Option<Usage>,
// }

// #[derive(Deserialize, Debug)]
// pub struct OpenAIChatChoice {
//     pub message: OpenAIChatMsg,
// }

// #[derive(Deserialize, Debug)]
// pub struct OpenAIChatMsg {
//     pub role: String,
//     pub content: Option<String>,
//     pub reasoning_content: Option<String>,
//     pub tool_calls: Option<Vec<ToolCall>>,
// }

// impl ChatResponse for OpenAIChatResponse {
//     fn text(&self) -> Option<String> {
//         self.choices.first().and_then(|c| c.message.content.clone())
//     }

//     fn tool_calls(&self) -> Option<Vec<ToolCall>> {
//         self.choices
//             .first()
//             .and_then(|c| c.message.tool_calls.clone())
//     }

//     fn usage(&self) -> Option<Usage> {
//         self.usage.clone()
//     }
//     fn thinking(&self) -> Option<String> {
//         self.choices
//             .first()
//             .and_then(|c| c.message.reasoning_content.clone())
//     }
// }
// impl std::fmt::Display for OpenAIChatResponse {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         match (
//             &self.choices.first().unwrap().message.content,
//             &self.choices.first().unwrap().message.tool_calls,
//         ) {
//             (Some(content), Some(tool_calls)) => {
//                 for tool_call in tool_calls {
//                     write!(f, "{tool_call}")?;
//                 }
//                 write!(f, "{content}")
//             }
//             (Some(content), None) => write!(f, "{content}"),
//             (None, Some(tool_calls)) => {
//                 for tool_call in tool_calls {
//                     write!(f, "{tool_call}")?;
//                 }
//                 Ok(())
//             }
//             (None, None) => write!(f, ""),
//         }
//     }
// }
