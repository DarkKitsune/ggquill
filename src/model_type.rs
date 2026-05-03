use std::fmt::Display;
use std::path::PathBuf;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;
use candle_transformers::models::qwen3::ModelForCausalLM as Qwen3;
use hf_hub::api::sync::Api;

use crate::chat::{ChatMessage, ChatRole, Tool};
use crate::data::StringMap;
use crate::model::{DynConfig, ModelPipeline};

/// Represents the size of model to use.
/// This is used to determine which model files to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSize {
    /// A super lightweight model.
    /// Usually around 0 - 7B parameters.
    Small,
    /// A medium-weight model balanced between speed and power.
    /// Usually around 8B - 15B parameters.
    Medium,
    /// A heavyweight model sacrificing efficiency for power.
    /// Usually around 16B+ parameters.
    Large,
}

/// Represents the base architecture of the model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelArchitecture {
    Qwen3,
}

/// Represents the type of model to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    Qwen3Instruct(ModelSize),
}

impl ModelType {
    /// Get the base architecture of this model type.
    pub fn architecture(&self) -> ModelArchitecture {
        match self {
            ModelType::Qwen3Instruct(_) => ModelArchitecture::Qwen3,
        }
    }

    /// Returns true if this model type supports chat functionality.
    pub fn can_chat(&self) -> bool {
        match self.architecture() {
            ModelArchitecture::Qwen3 => true,
        }
    }

    /// Returns true if this model requires the think block to be present regardless.
    pub fn must_use_think(&self) -> bool {
        matches!(self.architecture(), ModelArchitecture::Qwen3)
    }

    /// Returns true if this is a GGUF quantized model type, which requires special handling
    pub fn is_gguf_quantized(&self) -> bool {
        matches!(self, ModelType::Qwen3Instruct(_))
    }

    pub fn model_repo(&self) -> ModelRepo {
        match self {
            // Qwen 3 instruct (preferably abliterated)
            ModelType::Qwen3Instruct(model_size) => match model_size {
                ModelSize::Small => {
                    ModelRepo::hub("DarkKitsune/qwen3-4b-instruct-special-Q4_K_M-GGUF")
                }
                ModelSize::Medium => {
                    ModelRepo::hub("mradermacher/Ophiuchi-Qwen3-14B-Instruct-i1-GGUF")
                }
                ModelSize::Large => {
                    unimplemented!("No Qwen3 Instruct model available in {:?} yet", model_size)
                }
            },
        }
    }

    pub fn tokenizer_repo(&self) -> ModelRepo {
        match self {
            ModelType::Qwen3Instruct(_) => ModelRepo::hub("DarkKitsune/qwen3-4b-instruct-special"),
        }
    }

    pub fn model_names(&self) -> &[&'static str] {
        match self {
            ModelType::Qwen3Instruct(model_size) => match model_size {
                ModelSize::Small => &["qwen3-4b-instruct-special-q4_k_m-imat.gguf"],
                ModelSize::Medium => &["Ophiuchi-Qwen3-14B-Instruct.i1-Q3_K_S.gguf"],
                ModelSize::Large => {
                    unimplemented!("No Qwen3 Instruct model available in {:?} yet", model_size)
                }
            },
        }
    }

    pub fn tokenizer_json_name(&self) -> &'static str {
        "tokenizer.json"
    }

    /// Load and create a config for this type of model.
    pub fn create_config(&self, repo: &ModelRepo, api: &Api) -> DynConfig {
        let config_filename = repo.file_paths(&["config.json"], api).pop().unwrap();
        let config = std::fs::read_to_string(config_filename).unwrap();
        match self.architecture() {
            ModelArchitecture::Qwen3 => {
                let config = serde_json::from_str(&config).unwrap();
                DynConfig::Qwen3(config)
            }
        }
    }

    /// Create a pipeline for this type of model.
    pub fn create_pipeline(&self, config: &DynConfig, var: VarBuilder) -> ModelPipeline {
        match self.architecture() {
            ModelArchitecture::Qwen3 => {
                ModelPipeline::Qwen3(Qwen3::new(config.as_qwen3().unwrap(), var).unwrap())
            }
        }
    }

    /// Create a pipeline for a GGUF quantized model of this type, which requires special handling.
    pub fn create_gguf_quantized_pipeline(
        &self,
        model_path: &PathBuf,
        device: &Device,
    ) -> ModelPipeline {
        match self.architecture() {
            ModelArchitecture::Qwen3 => {
                let mut reader = std::fs::File::open(model_path).unwrap();
                let content = gguf_file::Content::read(&mut reader).unwrap();
                ModelPipeline::QuantizedQwen3(
                    QuantizedQwen3::from_gguf(content, &mut reader, device).unwrap(),
                )
            }
        }
    }

    /// Preprocesses the logits for this model type.
    pub fn process_logits(&self, logits: Tensor) -> Tensor {
        match self.architecture() {
            ModelArchitecture::Qwen3 => logits
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap(),
        }
    }

    /// Creates a chat prompt meant for this type of model.
    pub fn create_chat_prompt(
        &self,
        chat_system_prompt: impl AsRef<str>,
        chat_history: &[ChatMessage],
        how_to_respond: &[&str],
        extra_data: Option<&StringMap>,
        tools: &[Tool],
    ) -> String {
        let mut prompt = String::new();

        // Add the system prompt to the system section
        prompt.push_str(&format!(
            "<|im_start|>system\n{}\n",
            chat_system_prompt.as_ref()
        ));

        // If there are instructions for how to respond, then we add them to the system prompt as a bullet list in a <how_to_respond> block
        if !how_to_respond.is_empty() {
            prompt.push_str("---\n# How you should respond:\n<how_to_respond>\n");
            for instruction in how_to_respond {
                // Replace newlines in the instruction with spaces so that it doesn't break the formatting of the bullet points
                let instruction = instruction.replace("\n", " ");

                // Push the instruction to the prompt as a bullet point
                prompt.push_str(&format!("- {}\n", instruction));
            }
            prompt.push_str("</how_to_respond>\n");
        }

        // If there is extra data to provide to the model, then we add it to the system prompt in a <knowledge> block
        if let Some(extra_data) = extra_data {
            prompt.push_str("---\n# What you know:\n<knowledge>\n");

            // Add each key-value pair in the extra data with formatting as a bullet list
            for (key, value) in extra_data {
                // Replace newlines in the key with spaces so that it doesn't break the formatting of the bullet points
                let key = key.replace("\n", " ");

                // The value will be in a code block, so replace backticks in the value with escaped backticks
                let value = value.replace("`", "\\`");

                // For each newline in value we need to add an extra indentation level so that the formatting of the code block isn't broken by newlines
                let value = value.replace("\n", "\n    ");

                // Push the key-value pair to the prompt as a bullet point with the key and the value in a code block
                prompt.push_str(&format!("- {}: `{}`\n", key, value));
            }

            // End the knowledge block
            prompt.push_str("</knowledge>\n");
        }

        // If tools are provided, then we add them to the system prompt in a <tools> block
        if !tools.is_empty() {
            // Start the tools block with a header
            prompt.push_str("---\n# Tools\nYou have access to the following tools:\n<tools>\n");
            // Present the tool JSON schemas to the model as a JSON array
            let tools_vec = tools
                .iter()
                .map(Tool::to_json_schema)
                .collect::<Vec<_>>();
            let tools_json = serde_json::to_string_pretty(&tools_vec).unwrap();
            prompt.push_str(&tools_json);
            println!("Tools JSON:\n{}\n", tools_json);
            // End the tools block
            prompt.push_str("\n</tools>\n");
            
            // Instruct the model on how to format a tool call
            prompt.push_str("When calling a tool, you must **only** use the following format: ```\n\
            <tool_call>\n\
            { \"tool\": \"tool_name\", \"args\": { ... } }\n\
            </tool_call>\n\
            ```\n\
            Do not use any other format or add unnecessary text before or after the tool call.");
        }

        // Finally end the system section
        prompt.push_str("<|im_end|>\n");

        // Add each message in the chat to the prompt as a new section
        for message in chat_history {
            match message.sender() {
                ChatRole::User => {
                    prompt.push_str(&format!(
                        "<|im_start|>user\n{}{}\n<|im_end|>\n",
                        if self.must_use_think() {
                            "/no_think "
                        } else {
                            ""
                        },
                        message.content()
                    ));
                }
                ChatRole::Assistant => {
                    prompt.push_str(&format!(
                        "<|im_start|>assistant\n{}{}\n<|im_end|>\n",
                        if self.must_use_think() {
                            "<think>\n\n</think>\n"
                        } else {
                            ""
                        },
                        message.content()
                    ));
                }
                ChatRole::System => {
                    prompt.push_str(&format!(
                        "<|im_start|>system\n{}\n<|im_end|>\n",
                        message.content()
                    ));
                }
                ChatRole::Tool => {
                    prompt.push_str(&format!(
                        "<|im_start|>tool\n{}\n<|im_end|>\n",
                        message.content()
                    ));
                }
                ChatRole::Other(name) => {
                    prompt.push_str(&format!(
                        "<|im_start|>{}\n{}\n<|im_end|>\n",
                        name,
                        message.content()
                    ));
                }
            }
        }

        prompt
    }

    pub fn create_chat_message_begin_prompt(&self, sender: &ChatRole) -> String {
        // Start prompt
        let mut prompt = format!("<|im_start|>{}\n", self.chat_role_name(sender));

        // Additional propmt
        match sender {
            // If this is the user and we must include the think block, then we include the /no_think command in the prompt
            ChatRole::User => {
                if self.must_use_think() {
                    prompt.push_str("/no_think ");
                }
            }
            // If this is the assistant and we must think, then we need to include the empty think block
            ChatRole::Assistant => {
                if self.must_use_think() {
                    prompt.push_str("<think>\n\n</think>\n");
                }
            }
            _ => {}
        }

        prompt
    }

    pub fn create_chat_message_end_prompt(&self) -> String {
        "<|im_end|>\n".to_string()
    }

    pub fn chat_message_end_sequence(&self) -> &'static str {
        "<|im_end|>"
    }

    pub fn chat_role_name<'a>(&self, role: &'a ChatRole) -> &'a str {
        match role {
            ChatRole::Assistant => "assistant",
            ChatRole::User => "user",
            ChatRole::System => "system",
            ChatRole::Tool => "tool",
            ChatRole::Other(name) => name,
        }
    }
}

/// Represents a model repo, either remote on HuggingFace or local
pub enum ModelRepo {
    Hub(String),
    Local(String),
}

impl ModelRepo {
    pub fn hub(repo: impl Display) -> Self {
        Self::Hub(repo.to_string())
    }

    pub fn local(path: impl Display) -> Self {
        Self::Local(path.to_string())
    }

    /// Get the file paths for the given file names, either from the HuggingFace Hub or from the local filesystem depending on the repo type.
    pub fn file_paths(&self, file_names: &[&str], api: &Api) -> Vec<PathBuf> {
        match self {
            ModelRepo::Hub(repo) => {
                let api_repo = api.model(repo.clone());
                file_names
                    .iter()
                    .map(|&file_name| {
                        api_repo.get(file_name).unwrap_or_else(|e| {
                            panic!("Failed to get file {} from {}: {}", file_name, repo, e)
                        })
                    })
                    .collect()
            }
            ModelRepo::Local(path) => file_names
                .iter()
                .map(|&file_name| PathBuf::from(path).join(file_name))
                .collect(),
        }
    }
}
