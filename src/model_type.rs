use std::fmt::Display;
use std::path::PathBuf;

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::ModelForCausalLM as Qwen2;
use candle_transformers::models::qwen3::ModelForCausalLM as Qwen3;
use candle_transformers::models::qwen3_vl::Qwen3VLModel as Qwen3Vl;
use hf_hub::api::sync::Api;

use crate::chat::{ChatMessage, ChatRole};
use crate::data::JsonMap;
use crate::model::{DynConfig, ModelPipeline};

/// Represents the size of model to use.
/// This is used to determine which model files to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelSize {
    /// 0 - 3B parameters
    Small,
    /// 3B - 6B parameters
    Medium,
    /// 6B - 20B parameters
    Large,
}

/// Represents the type of model to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelType {
    Qwen25Instruct,
    Qwen3(ModelSize),
    Qwen3InstructAbl,
    Qwen3Special,
    Qwen3Vl(ModelSize),
}

impl ModelType {
    /// Returns true if this model type supports chat functionality.
    pub fn can_chat(&self) -> bool {
        match self {
            ModelType::Qwen25Instruct
            | ModelType::Qwen3(_)
            | ModelType::Qwen3InstructAbl
            | ModelType::Qwen3Special
            | ModelType::Qwen3Vl(_) => true,
        }
    }

    /// Returns true if this model type supports "thinking" functionality.
    pub fn can_think(&self) -> bool {
        matches!(
            self,
            ModelType::Qwen25Instruct | ModelType::Qwen3(_) | ModelType::Qwen3Vl(_)
        )
    }

    /// Returns true if this model requires the think tag to be present regardless.
    pub fn must_think(&self) -> bool {
        matches!(self, ModelType::Qwen3(_) | ModelType::Qwen3Vl(_))
    }

    /// Returns true if this model needs "/think " in the prompt to enable thinking.
    pub fn use_think_in_prompt(&self) -> bool {
        matches!(self, ModelType::Qwen3(_) | ModelType::Qwen3Vl(_))
    }

    pub fn model_repo(&self) -> ModelRepo {
        match self {
            ModelType::Qwen25Instruct => ModelRepo::hub("Qwen/Qwen2.5-1.5B-Instruct"),
            ModelType::Qwen3(model_size) => match model_size {
                ModelSize::Small => ModelRepo::hub("Qwen/Qwen3-1.7B"),
                ModelSize::Medium => ModelRepo::hub("Qwen/Qwen3-4B"),
                ModelSize::Large => ModelRepo::hub("Qwen/Qwen3-8B"),
            },
            ModelType::Qwen3Vl(_) => ModelRepo::hub("Qwen/Qwen3-VL-2B-Instruct"),
            ModelType::Qwen3Special => ModelRepo::hub("DarkKitsune/qwen3-4b-instruct-special"),
            ModelType::Qwen3InstructAbl => {
                ModelRepo::hub("Goekdeniz-Guelmez/Josiefied-Qwen3-4B-abliterated-v2")
            }
        }
    }

    pub fn tokenizer_repo(&self) -> ModelRepo {
        self.model_repo()
    }

    pub fn tokenizer_json_name(&self) -> &'static str {
        "tokenizer.json"
    }

    pub fn model_names(&self) -> &[&'static str] {
        match self {
            ModelType::Qwen3(model_size) => match model_size {
                ModelSize::Small => &[
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                ],
                ModelSize::Medium => &[
                    "model-00001-of-00003.safetensors",
                    "model-00002-of-00003.safetensors",
                    "model-00003-of-00003.safetensors",
                ],
                ModelSize::Large => &[
                    "model-00001-of-00005.safetensors",
                    "model-00002-of-00005.safetensors",
                    "model-00003-of-00005.safetensors",
                    "model-00004-of-00005.safetensors",
                    "model-00005-of-00005.safetensors",
                ],
            },
            ModelType::Qwen3Special | ModelType::Qwen3InstructAbl => &[
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ],
            _ => &["model.safetensors"],
        }
    }

    /// Load and create a config for this type of model.
    pub fn create_config(&self, repo: &ModelRepo, api: &Api) -> DynConfig {
        let config_filename = repo.file_paths(&["config.json"], api).pop().unwrap();
        let config = std::fs::read_to_string(config_filename).unwrap();
        match self {
            ModelType::Qwen25Instruct => {
                let config = serde_json::from_str(&config).unwrap();
                DynConfig::Qwen2(config)
            }
            ModelType::Qwen3(_) | ModelType::Qwen3Special | ModelType::Qwen3InstructAbl => {
                let config = serde_json::from_str(&config).unwrap();
                DynConfig::Qwen3(config)
            }
            ModelType::Qwen3Vl(_) => {
                let config = serde_json::from_str(&config).unwrap();
                DynConfig::Qwen3Vl(config)
            }
        }
    }

    /// Create a pipeline for this type of model.
    pub fn create_pipeline(&self, config: &DynConfig, var: VarBuilder) -> ModelPipeline {
        match self {
            ModelType::Qwen25Instruct => {
                ModelPipeline::Qwen2(Qwen2::new(config.as_qwen2().unwrap(), var).unwrap())
            }
            ModelType::Qwen3(_) | ModelType::Qwen3Special | ModelType::Qwen3InstructAbl => {
                ModelPipeline::Qwen3(Qwen3::new(config.as_qwen3().unwrap(), var).unwrap())
            }
            ModelType::Qwen3Vl(_) => ModelPipeline::Qwen3Vl(Box::new(
                Qwen3Vl::new(config.as_qwen3_vl().unwrap(), var).unwrap(),
            )),
        }
    }

    /// Preprocesses the logits for this model type.
    pub fn process_logits(&self, logits: Tensor) -> Tensor {
        match self {
            ModelType::Qwen25Instruct
            | ModelType::Qwen3(_)
            | ModelType::Qwen3Special
            | ModelType::Qwen3InstructAbl
            | ModelType::Qwen3Vl(_) => {
                // Process logits for Qwen3 model
                logits
                    .squeeze(0)
                    .unwrap()
                    .squeeze(0)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .unwrap()
            }
        }
    }

    /// Creates a chat prompt meant for this type of Phi model.
    pub fn create_chat_prompt(&self, chat_system_prompt: impl AsRef<str>, chat_history: &[ChatMessage], extra_data: Option<&JsonMap>) -> String {
        let mut prompt = String::new();

        // Add the system prompt to the system section
        prompt.push_str(&format!("<|im_start|>system\n{}\n", chat_system_prompt.as_ref()));

        // Add extra data as key-value pairs for the model to understand
        if let Some(extra_data) = extra_data {
            prompt.push_str(&format!(
                "<notes>\n{}\n</notes>\n",
                serde_json::to_string_pretty(extra_data).unwrap()
            ));
        }

        // Finally end the system section
        prompt.push_str("<|im_end|>\n");

        // Add each message in the chat to the prompt as a new section
        for message in chat_history {
            match message.sender() {
                ChatRole::User => {
                    prompt.push_str(&format!(
                        "<|im_start|>user\n{}\n<|im_end|>\n",
                        message.content()
                    ));
                }
                ChatRole::Model => {
                    prompt.push_str(&format!(
                        "<|im_start|>assistant\n{}\n<|im_end|>\n",
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

    pub fn create_chat_message_begin_prompt(
        &self,
        sender: &ChatRole,
    ) -> String {
        format!("<|im_start|>{}\n", self.chat_role_name(sender))
    }

    pub fn create_chat_message_end_prompt(
        &self,
    ) -> String {
        "<|im_end|>\n".to_string()
    }

    pub fn chat_message_end_sequence(&self) -> &'static str {
        "<|im_end|>"
    }

    pub fn chat_role_name<'a>(&self, role: &'a ChatRole) -> &'a str {
        match role {
            ChatRole::Model => "assistant",
            ChatRole::User => "user",
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
