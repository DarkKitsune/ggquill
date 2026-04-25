use std::fmt::Display;
use std::path::PathBuf;

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;
use candle_transformers::models::qwen2::ModelForCausalLM as Qwen2;
use candle_transformers::models::qwen3::ModelForCausalLM as Qwen3;
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
    Qwen3InstructQuantized,
    Qwen3InstructAbl,
    Qwen3Special,
}

impl ModelType {
    /// Returns true if this model type supports chat functionality.
    pub fn can_chat(&self) -> bool {
        match self {
            ModelType::Qwen25Instruct
            | ModelType::Qwen3(_)
            | ModelType::Qwen3InstructQuantized
            | ModelType::Qwen3InstructAbl
            | ModelType::Qwen3Special => true,
        }
    }

    /// Returns true if this model requires the think block to be present regardless.
    pub fn must_use_think(&self) -> bool {
        matches!(self, ModelType::Qwen3(_))
    }

    /// Returns true if this is a GGUF quantized model type, which requires special handling
    pub fn is_gguf_quantized(&self) -> bool {
        matches!(
            self,
            ModelType::Qwen3InstructQuantized | ModelType::Qwen3Special
        )
    }

    pub fn model_repo(&self) -> ModelRepo {
        match self {
            ModelType::Qwen25Instruct => ModelRepo::hub("Qwen/Qwen2.5-1.5B-Instruct"),
            ModelType::Qwen3(model_size) => match model_size {
                ModelSize::Small => ModelRepo::hub("Qwen/Qwen3-1.7B"),
                ModelSize::Medium => ModelRepo::hub("Qwen/Qwen3-4B"),
                ModelSize::Large => ModelRepo::hub("Qwen/Qwen3-8B"),
            },
            ModelType::Qwen3InstructQuantized => {
                ModelRepo::hub("mradermacher/Ophiuchi-Qwen3-14B-Instruct-i1-GGUF")
            }
            ModelType::Qwen3Special => {
                ModelRepo::hub("DarkKitsune/qwen3-4b-instruct-special-Q4_K_M-GGUF")
            }
            ModelType::Qwen3InstructAbl => {
                ModelRepo::hub("Goekdeniz-Guelmez/Josiefied-Qwen3-4B-abliterated-v2")
            }
        }
    }

    pub fn tokenizer_repo(&self) -> ModelRepo {
        match self {
            ModelType::Qwen3InstructQuantized => ModelRepo::hub("Qwen/Qwen3-14B"),
            ModelType::Qwen3Special => ModelRepo::hub("DarkKitsune/qwen3-4b-instruct-special"),
            _ => self.model_repo(),
        }
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
            ModelType::Qwen3InstructQuantized => &["Ophiuchi-Qwen3-14B-Instruct.i1-Q3_K_M.gguf"],
            ModelType::Qwen3Special => &["qwen3-4b-instruct-special-q4_k_m-imat.gguf"],
            ModelType::Qwen3InstructAbl => &[
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
            ModelType::Qwen3InstructQuantized => {
                unreachable!("GGUF quantized models should not use create_config")
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
            ModelType::Qwen3InstructQuantized => {
                unreachable!("GGUF quantized models should not use create_pipeline")
            }
        }
    }

    /// Create a pipeline for a GGUF quantized model of this type, which requires special handling.
    pub fn create_gguf_quantized_pipeline(
        &self,
        model_path: &PathBuf,
        device: &Device,
    ) -> ModelPipeline {
        match self {
            ModelType::Qwen3InstructQuantized | ModelType::Qwen3Special => {
                let mut reader = std::fs::File::open(model_path).unwrap();
                let content = gguf_file::Content::read(&mut reader).unwrap();
                ModelPipeline::QuantizedQwen3(
                    QuantizedQwen3::from_gguf(content, &mut reader, device).unwrap(),
                )
            }
            _ => unreachable!("GGUF quantized pipeline not supported for this model type"),
        }
    }

    /// Preprocesses the logits for this model type.
    pub fn process_logits(&self, logits: Tensor) -> Tensor {
        match self {
            ModelType::Qwen25Instruct
            | ModelType::Qwen3(_)
            | ModelType::Qwen3InstructQuantized
            | ModelType::Qwen3Special
            | ModelType::Qwen3InstructAbl => {
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
    pub fn create_chat_prompt(
        &self,
        chat_system_prompt: impl AsRef<str>,
        chat_history: &[ChatMessage],
        extra_data: Option<&JsonMap>,
    ) -> String {
        let mut prompt = String::new();

        // Add the system prompt to the system section
        prompt.push_str(&format!(
            "<|im_start|>system\n{}\n",
            chat_system_prompt.as_ref()
        ));

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
        let mut prompt = format!("<|im_start|>{}\n", self.chat_role_name(sender));
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
            ChatRole::System | ChatRole::Other(_) => {}
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
