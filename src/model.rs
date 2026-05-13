use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::rc::Rc;
use std::time::Instant;

use anyhow::{Error as E, Result};

use candle_transformers::models::quantized_qwen3::ModelWeights as QuantizedQwen3;
use candle_transformers::models::qwen2::{Config as Qwen2Config, ModelForCausalLM as Qwen2};
use candle_transformers::models::qwen3::{Config as Qwen3Config, ModelForCausalLM as Qwen3};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use crate::inference::InferIter;
use crate::model_type::ModelType;
use crate::prelude::InferParams;
use crate::token_string::{IntoTokenString, TokenString};

pub const THINK_TEMP_MULTIPLIER: f64 = 0.85;
const TPS_MEASUREMENT_INTERVAL: f64 = 4.0; // Measure tokens per second every n seconds
const TPS_MEASUREMENT_THRESHOLD: usize = 8; // Minimum number of tokens generated before we consider submitting a timing for TPS measurement

#[derive(Clone)]
pub struct Model {
    model_type: ModelType,
    pipeline: ModelPipeline,
    tokenizer: Tokenizer,
    device: Device,
    eos_token: u32,
    seed: u64,
    // The measured average tokens per second for this model, as well as the last time it was updated
    avg_tokens_per_second: Rc<RefCell<Option<(f64, Instant)>>>,
}

impl Model {
    pub fn new(model_type: ModelType, seed: u64, use_cuda: bool) -> Result<Self> {
        let (is_cuda, device) = if use_cuda && candle_core::utils::cuda_is_available() {
            (true, Device::new_cuda(0).unwrap())
        } else {
            (false, Device::Cpu)
        };
        println!("Using device: {:?}, is_cuda: {}", device, is_cuda);

        // Get the model repos
        let api = Api::new()?;
        let model_repo = model_type.model_repo();
        let tokenizer_repo = model_type.tokenizer_repo();

        // Get the tokenizer and model files
        let tokenizer_filename = tokenizer_repo
            .file_paths(&[model_type.tokenizer_json_name()], &api)
            .pop()
            .unwrap();
        let model_filenames = model_type
            .model_names()
            .iter()
            .map(|name| model_repo.file_paths(&[name], &api).pop().unwrap())
            .collect::<Vec<_>>();

        let pipeline = if model_type.is_gguf_quantized() {
            model_type.create_gguf_quantized_pipeline(&model_filenames[0], &device)
        } else {
            let dtype = if is_cuda { DType::BF16 } else { DType::F32 };

            // Create model config
            let config = model_type.create_config(&model_repo, &api);

            // Create VarBuilder
            let vb =
                unsafe { VarBuilder::from_mmaped_safetensors(&model_filenames, dtype, &device)? };

            // Create pipeline
            model_type.create_pipeline(&config, vb)
        };

        // Create tokenizer
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        // Get the end of text token
        let eos_token = *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap();

        Ok(Self {
            model_type,
            pipeline,
            tokenizer,
            device: device.clone(),
            seed,
            eos_token,
            avg_tokens_per_second: Rc::new(RefCell::new(None)),
        })
    }

    /// Forwards the given input tokens through the model and returns the output logits.
    pub fn forward(&mut self, xs: &Tensor, start_pos: usize, seq_len: usize) -> Tensor {
        self.pipeline.forward(xs, start_pos, seq_len)
    }

    pub fn model_type(&self) -> ModelType {
        self.model_type
    }

    pub fn seed(&self) -> u64 {
        self.seed
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn new_token_string(&self) -> TokenString {
        let mut cloned_self = self.clone();
        cloned_self.clear_cache();
        TokenString::new(Vec::new(), Rc::new(RefCell::new(cloned_self)))
    }

    pub fn tokenize_str(&self, text: impl Display) -> TokenString {
        // Tokenize the text
        let tokens = self.tokenizer.encode(text.to_string(), true).unwrap();

        // Get the token ids
        let token_ids = tokens.get_ids().to_vec();

        // Clone the model and clear the cache for the new TokenString
        let mut cloned_self = self.clone();
        cloned_self.clear_cache();

        TokenString::new(token_ids, Rc::new(RefCell::new(cloned_self)))
    }

    pub fn tokenize(&self, text: impl IntoTokenString) -> TokenString {
        text.into_token_string(self)
    }

    pub(crate) fn detokenize(&self, tokens: impl AsRef<[u32]>) -> String {
        // Decode the tokens into a string
        self.tokenizer
            .decode(tokens.as_ref(), false)
            .map_err(E::msg)
            .unwrap()
    }

    /// Attempt to get the token matching a given string (if any)
    pub fn get_token(&self, s: impl Display) -> Result<u32> {
        match self.tokenizer.get_vocab(true).get(&s.to_string()) {
            Some(token) => Ok(*token),
            None => anyhow::bail!("cannot find the token for {:?}", s.to_string()),
        }
    }

    /// Get the EOS token id for this model's tokenizer
    pub fn eos_token(&self) -> u32 {
        self.eos_token
    }

    /// Increment the seed and return the new value. Useful for iterative tasks.
    pub fn next_seed(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(1);
        self.seed
    }

    /// Consume self to produce an iterator that yields tokens generated by the model.
    /// Returns an error if the prompt is empty.
    pub fn infer_iter(
        self,
        prompt: impl IntoTokenString,
        params: &InferParams,
    ) -> Result<InferIter> {
        // Tokenize the prompt
        let prompt = self.tokenize(prompt);

        // Fail if the prompt is empty
        if prompt.is_empty() {
            anyhow::bail!("prompt was empty")
        }

        // Create the iterator
        let device = self.device.clone();
        Ok(InferIter::new(self, device, prompt, params))
    }

    /// Predict the text which follows the given prompt.
    pub fn predict_next(self, prompt: impl IntoTokenString, params: &InferParams) -> InferIter {
        self.infer_iter(prompt, params).unwrap()
    }

    /// Clear the model's KV cache.
    pub fn clear_cache(&mut self) {
        self.pipeline.clear_cache();
    }

    /// Returns the amount of time since the last timing was submitted for this model, in seconds. Returns None if no timing has been submitted yet.
    pub(crate) fn time_since_last_timing(&self) -> Option<f64> {
        self.avg_tokens_per_second
            .borrow()
            .map(|(_, last_updated)| last_updated.elapsed().as_secs_f64())
    }

    /// Submit a timing for a generation and update the average tokens per second for this model.
    pub(crate) fn submit_timing(&self, tokens_generated: usize, seconds: f64) {
        // Exit early if we haven't generated enough tokens to consider this timing for TPS measurement, to avoid skewing the average with outliers
        if tokens_generated < TPS_MEASUREMENT_THRESHOLD {
            return;
        }

        // Exit early if we submitted a timing recently
        if let Some(time_since_last) = self.time_since_last_timing()
            && time_since_last < TPS_MEASUREMENT_INTERVAL
        {
            // If the last timing was submitted recently then we skip this timing to avoid skewing the average with outliers
            return;
        }

        let tokens_per_second = tokens_generated as f64 / seconds;
        let mut avg_tokens_per_second = self.avg_tokens_per_second.borrow_mut();
        if let Some((avg, _)) = *avg_tokens_per_second {
            let new_avg = (avg + tokens_per_second) / 2.0;
            *avg_tokens_per_second = Some((new_avg, Instant::now()));
        } else {
            *avg_tokens_per_second = Some((tokens_per_second, Instant::now()));
        }
        println!(
            "\nGenerated {} tokens in {:.2} seconds (Avg: {:.2} tokens/sec)\n",
            tokens_generated,
            seconds,
            avg_tokens_per_second.unwrap().0
        );
    }

    /// Get the average tokens per second for this model. Returns none if time has not been measured.
    pub fn average_tokens_per_second(&self) -> Option<f64> {
        self.avg_tokens_per_second.borrow().map(|(avg, _)| avg)
    }
}

/// Contains a pipeline, could be one of multiple types.
#[derive(Debug, Clone)]
pub enum ModelPipeline {
    Qwen2(Qwen2),
    Qwen3(Qwen3),
    QuantizedQwen3(QuantizedQwen3),
}

impl ModelPipeline {
    /// Forward the given input through the model pipeline and return the output logits.
    pub fn forward(&mut self, xs: &Tensor, start_pos: usize, _seq_len: usize) -> Tensor {
        match self {
            ModelPipeline::Qwen2(qwen2) => qwen2.forward(xs, start_pos).unwrap(),
            ModelPipeline::Qwen3(qwen3) => qwen3.forward(xs, start_pos).unwrap(),
            ModelPipeline::QuantizedQwen3(qwen3) => qwen3.forward(xs, start_pos).unwrap(),
        }
    }

    /// Clear the pipeline's KV cache.
    pub fn clear_cache(&mut self) {
        match self {
            ModelPipeline::Qwen2(qwen2) => qwen2.clear_kv_cache(),
            ModelPipeline::Qwen3(qwen3) => qwen3.clear_kv_cache(),
            ModelPipeline::QuantizedQwen3(qwen3) => qwen3.clear_kv_cache(),
        }
    }
}

/// Contains a model config
#[derive(Debug, Clone)]
pub enum DynConfig {
    Qwen2(Qwen2Config),
    Qwen3(Qwen3Config),
}

impl DynConfig {
    pub fn as_qwen2(&self) -> Option<&Qwen2Config> {
        match self {
            DynConfig::Qwen2(config) => Some(config),
            _ => None,
        }
    }

    pub fn as_qwen3(&self) -> Option<&Qwen3Config> {
        match self {
            DynConfig::Qwen3(config) => Some(config),
            _ => None,
        }
    }
}

/// Nudge a temperature value towards 1.0 without ever reaching it.
/// Useful for iterative tasks.
pub fn nudge_temperature(temp: &mut f64) {
    *temp += (1.0 - *temp) * 0.15;
}
