use std::time::Instant;

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::{
    model::Model,
    prelude::{IntoTokenString, TokenString},
};

pub const TARGET_CONTEXT_WINDOW_HIGH: usize = 6100; // Try to aim for <1 GB VRAM usage for Qwen3-14b's KV cache
pub const TARGET_CONTEXT_WINDOW_LOW: usize = TARGET_CONTEXT_WINDOW_HIGH / 2; // Try to aim for <500 MB VRAM usage for KV cache

/// The more memory we devote to a single inference, the more stable it will be after long contexts, but the more VRAM it will use.
/// Behind the scenes this controls the target context window size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryPriority {
    /// Devote more memory to a single inference, allowing for more stable output after long contexts, but using more VRAM.
    /// Best for things like chat conversations, agents, or other long and complex tasks.
    High,
    /// Devote less memory to a single inference, which may cause the output to become less
    /// stable after long contexts, but will use less VRAM.
    /// Best for simple tasks such as summarizing text or generating short responses,
    /// where a single task is unlikely to use the entire context window.
    Low,
}

/// Parameters for inference. This is used to configure the inference process, such as the repeat penalty.
#[derive(Debug, Clone)]
pub struct InferParams {
    /// The temperature to apply to the logits. 0.0 results in deterministic sampling, while ~0.7 is creative.
    pub temperature: f64,
    /// The repeat penalty to apply to the logits.
    /// 1.0 means no penalty, while values > 1.0 penalize repeated tokens and values < 1.0 encourage them.
    pub repeat_penalty: f32,
    /// The number of last tokens to apply the repeat penalty to.
    /// 0 means no repeat penalty will be applied.
    pub repeat_scan_length: usize,
    /// The memory priority for this InferIter, which controls how much memory is devoted to a single inference and how stable the output is after long contexts.
    pub memory_priority: MemoryPriority,
}

impl InferParams {
    /// Returns a new InferParams with basic creative parameters set.
    /// This is good for general conversation and creative tasks.
    /// Also uses a higher memory priority by default and so may use more VRAM, but this can be changed with `with_memory_priority`.
    pub fn new_creative() -> Self {
        Self {
            temperature: 0.6,
            repeat_penalty: 1.05,
            repeat_scan_length: 64,
            memory_priority: MemoryPriority::High,
        }
    }

    /// Returns a new InferParams with basic parameters set for balanced output.
    /// This is good for general use and is a good starting point for most tasks.
    pub fn new_balanced() -> Self {
        Self {
            temperature: 0.4,
            repeat_penalty: 1.05,
            repeat_scan_length: 48,
            memory_priority: MemoryPriority::Low,
        }
    }

    /// Returns a new InferParams with basic parameters set for logical, near-deterministic output.
    /// This is good for tasks like parsing, extracting information, or code generation.
    pub fn new_logical() -> Self {
        Self {
            temperature: 0.2,
            repeat_penalty: 1.0,
            repeat_scan_length: 0,
            memory_priority: MemoryPriority::Low,
        }
    }

    /// Returns a new InferParams with basic parameters set for deterministic output.
    pub fn new_deterministic() -> Self {
        Self {
            temperature: 0.0,
            repeat_penalty: 1.0,
            repeat_scan_length: 0,
            memory_priority: MemoryPriority::Low,
        }
    }

    /// Returns the same InferParams but with the given memory priority.
    pub fn with_memory_priority(&self, memory_priority: MemoryPriority) -> Self {
        Self {
            temperature: self.temperature,
            repeat_penalty: self.repeat_penalty,
            repeat_scan_length: self.repeat_scan_length,
            memory_priority,
        }
    }
}

impl Default for InferParams {
    fn default() -> Self {
        Self::new_balanced()
    }
}

/// A single result from InferIter::complete.
/// Provides the completed text as well as the end sequence that was reached, if any.
pub struct InferCompletion<'a> {
    /// The completed text from the inference process.
    text: String,
    /// The end sequence that was reached, if any. This is useful for determining why the inference process stopped.
    end_sequence: Option<&'a str>,
}

impl InferCompletion<'_> {
    /// Unwrap the completion, returning the completed text.
    pub fn unwrap(self) -> String {
        self.text
    }

    /// Get the end sequence that was reached, if any.
    pub fn end_sequence(&self) -> Option<&str> {
        self.end_sequence
    }

    /// Get the complete result as it was generated. Use `unwrap` if you want the owned String.
    pub fn result(&self) -> &str {
        &self.text
    }

    /// Get the complete result trimmed of leading and trailing whitespace.
    pub fn trim(&self) -> &str {
        self.text.trim()
    }
}

/// An iterator that can be used to infer tokens from a model.
pub struct InferIter {
    model: Model,
    device: Device,
    tokens: TokenString,
    pending_tokens: Option<TokenString>,
    logits_processor: LogitsProcessor,
    seed: u64,
    last_set_temperature: f64,
    temperature: f64,
    steps_since_last_temperature_reduction: usize,
    repeat_penalty: f32,
    repeat_scan_length: usize,
    eos_token: u32,
    reached_eos: bool,
    step: usize,
    target_context_window: usize,
}

impl InferIter {
    const TOP_P: f64 = 0.85;
    const TEMPERATURE_REDUCTION_INTERVAL: usize = 64;
    const TEMPERATURE_REDUCTION_FACTOR: f64 = 0.97;

    pub(crate) fn new(
        mut model: Model,
        device: Device,
        tokens: TokenString,
        params: &InferParams,
    ) -> Self {
        // Create logits processor
        let seed = model.next_seed();
        let logits_processor =
            LogitsProcessor::new(seed, Some(params.temperature), Some(Self::TOP_P));

        let eos_token = model.eos_token();
        Self {
            model,
            device,
            tokens,
            logits_processor,
            seed,
            last_set_temperature: params.temperature,
            temperature: params.temperature,
            steps_since_last_temperature_reduction: 0,
            repeat_penalty: params.repeat_penalty,
            repeat_scan_length: params.repeat_scan_length,
            eos_token,
            reached_eos: false,
            pending_tokens: None,
            target_context_window: match params.memory_priority {
                MemoryPriority::High => TARGET_CONTEXT_WINDOW_HIGH,
                MemoryPriority::Low => TARGET_CONTEXT_WINDOW_LOW,
            },
            step: 0,
        }
    }

    /// Push some text into the context.
    pub fn push_str(&mut self, text: impl AsRef<str>) {
        if let Some(pending_tokens) = &mut self.pending_tokens {
            pending_tokens.push_str(text.as_ref());
        } else {
            self.pending_tokens = Some(self.model.tokenize_str(text.as_ref()));
        }
    }

    /// Infer the next token. Returns None if we have reached the end of the response (EOS token).
    pub fn next_token(&mut self) -> Option<u32> {
        // Exit early if we already got the end of text token
        if self.reached_eos {
            return None;
        }

        // Insert the pending context onto self.tokens if it is not empty
        // Also get the size of the inserted text in tokens to calculate the context correctly
        let context_add = if let Some(pending_tokens) = &mut self.pending_tokens {
            let old_len = self.tokens.len();
            self.tokens.push(&*pending_tokens);
            pending_tokens.clear();
            self.tokens.len() - old_len
        } else {
            0
        };

        // Get the context size for this step
        let context_size = (if self.step > 0 { 1 } else { self.tokens.len() }) + context_add;

        // Get the start position for the context
        let start_pos = self.tokens.len().saturating_sub(context_size);

        // Get the context
        let context = self.tokens.get(start_pos..).unwrap();

        // Create the input tensor containing the context
        let input = Tensor::new(context, &self.device)
            .unwrap()
            .unsqueeze(0)
            .unwrap();

        // Forward the input through the pipeline
        let logits = self.model.forward(&input, start_pos, context.len());

        // Preprocess the logits for this model type
        let logits = self.model.model_type().process_logits(logits);

        // Apply the repeat penalty
        let logits = if self.repeat_penalty == 1.0 || self.repeat_scan_length == 0 {
            logits
        } else {
            // Apply the repeat penalty to the last repeat_scan_length tokens
            let start_at = self.tokens.len().saturating_sub(self.repeat_scan_length);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                self.tokens.get(start_at..).unwrap(),
            )
            .unwrap()
        };

        // Sample the next token
        let next_token = self.logits_processor.sample(&logits).unwrap();

        // If the token is not the end of text token, add it to the tokens
        if next_token != self.eos_token {
            self.tokens.push_token(next_token);
        }
        // Otherwise, set reached_eos to true and return None
        else {
            self.reached_eos = true;
            return None;
        }

        // Increment the step
        self.step += 1;
        self.steps_since_last_temperature_reduction += 1;

        // Reduce the temperature every TEMPERATURE_REDUCTION_INTERVAL steps to help stabilize long contexts
        if self.steps_since_last_temperature_reduction >= Self::TEMPERATURE_REDUCTION_INTERVAL {
            self.temperature *= Self::TEMPERATURE_REDUCTION_FACTOR;
            self.logits_processor =
                LogitsProcessor::new(self.seed, Some(self.temperature), Some(Self::TOP_P));
            self.steps_since_last_temperature_reduction = 0;
        }

        // Return the next token
        Some(next_token)
    }

    /// Run the iterator until completion or until one of `end_sequences` is generated
    /// and return everything up to that point as an `InferCompletion`, as well as the end sequence that was reached
    pub fn complete<'a>(&mut self, end_sequences: &'a [&str]) -> InferCompletion<'a> {
        let time = Instant::now();

        let mut tokens_generated = 0;
        let mut response = String::new();
        while let Some(token) = self.next_token() {
            tokens_generated += 1;

            // Detokenize the token and add it to the response and the search window
            let token_str = self.tokens.model.borrow().detokenize([token]);
            response.push_str(&token_str);

            // Exit early at the first stop sequence from end_sequences encountered in response, truncating.
            let found_stop_sequence_position = end_sequences
                .iter()
                .enumerate()
                .filter_map(|(idx, &seq)| response.find(seq).map(|pos| (idx, pos)))
                .min_by_key(|&(_, pos)| pos);
            if let Some((idx, pos)) = found_stop_sequence_position {
                response.truncate(pos);

                let elapsed = time.elapsed().as_secs_f64();
                self.tokens
                    .model
                    .borrow()
                    .submit_timing(tokens_generated, elapsed);

                return InferCompletion {
                    text: response,
                    end_sequence: Some(end_sequences[idx]),
                };
            }
        }

        let elapsed = time.elapsed().as_secs_f64();
        self.tokens
            .model
            .borrow()
            .submit_timing(tokens_generated, elapsed);

        InferCompletion {
            text: response,
            end_sequence: None,
        }
    }

    /// Force inference of a single value or idea following the current context.
    /// This is useful for parsing a single value from the model, such as a number or a name, without
    /// consuming the entire response.
    /// Currently, this is implemented using "**" on both sides of the value, which may cause the model
    /// to pay special attention to the value.
    pub fn next_value(&mut self) -> String {
        let time = Instant::now();

        // Insert the "**" before the first token to force the model to generate a useful value.
        // Run the iterator until we get "**" back, returning everything in between as a string.
        self.push_str("**");
        let mut response = String::new();
        let mut tokens_generated = 0;
        while let Some(token) = self.next_token() {
            tokens_generated += 1;

            let token_str = self.tokens.model.borrow().detokenize([token]);

            response.push_str(&token_str);

            if let Some(pos) = response.find("**") {
                response.truncate(pos);
                break;
            }
        }

        let elapsed = time.elapsed().as_secs_f64();
        self.tokens
            .model
            .borrow()
            .submit_timing(tokens_generated, elapsed);

        response
    }

    /// Run the iterator until the current bracket is closed and return everything up to that point as a `String`.
    /// The end sequence will always be None for this method, since it is determined by the brackets rather than a specific string.
    pub fn complete_bracket<'a>(
        &mut self,
        open_bracket: char,
        close_bracket: char,
    ) -> InferCompletion<'a> {
        let time = Instant::now();

        // First start the response with the open bracket
        let mut response = open_bracket.to_string();
        self.push_str(&response);

        // Then run until we close the bracket, keeping track of nested brackets and ignoring brackets in strings
        let mut bracket_count = 1;
        let mut in_string = false;
        let mut escaped_last = false;
        let mut tokens_generated = 0;
        while let Some(token) = self.next_token() {
            tokens_generated += 1;

            let token_str = self.tokens.model.borrow().detokenize([token]);
            for c in token_str.chars() {
                if c == '\\' && !escaped_last {
                    escaped_last = true;
                } else {
                    if c == '"' && !escaped_last {
                        in_string = !in_string;
                    } else if !in_string {
                        if c == open_bracket {
                            bracket_count += 1;
                        } else if c == close_bracket {
                            bracket_count -= 1;
                        }
                    }
                    escaped_last = false;
                }
                response.push(c);
                if bracket_count < 1 {
                    let elapsed = time.elapsed().as_secs_f64();
                    self.tokens
                        .model
                        .borrow()
                        .submit_timing(tokens_generated, elapsed);

                    return InferCompletion {
                        text: response,
                        end_sequence: None,
                    };
                }
            }
        }

        let elapsed = time.elapsed().as_secs_f64();
        self.tokens
            .model
            .borrow()
            .submit_timing(tokens_generated, elapsed);

        InferCompletion {
            text: response,
            end_sequence: None,
        }
    }

    /// Completely reset the context, starting the iterator over again with the given tokens as the new context.
    pub fn reset(&mut self, new_context: impl IntoTokenString) {
        let new_tokens = self.tokens.model.borrow().tokenize(new_context);
        if let Some(pending_tokens) = &mut self.pending_tokens {
            pending_tokens.clear();
        }
        self.tokens = new_tokens;
        self.reached_eos = false;
        self.step = 0;
        self.temperature = self.last_set_temperature;
        self.steps_since_last_temperature_reduction = 0;
        self.seed = self.seed.wrapping_add(1);
        self.logits_processor =
            LogitsProcessor::new(self.seed, Some(self.temperature), Some(Self::TOP_P));
        self.model.clear_cache();
    }

    /// Get the context which was last used for inference. This does not include any text that has been pushed into the context
    /// via `push_str` since the last inference.
    pub fn last_context(&self) -> &TokenString {
        &self.tokens
    }

    /// Get the full context including any text that has been pushed into the context since the last inference.
    pub fn full_context(&self) -> TokenString {
        let mut context = self.tokens.clone();
        if let Some(pending_tokens) = self.pending_context() {
            context.push(pending_tokens);
        }
        context
    }

    /// Get the text that has been pushed into the context via `push_str` since the last inference,
    /// and which has not yet been included in any inference context.
    /// This may be None if
    pub(crate) fn pending_context(&self) -> Option<&TokenString> {
        self.pending_tokens.as_ref()
    }

    /// Get the target size of the context (determined by the `Model` at the creation of this InferIter).
    /// This is used to determine when to start truncating the context.
    pub fn target_context_window(&self) -> usize {
        self.target_context_window
    }

    /// Get a clone of the model used by this InferIter.
    /// For safety, this requires clearing the model's cache, so it may have a slight extra overhead.
    pub fn clone_model(&self) -> Model {
        let mut model = self.model.clone();
        model.clear_cache();
        model
    }

    /// Tokenize some text using the model's tokenizer and return it as a TokenString with the correct model reference.
    pub fn tokenize(&self, text: impl IntoTokenString) -> TokenString {
        self.model.tokenize(text)
    }

    /// Updates the inference parameters for this InferIter.
    pub fn update_params(&mut self, params: &InferParams) {
        self.repeat_penalty = params.repeat_penalty;
        self.repeat_scan_length = params.repeat_scan_length;
        self.last_set_temperature = params.temperature;
        self.temperature = params.temperature;
        self.steps_since_last_temperature_reduction = 0;
        self.logits_processor = LogitsProcessor::new(
            self.model.next_seed(),
            Some(self.temperature),
            Some(Self::TOP_P),
        );
    }
}

impl From<InferIter> for String {
    fn from(mut infer_iter: InferIter) -> Self {
        infer_iter.complete(&[]).unwrap()
    }
}

impl Iterator for InferIter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}
