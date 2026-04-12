use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::{
    model::{Model, ModelPipeline},
    prelude::{IntoTokenString, ModelType, TokenString},
};

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
}

impl InferParams {
    /// Returns a new InferParams with basic creative parameters set.
    /// This is good for general conversation and creative tasks.
    pub fn new_creative() -> Self {
        Self {
            temperature: 0.7,
            repeat_penalty: 1.1,
            repeat_scan_length: 72,
        }
    }
    
    /// Returns a new InferParams with basic parameters set for balanced output.
    /// This is good for general use and is a good starting point for most tasks.
    pub fn new_balanced() -> Self {
        Self {
            temperature: 0.55,
            repeat_penalty: 1.05,
            repeat_scan_length: 36,
        }
    }
    
    /// Returns a new InferParams with basic parameters set for logical, near-deterministic output.
    /// This is good for tasks like parsing, extracting information, or code generation.
    pub fn new_logical() -> Self {
        Self {
            temperature: 0.25,
            repeat_penalty: 1.0,
            repeat_scan_length: 0,
        }
    }

    /// Returns a new InferParams with basic parameters set for deterministic output.
    pub fn new_deterministic() -> Self {
        Self {
            temperature: 0.0,
            repeat_penalty: 1.0,
            repeat_scan_length: 0,
        }
    }
}

impl Default for InferParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            repeat_penalty: 1.05,
            repeat_scan_length: 64,
        }
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

    /// Get the complete result as it was generated.
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
    vocab_size: usize,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_scan_length: usize,
    eos_token: u32,
    reached_eos: bool,
    pending_context: String,
    step: usize,
}

impl InferIter {
    pub(crate) fn new(
        model: Model,
        device: Device,
        tokens: TokenString,
        vocab_size: usize,
        logits_processor: LogitsProcessor,
        params: &InferParams,
    ) -> Self {
        let eos_token = model.eos_token();
        Self {
            model,
            device,
            tokens,
            vocab_size,
            logits_processor,
            repeat_penalty: params.repeat_penalty,
            repeat_scan_length: params.repeat_scan_length,
            eos_token,
            reached_eos: false,
            pending_context: String::new(),
            step: 0,
        }
    }
    
    /// Push some text into the context.
    pub fn push_str(&mut self, text: impl AsRef<str>) {
        self.pending_context.push_str(text.as_ref());
    }

    /// Infer the next token. Returns None if we have reached the end of the response (EOS token).
    pub fn next_token(&mut self) -> Option<u32> {
        // Exit early if we already got the end of text token
        if self.reached_eos {
            return None;
        }

        // Insert the pending context onto self.tokens if it is not empty
        // Also get the size of the inserted text in tokens to calculate the context correctly
        let context_add = if !self.pending_context.is_empty() {
            let old_len = self.tokens.len();
            self.tokens.push_str(&self.pending_context);
            self.pending_context.clear();
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

        // Increment the step
        self.step += 1;

        // If the token is not the end of text token, add it to the tokens
        if next_token != self.eos_token {
            self.tokens.push_token(next_token);
        }
        // Otherwise, set reached_eos to true and return None
        else {
            self.reached_eos = true;
            return None;
        }

        // Return the next token
        Some(next_token)
    }

    /// Run the iterator until completion or until one of `end_sequences` is generated
    /// and return everything up to that point as a `String`, as well as the end sequence that was reached
    pub fn complete<'a>(
        &mut self,
        end_sequences: &'a [&str],
    ) -> InferCompletion<'a> {
        let mut response = String::new();
        while let Some(token) = self.next_token()
            && token < self.vocab_size as u32 - 1
        {
            let token_str = self.tokens.model.borrow().detokenize([token]);

            response.push_str(&token_str);

            // Exit early at the first stop sequence from end_sequences encountered in response, truncating.
            // Only search in the last END_SEQUENCE_SEARCH_WINDOW characters of the response
            let found_stop_sequence_position = end_sequences
                .iter()
                .enumerate()
                .filter_map(|(idx, &seq)| response.find(seq).map(|pos| (idx, pos)))
                .min_by_key(|&(_, pos)| pos);

            if let Some((idx, pos)) = found_stop_sequence_position {
                response.truncate(pos);
                return InferCompletion {
                    text: response,
                    end_sequence: Some(end_sequences[idx]),
                };
            }
        }

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
        // Insert the "**" before the first token to force the model to generate a useful value.
        // Run the iterator until we get "**" back, returning everything in between as a string.
        self.push_str("**");
        let mut response = String::new();
        while let Some(token) = self.next_token()
            && token < self.vocab_size as u32 - 1
        {
            let token_str = self.tokens.model.borrow().detokenize([token]);

            response.push_str(&token_str);

            if let Some(pos) = response.find("**") {
                response.truncate(pos);
                break;
            }
        }
        response
    }

    /// Run the iterator until the current bracket is closed and return everything up to that point as a `String`.
    pub fn complete_bracket(&mut self, open_bracket: char, close_bracket: char) -> String {
        let mut response = String::new();
        let mut bracket_count = 0;
        let mut in_string = false;
        let mut escaped_last = false;
        while let Some(token) = self.next_token() {
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
                            if bracket_count == 0 {
                                return response;
                            }
                            bracket_count -= 1;
                        }
                    }
                    escaped_last = false;
                }
                response.push(c);
            }
        }
        response
    }

    /// Completely reset the context, starting the iterator over again with the given tokens as the new context.
    pub fn reset(&mut self, new_context: impl IntoTokenString) {
        let new_tokens = self.tokens.model.borrow().tokenize(new_context);
        self.tokens = new_tokens;
        self.reached_eos = false;
        self.pending_context.clear();
        self.step = 0;
        self.model.clear_cache();
    }

    /// Get the context which was last used for inference. This does not include any text that has been pushed into the context
    /// via `push_str` since the last inference.
    pub fn last_context(&self) -> &TokenString {
        &self.tokens
    }

    /// Get the full context including any text that has been pushed into the context via `push_str` since the last inference.
    pub fn full_context(&self) -> TokenString {
        let mut context = self.tokens.clone();
        if !self.pending_context.is_empty() {
            context.push_str(&self.pending_context);
        }
        context
    }

    /// Get the text that has been pushed into the context via `push_str` since the last inference,
    /// and which has not yet been included in any inference context.
    pub(crate) fn pending_context(&self) -> &str {
        &self.pending_context
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
