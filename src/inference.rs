use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;

use crate::{
    model::ModelPipeline,
    prelude::{IntoTokenString, ModelType, TokenString},
};

pub struct InferIter {
    model_type: ModelType,
    device: Device,
    tokens: TokenString,
    vocab_size: usize,
    pipeline: ModelPipeline,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
    eos_token: u32,
    reached_eos: bool,
    insert_before: String,
    step: usize,
}

impl InferIter {
    pub(crate) fn new(
        model_type: ModelType,
        device: Device,
        tokens: TokenString,
        vocab_size: usize,
        pipeline: ModelPipeline,
        logits_processor: LogitsProcessor,
        repeat_penalty: f32,
        repeat_last_n: usize,
        eos_token: u32,
    ) -> Self {
        Self {
            model_type,
            device,
            tokens,
            vocab_size,
            pipeline,
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            eos_token,
            reached_eos: false,
            insert_before: String::new(),
            step: 0,
        }
    }
    
    /// Push some text into the context.
    pub fn push_str(&mut self, text: impl AsRef<str>) {
        self.insert_before.push_str(text.as_ref());
    }

    /// Infer the next token, optionally inserting some text first to influence the inference.
    /// Returns None if the end of text token is reached.
    pub fn next_token(&mut self) -> Option<u32> {
        // Exit early if we already got the end of text token
        if self.reached_eos {
            return None;
        }

        // Insert the self.insert_before onto self.tokens if insert_before is not empty
        // Also get the size of the inserted text in tokens to calculate the context correctly
        let context_add = if !self.insert_before.is_empty() {
            let old_len = self.tokens.len();
            self.tokens.push_str(&self.insert_before);
            self.insert_before.clear();
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
        let logits = self.pipeline.forward(&input, start_pos, context.len());

        // Preprocess the logits for this model type
        let logits = self.model_type.process_logits(logits);

        // Apply the repeat penalty
        let logits = if self.repeat_penalty == 1.0 || self.repeat_last_n == 0 {
            logits
        } else {
            // Apply the repeat penalty to the last repeat_last_n tokens
            let start_at = self.tokens.len().saturating_sub(self.repeat_last_n);
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
    ) -> (String, Option<&'a str>) {
        let mut response = String::new();
        while let Some(token) = self.next_token()
            && token < self.vocab_size as u32 - 1
        {
            let token_str = self.tokens.model.detokenize([token]);

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
                return (response, Some(end_sequences[idx]));
            }
        }

        (response, None)
    }

    /// Force inference of a single value or idea following the current context and an optional prefix.
    /// This is useful for parsing a single value from the model, such as a number or a name, without
    /// consuming the entire response.
    /// Inserts the prefix into the context.
    pub fn next_value(&mut self) -> String {
        // Insert the prefix and "**" before the first token to force the model to generate a useful value.
        // Run the iterator until we get "**" back, returning everything in between as a string.
        self.push_str("**");
        let mut response = String::new();
        while let Some(token) = self.next_token()
            && token < self.vocab_size as u32 - 1
        {
            let token_str = self.tokens.model.detokenize([token]);

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
            let token_str = self.tokens.model.detokenize([token]);
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
        let new_tokens = self.tokens.model.tokenize(new_context);
        self.tokens = new_tokens;
        self.reached_eos = false;
        self.insert_before.clear();
        self.step = 0;
        self.pipeline.reset_cache();
    }

    /// Get the context which was last used for inference. This does not include any text that has been pushed into the context
    /// via `push_str` since the last inference.
    pub fn last_context(&self) -> &TokenString {
        &self.tokens
    }

    /// Get the full context including any text that has been pushed into the context via `push_str` since the last inference.
    pub fn full_context(&self) -> String {
        let mut context = self.tokens.to_string();
        if !self.insert_before.is_empty() {
            context.push_str(&self.insert_before);
        }
        context
    }
}

impl From<InferIter> for String {
    fn from(mut infer_iter: InferIter) -> Self {
        infer_iter.complete(&[]).0
    }
}

impl Iterator for InferIter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_token()
    }
}
