use std::fmt::Display;

use anyhow::Result;

use crate::{actor::Actor, model::Model};

/// Represents a scene as part of a story or event, made up of actors and turns.
pub struct Scene {
    name: String,
    model: Model,
    actors: Vec<Actor>,
    turns: Vec<SceneTurn>,
}

impl Scene {
    /// Creates a new, empty scene.
    pub fn new(name: impl Display, intro: impl Display, model: Model) -> Self {
        Self {
            name: name.to_string(),
            model,
            actors: Vec::new(),
            turns: vec![SceneTurn::Story(intro.to_string())],
        }
    }

    pub fn add_actor(&mut self, actor: Actor) {
        self.actors.push(actor);
    }

    pub fn actors(&self) -> &[Actor] {
        &self.actors
    }

    pub fn actor_with_name(&self, name: &str) -> Option<&Actor> {
        self.actors.iter().find(|actor| actor.name() == name)
    }

    /// Adds a new turn to the scene.
    ///
    /// # Errors
    ///
    /// Returns an error if the actor for this turn does not exist in the scene.
    pub fn add_turn(&mut self, turn: SceneTurn) -> Result<()> {
        // Bail if the actor for this turn does not exist in the scene
        if let Some(actor_name) = turn.actor_name()
            && self.actor_with_name(actor_name).is_none()
        {
            anyhow::bail!("Actor '{}' does not exist in the scene", actor_name);
        }

        /*
        // Compress the scene turns if needed
        if let Some(compress_turns_string) = self.ready_for_compression() {
            // Summarize the turns using the model
            let summary = self.model.summarize(&compress_turns_string, 0, None);

            // Insert the summarized turn back into the scene at position 0
            self.turns.insert(0, SceneTurn::Story(summary));
        }*/

        // Add the new turn to the scene
        self.turns.push(turn);

        Ok(())
    }

    /// Infers the next turn into the scene using the model.
    ///
    /// # Errors
    ///
    /// Returns an error if the actor for this turn does not exist in the scene.
    pub fn infer_next_turn(
        &mut self,
        turn: InferredSceneTurn,
        seed: u64,
        temp: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
    ) -> Result<&SceneTurn> {
        // Bail if the actor for this turn does not exist in the scene
        if let Some(actor_name) = turn.actor_name()
            && self.actor_with_name(actor_name).is_none()
        {
            anyhow::bail!("Actor '{}' does not exist in the scene", actor_name);
        }

        // Start the prompt with self::to_string
        let mut prompt = self.to_string();
        prompt.push_str("\n\n");

        // Append the begin sequence for this inferred turn
        prompt.push_str(&turn.begin_sequence());

        // Infer until one of the end sequences for this inferred turn is found
        let inferred = self
            .model
            .predict_next(prompt, seed, temp, None, repeat_penalty, repeat_last_n)
            .complete(turn.end_sequences(), None)
            .0
            .trim()
            .to_string();

        // Complete the inferred turn with the inferred text
        let completed_turn = turn.complete(inferred);
        self.turns.push(completed_turn);

        Ok(self.turns.last().unwrap())
    }
    /*
    fn ready_for_compression(&mut self) -> Option<String> {
        if self.turns.len() > Self::COMPRESS_THRESHOLD {
            // Gather the turns to compress
            let to_compress = self
                .turns
                .drain(0..self.turns.len().saturating_sub(Self::COMPRESS_RETAIN_TURNS));

            // Format them as a string
            Some(
                to_compress
                    .map(|turn| turn.to_string())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
            )
        } else {
            None
        }
    }*/

    pub fn turns(&self) -> &[SceneTurn] {
        &self.turns
    }
}

impl Display for Scene {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Describe the cast of actors and their identities
        let cast_string = self
            .actors
            .iter()
            .map(|actor| format!("{}:\n{}", actor.name(), actor.identity()))
            .collect::<Vec<_>>()
            .join("\n\n");
        let turns_string = self
            .turns
            .iter()
            .map(|turn| turn.to_string())
            .collect::<Vec<_>>()
            .join("\n\n");
        write!(
            f,
            "**Cast:**\n\n{}\n\n**ACT ONE**\n\n{}\n\n{}",
            cast_string, self.name, turns_string
        )
    }
}

/// Represents a single turn in a scene.
pub enum SceneTurn {
    Story(String),
    Dialogue(String, String),
    Action(String, String),
}

impl SceneTurn {
    /// Create a new `SceneTurn` representing a story segment.
    pub fn story(text: impl Display) -> Self {
        SceneTurn::Story(text.to_string())
    }

    /// Create a new `SceneTurn` representing a dialogue segment.
    pub fn dialogue(actor: impl Display, text: impl Display) -> Self {
        SceneTurn::Dialogue(actor.to_string(), text.to_string())
    }

    /// Create a new `SceneTurn` representing an action segment.
    pub fn action(actor: impl Display, action: impl Display) -> Self {
        SceneTurn::Action(actor.to_string(), action.to_string())
    }

    /// Get the actor name associated with this turn, if it has one.
    pub fn actor_name(&self) -> Option<&str> {
        match self {
            SceneTurn::Story(_) => None,
            SceneTurn::Dialogue(actor, _) => Some(actor),
            SceneTurn::Action(actor, _) => Some(actor),
        }
    }

    /// Get the main content of this turn as a string slice.
    pub fn content(&self) -> &str {
        match self {
            SceneTurn::Story(text) => text,
            SceneTurn::Dialogue(_, text) => text,
            SceneTurn::Action(_, action) => action,
        }
    }
}

impl Display for SceneTurn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SceneTurn::Story(text) => write!(f, "<{}>", text),
            SceneTurn::Dialogue(actor, text) => write!(f, "**{}:** \"{}\"", actor, text),
            SceneTurn::Action(_, action) => write!(f, "*{}*", action),
        }
    }
}

/// Represents a potential turn which has yet to be inferred.
#[derive(Debug, Clone)]
pub enum InferredSceneTurn {
    Story,
    Dialogue(String),
    Action(String),
}

impl InferredSceneTurn {
    /// Create a new inferred turn representing a story segment.
    pub fn story() -> Self {
        InferredSceneTurn::Story
    }

    /// Create a new inferred turn representing a dialogue segment.
    pub fn dialogue(actor: impl Display) -> Self {
        InferredSceneTurn::Dialogue(actor.to_string())
    }

    /// Create a new inferred turn representing an action segment.
    pub fn action(actor: impl Display) -> Self {
        InferredSceneTurn::Action(actor.to_string())
    }

    /// Complete this inferred turn with the given text, producing a `SceneTurn`.
    pub fn complete(self, text: String) -> SceneTurn {
        match self {
            InferredSceneTurn::Story => SceneTurn::Story(text),
            InferredSceneTurn::Dialogue(actor) => SceneTurn::Dialogue(actor, text),
            InferredSceneTurn::Action(actor) => {
                let text = format!("{} {}", actor, text);
                SceneTurn::Action(actor, text)
            }
        }
    }

    /// Get the actor name associated with this turn, if it has one.
    pub fn actor_name(&self) -> Option<&str> {
        match self {
            InferredSceneTurn::Story => None,
            InferredSceneTurn::Dialogue(actor) => Some(actor),
            InferredSceneTurn::Action(actor) => Some(actor),
        }
    }

    /// Get the begin sequence for this inferred turn.
    pub fn begin_sequence(&self) -> String {
        match self {
            InferredSceneTurn::Story => "<".to_string(),
            InferredSceneTurn::Dialogue(actor) => format!("**{}:** \"", actor),
            InferredSceneTurn::Action(actor) => format!("*Next, {}", actor),
        }
    }

    /// Get the end sequences for this inferred turn.
    pub fn end_sequences(&self) -> &[&'static str] {
        match self {
            InferredSceneTurn::Story => &[">", "\n\n"],
            InferredSceneTurn::Dialogue(_) => &["\"", "\n\n"],
            InferredSceneTurn::Action(_) => &["*", "\n\n"],
        }
    }
}
