pub mod action;
pub mod actor;
pub mod chat;
pub mod data;
pub mod inference;
pub mod model;
pub mod model_type;
pub mod pipeline;
pub mod prelude;
pub mod scene;
pub mod token_string;

#[cfg(test)]
mod tests {
    use std::io::Write;

    use crate::prelude::*;

    #[test]
    fn pipeline() {
        const SEED: u64 = 46364;
        const TEMP: f64 = 0.65;
        const POEM_THEMES: &[&str] = &[
            "a magical adventure in a fantasy world",
            "the beauty of nature in spring",
            "breaking free from constraints and embracing freedom",
        ];

        // Create a new pipeline
        let mut pipeline = Pipeline::new(SEED);

        // Add a chat step to the pipeline which generates a poem based on the theme in {theme}
        // And store the generated poem in the context under {poem}
        pipeline.chat(
            "poem",
            "You are a creative and imaginative writer who writes rhyming poems based on themes. \
                Use descriptive and engaging language but not too many long or obscure words. \
                Make sure the poem rhymes and has a nice flow.",
            [],
            "Write a rhyming poem based on the following theme: {theme}",
            "Certainly, here is the poem:\n\n\"",
            ["\"".to_string()],
            TEMP,
        );

        // Add a step to the pipeline which summarizes the poem in {poem} and stores the summary in the context under {summary}
        pipeline.summarize(
            "summary",
            "poem",
            Some("The summary must be a poem too.".to_string()),
        );

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Execute the pipeline on the model for each poem theme and print the poem and summary outputs
        for theme in POEM_THEMES {
            let input = json_map! {
                "theme" => theme,
            };
            let output = model.execute_pipeline(&pipeline, input);
            let poem = output["poem"].as_str().unwrap();
            let summary = output["summary"].as_str().unwrap();
            println!(
                "Theme: {}\n\nGenerated poem:\n{}\n\n\n\nSummarized version:\n{}\n\n\n\n------\n\n\n",
                theme, poem, summary
            );
        }
    }

    #[test]
    fn chat() {
        const SEED: u64 = 477474;
        const TEMP: f64 = 0.6;
        const CONVERSATION_TURNS: usize = 14;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Start a chat
        let mut chat = Chat::new();
        chat.set_system_prompt(
            "You are a helpful assistant and friendly person who has a great imagination, \
        an open mind, and is fun to talk to. Talk to the user in a friendly and engaging manner.",
        );

        println!("System: {}", chat.system_prompt());

        // Infer a conversation
        for turn in 0..CONVERSATION_TURNS {
            // Get the user message from the console input
            let mut input = String::new();
            print!("User: ");
            std::io::stdout().flush().unwrap();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim().to_string();
            chat.add_message(ChatRole::User, input);

            // Infer a model response
            chat.infer_message(
                &ChatRole::Model,
                &model,
                false,
                SEED.wrapping_add(turn as u64),
                Some(TEMP),
                1.1,
                64,
            );
            let message = chat.last_message().unwrap().content();

            println!("\nAssistant: {}\n", message);
        }
    }

    #[test]
    fn action_extraction() {
        const SEED: u64 = 634667374;
        const ATTEMPTS: usize = 5;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Create an action extractor
        let mut extractor = ActionExtractor::new(model.clone());

        // Add some action patterns
        extractor
            .add_action_pattern(ActionPattern::new(
                "go_somewhere",
                vec![("destination_name".to_string(), ArgType::String)],
            ))
            .unwrap();
        extractor
            .add_action_pattern(ActionPattern::new(
                "attack_something",
                vec![
                    ("weapon_name".to_string(), ArgType::String),
                    ("target_name".to_string(), ArgType::String),
                ],
            ))
            .unwrap();
        extractor
            .add_action_pattern(ActionPattern::new(
                "say_something",
                vec![("what_to_say".to_string(), ArgType::String)],
            ))
            .unwrap();
        extractor
            .add_action_pattern(ActionPattern::new(
                "do_something_else",
                vec![("what_to_do".to_string(), ArgType::String)],
            ))
            .unwrap();

        // Extract some actions from text
        let text_strings = [
            "Go north",
            "Do a funny dance and wink",
            "Break the nearest crate with your sword",
            "Attack the goblin with your bow",
            "Tell a funny joke about dragons",
            "Kill the villagers",
            "Quickly run to the east",
        ];

        for text in text_strings {
            let action = extractor.extract_action(text, ATTEMPTS);
            println!(
                "Extracted action from '{}': {}",
                text,
                action
                    .as_ref()
                    .map(Action::to_string)
                    .unwrap_or("None".to_string())
            );
        }
    }

    #[test]
    fn predict_chain() {
        const SEED: u64 = 13579;
        const TEMP: f64 = 0.5;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        let mut prediction = model.predict_next(
            "Here is my character bio:\nName: Jessie\nAge: 19\nClass: Archer\n",
            SEED,
            Some(TEMP),
            None,
            1.1,
            64,
        );
        let weapon: String = prediction.next_value(Some(" Weapon: "));
        let clothing = prediction.next_value(Some(" Clothing: "));
        let hometown = prediction.next_value(Some(" Hometown: "));
        println!(
            "Predicted character bio:\nName: Jessie\nAge: 19\nClass: Archer\nWeapon: {}\nClothing: {}\nHometown: {}",
            weapon, clothing, hometown
        );
    }

    #[test]
    fn generate_story() {
        const SEED: u64 = 547845;
        const TEMP: f64 = 0.6;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Generate a story
        let mut story = "There was once".to_string();
        story.push_str(
            &model
                .predict_next(
                    "long_story = \"There was once",
                    SEED,
                    Some(TEMP),
                    None,
                    1.1,
                    64,
                )
                .complete(&["\""])
                .0,
        );
        println!("Generated story:\n{}", story);
    }

    #[test]
    fn thinking_simple_math() {
        const SEED: u64 = 3463;
        const TEMP: f64 = 0.6;

        // Create the model and chat
        let model = Model::new(ModelType::Qwen3(ModelSize::Medium), SEED, true).unwrap();

        // Give the model a simple problem to think about
        let (result, thoughts) = model.instruct(
            "If a train leaves Station A at 60 mph and another leaves Station B 100 \
                miles away at 40 mph towards each other, when do they meet?",
            true,
            SEED,
            Some(TEMP),
            None,
            1.1,
            64,
        );
        let result = result.complete(&[]).0.trim().to_string();

        println!(
            "\nThoughts:\n{}\nRest:\n{}\n",
            thoughts.unwrap_or_default(),
            result
        );
    }

    #[test]
    fn concat_strings() {
        const SEED: u64 = 3463;
        const CONCAT_STRINGS: &[&[&str]] = &[
            &["Johnathan", "take a long walk in the park"],
            &["The quick brown fox", "jumping over the lazy dog"],
            &["Rust", "programming", "is super fun", "and rewarding"],
            &[
                "This",
                "is",
                "a",
                "test",
                "of",
                "concatenating",
                "many",
                "strings",
            ],
            &["Please inform", "Jason", "the meeting", "at 2:00 PM"],
        ];

        // Create the model and chat
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        for strings in CONCAT_STRINGS {
            let concatenated = model.join(strings, SEED, Some(0.6));
            println!(
                "Joining the strings {:?} gives:\n{}\n",
                strings, concatenated
            );
        }
    }

    #[test]
    fn scene() {
        const SEED: u64 = 12345;
        const TEMP: f64 = 0.6;
        const STORY_CYCLES: usize = 5;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Create a new scene
        let mut scene = Scene::new(
            "Forest Encounter",
            "The scene opens in a dense forest. Alice and Bob are exploring the area, \
            looking for signs of a supposed nearby ruin.",
            model,
        );

        // Add actors to the scene
        scene.add_actor(Actor::new(
            "Alice",
            "A curious and adventurous young woman, with a knack for archery.",
        ));
        scene.add_actor(Actor::new(
            "Bob",
            "A cautious and thoughtful young man, always looking out for his friends.",
        ));

        // Add a turn to the scene
        scene
            .add_turn(SceneTurn::dialogue("Alice", "What is that over there?"))
            .unwrap();
        scene
            .add_turn(SceneTurn::dialogue(
                "Bob",
                "I think it's a mysterious creature.",
            ))
            .unwrap();

        for _ in 0..STORY_CYCLES {
            scene
                .infer_next_turn(InferredSceneTurn::story(), SEED, Some(TEMP), 1.1, 64)
                .unwrap();

            scene
                .infer_next_turn(
                    InferredSceneTurn::action("Alice".to_string()),
                    SEED,
                    Some(TEMP),
                    1.1,
                    64,
                )
                .unwrap();

            scene
                .infer_next_turn(
                    InferredSceneTurn::dialogue("Bob".to_string()),
                    SEED,
                    Some(TEMP),
                    1.1,
                    64,
                )
                .unwrap();

            println!("\n{}\n", scene);
        }

        // Print the scene
        println!("Scene:\n{}", scene);
    }
}
