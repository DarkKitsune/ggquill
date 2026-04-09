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
pub mod joiner;

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
        const TEMP: f64 = 0.65;
        const CONVERSATION_TURNS: usize = 14;

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Start a chat
        let mut chat = Chat::new(
            &model,
            "You are a helpful assistant and friendly person who has a great imagination, \
            an open mind, and is fun to talk to. Talk to the user in a friendly and engaging manner.",
            &[],
            SEED,
            Some(TEMP),
        );

        // Infer a conversation
        for _ in 0..CONVERSATION_TURNS {
            // Get the user message from the console input
            let mut input = String::new();
            print!("User: ");
            std::io::stdout().flush().unwrap();
            std::io::stdin().read_line(&mut input).unwrap();
            let input = input.trim().to_string();

            // Push the user message to the chat history
            chat.push_message(ChatMessage::new(ChatRole::User, input));

            // Infer a model response
            let message = chat.infer_message(&ChatRole::Model, &[], None);

            println!("\nAssistant: {}\n", message);
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
    fn joiner() {
        const SEED: u64 = 24680;
        const TEMP: f64 = 0.45;
        const ITEMS_TO_JOIN: &[&[&str]] = &[
            &["the cat", "sat on", "the mat"],
            &["jack saw", "red roses", "beautiful sunset"],
            &["the quick", "brown fox", "jumps over", "the lazy dog."],
            &["The ingredients for the recipe are", "flour", "sugar", "eggs", "milk", "butter", "."],
        ];

        // Create the model
        let model = Model::new(ModelType::Qwen3Special, SEED, true).unwrap();

        // Create a joiner
        let mut joiner = Joiner::new(&model, SEED, Some(TEMP));

        for items in ITEMS_TO_JOIN {
            let result = joiner.join(items);
            println!("Joining {:?} -> {}", items, result);
        }
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
                .complete(&["\""], None)
                .0,
        );
        println!("Generated story:\n{}", story);
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
