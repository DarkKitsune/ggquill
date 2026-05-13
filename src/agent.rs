use std::fmt::{Debug, Display};

use crate::prelude::*;

/// A single response from the agent after executing a step, which can either be a tool call or a reasoning step.
#[derive(Clone)]
pub enum AgentResponse {
    ToolCall(ToolCall),
    Reasoning(String),
}

impl Debug for AgentResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AgentResponse::ToolCall(tool_call) => write!(f, "{}", tool_call.to_json_schema()),
            AgentResponse::Reasoning(reasoning) => write!(f, "Reasoning:\n{}", reasoning),
        }
    }
}

/// An agent takes a numbered list of steps to complete a given task, where each step either involves calling a tool or reasoning about the information available to the agent.
pub struct Agent {
    task: String,
    task_steps: Vec<String>,
    tools: Vec<Tool>,
    chat: Chat,
}

impl Agent {
    /// Creates a new agent with the given task and model.
    pub fn new(model: Model, task: impl Display, task_steps: impl IntoIterator<Item = impl Display>, tools: impl Into<Vec<Tool>>) -> Self {
        let tools = tools.into();
        let tools_schemas = tools.iter().map(Tool::to_json_schema).collect::<Vec<_>>();
        let task_steps: Vec<String> = task_steps.into_iter().map(|s| s.to_string()).collect();

        // Create the system prompt
        let system_prompt = format!(
            "You are an assistant whose job is to complete a task by following a list of steps provided by the user. \
            Each step either involves calling a tool with specific arguments or reasoning about the information provided \
            in order to complete the step and make progress towards completing your ultimate task.\n\
            # Your Task:\n\
            {}\n\
            # Recommended Steps to Complete the Task:\n\
            - {}\n\
            # Tool Call Format:\n\
            When calling a tool, use the following JSON format:\n\
            {{\n\
                \"tool\": \"tool_name\",\n\
                \"args\": {{\n\
                    // arguments for the tool call\n\
                }}\n\
             }}\n\
            # Available Tools:```json\n\
            {}\n\
            ```",
            task,
            task_steps.join("\n- "),
            // We include the list of available tools in the system prompt so that the model can reference it when determining which tools to call and with what arguments
            serde_json::to_string_pretty(&tools_schemas).unwrap(),
        );

        let chat = Chat::new(
            model,
            system_prompt,
            &[],
            &InferParams::new_logical()
                .with_memory_priority(MemoryPriority::High),
            [
                "Ensure that the steps are well-structured and clearly indicate the order of execution.".to_string(),
                "Be creative where appropriate, yet accurate.".to_string(),
                "The user will start their message with \"Next Step:\" followed by the step they want you to execute.".to_string(),
                "After a tool's response you should tell the user what the tool response is with a \"Tool Response\" header as welll as any other relevant information from that step.".to_string(),
"Each time the user prompts you with the next step, determine if it is a tool call step (if you need to use a tool to complete that step), or a reasoning step (you do not need a tool to complete that step), \
and then respond accordingly.\n\
If it is a reasoning step then try to complete that step using only the information available to you and your best judgment.\n\
If it is a tool call then your response should only be that tool call in the correct JSON format:```json
{
    \"tool\": \"tool_name\",
    \"args\": {
        \"arg_string\": \"value1\",
        \"some_number\": 0,
    }
}".to_string(),
                "Keep your responses short and concise, only including the necessary information to complete the task accurately. Reasoning should be a very short but accurate train of thought.".to_string(),
            ],
            None,
            [],
        ).0;

        Self { task: task.to_string(), task_steps, tools, chat }
    }

    /// Pops the next step from the beginning of the list of steps and executes it with the agent.
    /// Returns the step, the agent's response for that step, and the tool result (if any), or None if there are no more steps to execute.
    fn execute_next_step(&mut self) -> Option<(String, String, Option<JsonValue>)> {
        fn parse_tool_call(tool_call_json: &JsonValue) -> Option<(&str, JsonMap)> {
            let tool_name = tool_call_json.get("tool")?.as_str()?;
            let args = tool_call_json.get("args")?.as_object()?.clone();
            Some((tool_name, args))
        }

        // If there are no more steps to execute, return None, otherwise remove the 0th step and execute it
        let next_step = if self.task_steps.is_empty() {
            return None;
        }
        else {
            self.task_steps.remove(0)
        };

        // Using the chat, ask the model to determine if the next step is a tool call or an reasoning step, and execute it accordingly
        let user_message = format!("# Next Step\n{}\n\nIs this a tool call or an reasoning step?", next_step);
        self.chat.push_message(ChatMessage::user(&user_message));

        // Infer the assistant's response to determine if it is a tool call or an reasoning step
        // Keep trying until we get a valid response that we can parse, since the model may not always respond in the correct format on the first try
        let checkpoint = self.chat.create_checkpoint();
        let mut assistant_response = None;
        while assistant_response.is_none() {
            // Try getting a response
            assistant_response = self.chat.infer_message_ext(
                &ChatRole::Assistant,
                Some("This step is a "),
                &["\n", ".", "!"],
                |message, infer_iter, _end_sequence| {
                    // Add a newline just in case
                    infer_iter.push_str("\n");

                    // For now we determine if the message is a tool call or an reasoning step by checking if it starts with "tool" (case-insensitive)
                    let message = message.to_lowercase().trim().to_string();
                    if message.starts_with("tool") {
                        // This is a tool call, so we infer the tool call JSON from the message
                        infer_iter.push_str("# Tool Call\n```json\n");
                        let tool_call_str = infer_iter.complete(&["```"]).trim().to_string();

                        // Create a ToolCall
                        let tool_call_json: JsonValue = serde_json::from_str(&tool_call_str).ok()?;
                        let (tool_name, args) = parse_tool_call(&tool_call_json)?;
                        let tool = self.tools.iter().find(|t| t.name() == tool_name)?;
                        let tool_call = ToolCall::new(tool.clone(), args);
                        Some(AgentResponse::ToolCall(tool_call))
                    }
                    else {
                        // This is an reasoning step, so we infer the reasoning
                        infer_iter.push_str("# Reasoning/Response For This Step\n<reasoning>");
                        let reasoning = infer_iter.complete(&["</reasoning>"]).trim().to_string();

                        Some(AgentResponse::Reasoning(reasoning))
                    }
                }
            );

            // If we couldn't get a valid response then roll back the chat to the checkpoint and try again
            if assistant_response.is_none() {
                self.chat.reset_to_checkpoint(&checkpoint);
            }
        }

        let assistant_response = assistant_response.unwrap().1;

        // If this is a tool call then execute the tool call and get the assistant response for that
        let (assistant_response, tool_response) = match assistant_response {
            AgentResponse::ToolCall(tool_call) => {
                let (tool_result, tool_assistant_response) = tool_call.execute(&mut self.chat, &["Next Step", "Next step"]).unwrap();
                (tool_assistant_response.into_content(), Some(tool_result))
            },
            AgentResponse::Reasoning(reasoning) => (reasoning, None),
        };

        Some((next_step, assistant_response, tool_response))
    }

    // Complete the task and return the final summary
    pub fn execute(&mut self, with_logging: bool) -> String {
        while let Some((next_step, assistant_response, tool_response)) = self.execute_next_step() {
            if with_logging {
                println!("Step: {}\nResponse: {}\nTool Response: {:?}\n", next_step, assistant_response, tool_response);
            }
        }

        // After executing all the steps, ask the model to provide a final summary of the steps and important information for the user
        self.chat.push_message(ChatMessage::user(
            "All steps have been completed. \
            Please provide a short and concise summary of the task, the steps taken, useful data from the tool responses, \
            and any other important information that would be useful for the user to know about, in a human-readable way."
        ));

        // Infer the assistant's final summary
        let final_summary = self.chat.infer_message(&ChatRole::Assistant, Some("# Task Summary and Important Information\n```\n"), &["```"]).into_content();
        final_summary
    }
}