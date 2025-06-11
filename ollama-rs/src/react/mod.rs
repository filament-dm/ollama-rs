//! ReAct-like (Reasoning and Acting) support for ollama-rs
//!
//! This module provides a ReAct-like coordinator that executes the ReAct pattern iteratively:
//! **Thought** → **Action** → **Observation** → repeat until completion.
//!
//! Unlike true ReAct, we don't insert explicit action statements into the conversation
//! trajectory. Instead, we leverage Ollama's function calling to handle actions automatically,
//! which provides a more natural integration with the underlying LLM capabilities.
//!
//! ## Key Features
//!
//! - **Iterator-based**: Get explicit control over each reasoning step
//! - **ReAct methodology**: Structured thinking → tool usage → observation loop
//! - **Ollama integration**: Automatic tool calling without manual action parsing
//! - **Step tracking**: Each step contains thought, actions taken, and observations
//! - **Customizable prompts**: Override the default system prompt for specialized use cases
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use ollama_rs::{
//!     react::{Coordinator, Step},
//!     generation::chat::ChatMessage,
//!     Ollama,
//! };
//!
//! # #[tokio::main]
//! # async fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Set up coordinator with tools (fresh start)
//! let ollama = Ollama::default();
//! let mut coordinator = Coordinator::<Vec<ChatMessage>>::new(ollama, "qwen3:0.6b".to_string(), None)
//!     .add_tool(MySearchTool)
//!     .add_tool(MyCalculatorTool)
//!     .system_prompt("You are a helpful assistant specialized in research and analysis.")
//!     .debug(true)
//!     .max_iterations(5);
//!
//! // Or with existing conversation history
//! let existing_history = vec![
//!     ChatMessage::user("Previous question".to_string()),
//!     ChatMessage::assistant("Previous answer".to_string()),
//! ];
//! let mut coordinator_with_history = Coordinator::<Vec<ChatMessage>>::new(ollama.clone(), "qwen3:0.6b".to_string(), Some(existing_history))
//!     .add_tool(MySearchTool);
//!
//! // Start the reasoning process
//! coordinator.start("What's the weather in Tokyo and how does it compare to New York?").await?;
//!
//! // Iterate through reasoning steps
//! while let Some(step) = coordinator.next_step().await? {
//!     println!("Step {}: {}", step.step_number, step.thought);
//!     
//!     if !step.actions.is_empty() {
//!         println!("Actions taken: {:?}", step.actions);
//!         println!("Observations: {:?}", step.observations);
//!     }
//!     
//!     if step.is_final {
//!         println!("Reasoning complete!");
//!         break;
//!     }
//! }
//!
//! // Or collect all steps at once
//! let all_steps = coordinator.collect_all_steps().await?;
//! for step in all_steps {
//!     // Process each step...
//! }
//! # Ok(())
//! # }
//! ```

pub mod coordinator;

pub use coordinator::{Coordinator, Step};
