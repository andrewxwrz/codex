#![cfg(not(target_os = "windows"))]

//! End-to-end coverage for the Chat Completions wire protocol (`wire_api =
//! "chat"`): the provider is reached through `ChatClient`, the request is
//! shaped like a chat-completions call, and the shared
//! `ResponseStream`/`ResponseEvent` machinery (including tool execution)
//! handles the rest.

use codex_core::TurnInputRequest;
use codex_model_provider_info::WireApi;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::mount_chat_sse_once;
use core_test_support::responses::mount_chat_sse_sequence;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use core_test_support::wait_for_event_with_timeout;
use serde_json::Value;
use serde_json::json;
use tokio::time::Duration;

const CHAT_TURN_COMPLETE_TIMEOUT: Duration = Duration::from_secs(30);

/// Builds a Chat-Completions-shaped SSE body. `done` appends the `[DONE]`
/// sentinel, which the parser needs to emit `ResponseEvent::Completed` after
/// a `finish_reason = "tool_calls"` stream.
fn chat_sse(events: Vec<Value>, done: bool) -> String {
    let mut body = String::new();
    for ev in events {
        body.push_str(&format!("data: {ev}\n\n"));
    }
    if done {
        body.push_str("data: [DONE]\n\n");
    }
    body
}

fn chat_delta_content(text: &str) -> Value {
    json!({"choices": [{"delta": {"content": text}}]})
}

fn chat_finish(finish_reason: &str) -> Value {
    json!({"choices": [{"finish_reason": finish_reason}]})
}

fn user_text_turn(test: &TestCodex, model: String, text: &str) -> TurnInputRequest {
    let (sandbox_policy, permission_profile) =
        turn_permission_fields(PermissionProfile::Disabled, test.config.cwd.as_path());
    TurnInputRequest::user_input(vec![UserInput::Text {
        text: text.to_string(),
        text_elements: Vec::new(),
    }])
    .with_thread_settings(ThreadSettingsOverrides {
        approval_policy: Some(AskForApproval::Never),
        sandbox_policy: Some(sandbox_policy),
        permission_profile,
        collaboration_mode: Some(CollaborationMode {
            mode: ModeKind::Default,
            settings: Settings {
                model,
                reasoning_effort: None,
                developer_instructions: None,
            },
        }),
        ..Default::default()
    })
}

/// Points the session's main provider at the mock server with the Chat wire
/// enabled and a fixed bearer token so no environment credentials are needed.
fn chat_wire_builder() -> core_test_support::test_codex::TestCodexBuilder {
    test_codex().with_config(|config| {
        config.model_provider.wire_api = WireApi::Chat;
        config.model_provider.requires_openai_auth = false;
        config.model_provider.experimental_bearer_token = Some("test-token".to_string());
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_provider_streams_turn_through_chat_completions() -> anyhow::Result<()> {
    let server = start_mock_server().await;
    let mock = mount_chat_sse_once(
        &server,
        chat_sse(
            vec![
                chat_delta_content("hello from chat"),
                chat_finish("stop"),
            ],
            /*done*/ false,
        ),
    )
    .await;

    let mut builder = chat_wire_builder();
    let test = builder.build_with_auto_env(&server).await?;
    let codex = test.codex.clone();
    let model = test.session_configured.model.clone();

    codex
        .start_or_steer_turn(user_text_turn(&test, model, "say hi"))
        .await?;
    wait_for_event_with_timeout(
        &codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        CHAT_TURN_COMPLETE_TIMEOUT,
    )
    .await;

    let req = mock.single_request();
    let body = req.body_json();
    assert_eq!(body["model"], "gpt-5.5");
    assert_eq!(body["stream"], true);
    let messages = body["messages"].as_array().expect("messages array");
    assert_eq!(messages[0]["role"], "system");
    assert!(messages[0]["content"].as_str().is_some());
    // History may contain developer-role messages before the user turn.
    assert!(
        messages.iter().any(|message| {
            message["role"] == "user"
                && serde_json::to_string(&message["content"])
                    .is_ok_and(|content| content.contains("say hi"))
        }),
        "user message should contain the user turn text"
    );

    let tools = body["tools"].as_array().expect("tools array");
    assert!(!tools.is_empty(), "chat request must include tool definitions");
    assert!(
        tools
            .iter()
            .all(|tool| tool.get("type").and_then(Value::as_str) == Some("function")),
        "only plain function tools are representable on the chat wire"
    );

    assert_eq!(
        req.headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok()),
        Some("Bearer test-token")
    );
    assert!(
        req.headers().get("session-id").is_some(),
        "chat request should carry the session header"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_provider_tool_call_reaches_existing_tool_machinery() -> anyhow::Result<()> {
    let server = start_mock_server().await;
    let call_id = "call-exec-chat";
    let arguments = json!({
        "cmd": "echo chat-tool-ok",
        "shell": "bash",
        "login": false,
    })
    .to_string();

    let tool_sse = chat_sse(
        vec![
            json!({"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "id": call_id,
                "function": {"name": "exec_command", "arguments": arguments}
            }]}}]}),
            chat_finish("tool_calls"),
        ],
        /*done*/ true,
    );
    let done_sse = chat_sse(
        vec![
            chat_delta_content("done"),
            chat_finish("stop"),
        ],
        /*done*/ false,
    );
    let mock = mount_chat_sse_sequence(&server, vec![tool_sse, done_sse]).await;

    let mut builder = chat_wire_builder();
    let test = builder.build_with_auto_env(&server).await?;
    let codex = test.codex.clone();
    let model = test.session_configured.model.clone();

    codex
        .start_or_steer_turn(user_text_turn(&test, model, "run a command"))
        .await?;
    wait_for_event_with_timeout(
        &codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        CHAT_TURN_COMPLETE_TIMEOUT,
    )
    .await;

    let requests = mock.requests();
    assert_eq!(requests.len(), 2, "tool call should trigger a follow-up request");

    // The follow-up request must contain the tool result as a chat `tool`
    // message, proving the existing tool loop executed the call and fed the
    // output back through the Chat request builder.
    let second = requests.last().expect("second request");
    let body = second.body_json();
    let messages = body["messages"].as_array().expect("messages array");
    let tool_message = messages
        .iter()
        .find(|message| message["role"] == "tool")
        .expect("follow-up request should contain a tool message");
    assert_eq!(tool_message["tool_call_id"], call_id);
    let serialized_content = serde_json::to_string(&tool_message["content"]).unwrap();
    assert!(
        serialized_content.contains("chat-tool-ok"),
        "tool output should be serialized back into the chat request: {serialized_content}"
    );

    Ok(())
}
