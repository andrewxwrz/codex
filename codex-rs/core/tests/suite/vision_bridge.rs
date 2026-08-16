#![cfg(not(target_os = "windows"))]

//! Integration coverage for the "DeepSeek brain + Luna eyes" vision bridge:
//! when the active model is text-only and a vision provider is configured,
//! user-supplied images are described through the vision provider and the
//! main provider only ever receives the textual description.

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use codex_core::TurnInputRequest;
use codex_exec_server::LOCAL_ENVIRONMENT_ID;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::config_types::CollaborationMode;
use codex_protocol::config_types::ModeKind;
use codex_protocol::config_types::Settings;
use codex_protocol::models::ImageDetail;
use codex_protocol::models::PermissionProfile;
use codex_protocol::openai_models::InputModality;
use codex_protocol::protocol::AskForApproval;
use codex_protocol::protocol::EventMsg;
use codex_protocol::protocol::ThreadSettingsOverrides;
use codex_protocol::user_input::UserInput;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_output_text_delta;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::mount_sse_sequence;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::test_codex::TestCodex;
use core_test_support::test_codex::test_codex;
use core_test_support::test_codex::turn_permission_fields;
use core_test_support::wait_for_event_with_timeout;
use image::DynamicImage;
use image::ImageBuffer;
use image::Rgba;
use serde_json::Value;
use serde_json::json;
use std::io::Cursor;
use tokio::time::Duration;

const VISION_TURN_COMPLETE_TIMEOUT: Duration = Duration::from_secs(30);
const VISION_DESCRIPTION: &str = "The screenshot shows a terminal with a compile error.";

fn user_turn_with_image(test: &TestCodex, model: String, image_url: String) -> TurnInputRequest {
    user_turn(
        test,
        model,
        vec![UserInput::Image {
            image_url,
            detail: Some(ImageDetail::High),
        }],
    )
}

fn user_text_turn(test: &TestCodex, model: String, text: &str) -> TurnInputRequest {
    user_turn(
        test,
        model,
        vec![UserInput::Text {
            text: text.to_string(),
            text_elements: Vec::new(),
        }],
    )
}

fn user_turn(test: &TestCodex, model: String, content: Vec<UserInput>) -> TurnInputRequest {
    let (sandbox_policy, permission_profile) =
        turn_permission_fields(PermissionProfile::Disabled, test.config.cwd.as_path());
    TurnInputRequest::user_input(content)
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

fn png_data_url() -> String {
    let image = ImageBuffer::from_pixel(16, 16, Rgba([20u8, 40, 60, 255]));
    let mut cursor = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("encode png");
    format!(
        "data:image/png;base64,{}",
        BASE64_STANDARD.encode(cursor.into_inner())
    )
}

/// Same image bytes as `png_data_url`, but labeled the way `view_image` labels
/// tool-returned screenshots (`application/octet-stream`). The vision bridge
/// must normalize this MIME type before the request reaches the provider.
fn octet_stream_data_url() -> String {
    let image = ImageBuffer::from_pixel(16, 16, Rgba([20u8, 40, 60, 255]));
    let mut cursor = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("encode png");
    format!(
        "data:application/octet-stream;base64,{}",
        BASE64_STANDARD.encode(cursor.into_inner())
    )
}

async fn write_workspace_png(test: &TestCodex, rel_path: &str) -> anyhow::Result<String> {
    let abs_path = test.config.cwd.join(rel_path);
    let abs_path_uri = codex_utils_path_uri::PathUri::from_host_native_path(&abs_path)?;
    test.fs()
        .write_file(
            &abs_path_uri,
            png_bytes(16, 16, [20u8, 40, 60, 255]),
            /*sandbox*/ None,
        )
        .await?;
    Ok(rel_path.to_string())
}

fn png_bytes(width: u32, height: u32, rgba: [u8; 4]) -> Vec<u8> {
    let image = ImageBuffer::from_pixel(width, height, Rgba(rgba));
    let mut cursor = Cursor::new(Vec::new());
    DynamicImage::ImageRgba8(image)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .expect("encode png");
    cursor.into_inner()
}

fn request_input_items(body: &Value) -> Vec<&Value> {
    body.get("input")
        .and_then(Value::as_array)
        .map(|items| items.iter().collect())
        .unwrap_or_default()
}

fn has_input_image(body: &Value) -> bool {
    request_input_items(body).iter().any(|item| {
        item.get("content")
            .and_then(Value::as_array)
            .is_some_and(|content| {
                content.iter().any(|span| {
                    span.get("type").and_then(Value::as_str) == Some("input_image")
                })
            })
    })
}

fn vision_image_urls(body: &Value) -> Vec<&str> {
    request_input_items(body)
        .iter()
        .flat_map(|item| {
            item.get("content")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(|span| {
                    if span.get("type").and_then(Value::as_str) == Some("input_image") {
                        span.get("image_url").and_then(Value::as_str)
                    } else {
                        None
                    }
                })
        })
        .collect()
}

fn has_text_containing(body: &Value, needle: &str) -> bool {
    request_input_items(body).iter().any(|item| {
        item.get("content")
            .and_then(Value::as_array)
            .is_some_and(|content| {
                content.iter().any(|span| {
                    span.get("text")
                        .and_then(Value::as_str)
                        .is_some_and(|text| text.contains(needle))
                })
            })
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn text_only_model_receives_luna_description_instead_of_image() -> anyhow::Result<()> {
    let main_server = start_mock_server().await;
    let vision_server = start_mock_server().await;
    let vision_base_url = format!("{}/v1", vision_server.uri());

    // The vision provider points at the second mock; `experimental_bearer_token`
    // avoids environment-variable auth in the test.
    let mut builder = test_codex()
        .with_model_info_override("gpt-5.5", |model_info| {
            model_info.input_modalities = vec![InputModality::Text];
        })
        .with_config(move |config| {
            config.vision_provider_id = Some("luna".to_string());
            config.vision_provider = Some(ModelProviderInfo {
                name: "Luna".to_string(),
                base_url: Some(vision_base_url),
                experimental_bearer_token: Some("test-token".to_string()),
                requires_openai_auth: false,
                ..Default::default()
            });
            config.vision_model = Some("gpt-5.6-luna".to_string());
        });
    let test = builder.build_with_auto_env(&main_server).await?;
    let codex = test.codex.clone();
    let model = test.session_configured.model.clone();

    let vision_sse = sse(vec![
        ev_response_created("vision-1"),
        ev_output_text_delta(VISION_DESCRIPTION),
        ev_completed("vision-1"),
    ]);
    let vision_mock = mount_sse_once(&vision_server, vision_sse).await;

    let main_sse = sse(vec![
        ev_response_created("main-1"),
        ev_assistant_message("msg-1", "ok"),
        ev_completed("main-1"),
    ]);
    let main_mock = mount_sse_once(&main_server, main_sse).await;

    codex
        .start_or_steer_turn(user_turn_with_image(&test, model, octet_stream_data_url()))
        .await?;

    wait_for_event_with_timeout(
        &codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        VISION_TURN_COMPLETE_TIMEOUT,
    )
    .await;

    // Luna received one request containing the image and the vision model.
    let vision_body = vision_mock.single_request().body_json();
    assert_eq!(
        vision_body
            .get("model")
            .and_then(Value::as_str),
        Some("gpt-5.6-luna")
    );
    assert!(has_input_image(&vision_body), "Luna request must include the image");
    assert!(
        vision_image_urls(&vision_body)
            .iter()
            .any(|url| url.starts_with("data:image/png;base64,")),
        "Luna request must receive a normalized image MIME type"
    );

    // DeepSeek received exactly one request, with the description in history
    // and no image content anywhere in the prompt.
    let main_body = main_mock.single_request().body_json();
    assert_eq!(
        main_body.get("model").and_then(Value::as_str),
        Some("gpt-5.5")
    );
    assert!(
        has_text_containing(&main_body, "[Image described by vision model]"),
        "main request must contain the vision description"
    );
    assert!(
        has_text_containing(&main_body, VISION_DESCRIPTION),
        "main request must contain the Luna description text"
    );
    assert!(
        has_text_containing(&main_body, "image content omitted"),
        "the original image must survive internally and only be stripped at serialization"
    );
    assert!(
        !has_input_image(&main_body),
        "text-only main provider must never receive image content"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn view_image_tool_output_is_bridged_for_text_only_model() -> anyhow::Result<()> {
    let main_server = start_mock_server().await;
    let vision_server = start_mock_server().await;
    let vision_base_url = format!("{}/v1", vision_server.uri());

    let mut builder = test_codex()
        .with_model_info_override("gpt-5.5", |model_info| {
            model_info.input_modalities = vec![InputModality::Text];
        })
        .with_config(move |config| {
            config.vision_provider_id = Some("luna".to_string());
            config.vision_provider = Some(ModelProviderInfo {
                name: "Luna".to_string(),
                base_url: Some(vision_base_url),
                experimental_bearer_token: Some("test-token".to_string()),
                requires_openai_auth: false,
                ..Default::default()
            });
            config.vision_model = Some("gpt-5.6-luna".to_string());
        });
    let test = builder.build_with_auto_env(&main_server).await?;
    let codex = test.codex.clone();
    let model = test.session_configured.model.clone();
    let screenshot_path = write_workspace_png(&test, "screenshot.png").await?;

    let vision_sse = sse(vec![
        ev_response_created("vision-1"),
        ev_output_text_delta(VISION_DESCRIPTION),
        ev_completed("vision-1"),
    ]);
    let vision_mock = mount_sse_once(&vision_server, vision_sse).await;

    let call_id = "call-view-image";
    let main_mock = mount_sse_sequence(
        &main_server,
        vec![
            // First response: the model calls view_image on the screenshot.
            sse(vec![
                ev_response_created("main-1"),
                ev_function_call(
                    call_id,
                    "view_image",
                    &json!({
                        "path": screenshot_path,
                        "environment_id": LOCAL_ENVIRONMENT_ID,
                    })
                    .to_string(),
                ),
                ev_completed("main-1"),
            ]),
            // Second response: the bridged description is in history.
            sse(vec![
                ev_response_created("main-2"),
                ev_assistant_message("msg-1", "ok"),
                ev_completed("main-2"),
            ]),
        ],
    )
    .await;

    codex
        .start_or_steer_turn(user_text_turn(
            &test,
            model,
            "look at the screenshot and tell me what the error is",
        ))
        .await?;

    wait_for_event_with_timeout(
        &codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        VISION_TURN_COMPLETE_TIMEOUT,
    )
    .await;

    // Luna saw the screenshot.
    let vision_body = vision_mock.single_request().body_json();
    assert_eq!(
        vision_body.get("model").and_then(Value::as_str),
        Some("gpt-5.6-luna")
    );
    assert!(has_input_image(&vision_body), "Luna request must include the image");

    // The second main request carries the description inside the tool output,
    // and no image content anywhere.
    let requests = main_mock.requests();
    assert_eq!(requests.len(), 2, "exactly two sampling requests");
    let second = requests.last().expect("second request");
    let body = second.body_json();
    let function_output = second.function_call_output(call_id);
    let output = function_output
        .get("output")
        .and_then(Value::as_array)
        .expect("function_call_output should be a content item array");
    // The image is preserved internally; serialization replaces it with the
    // omitted-content placeholder and the bridge attaches the description.
    assert_eq!(output.len(), 2, "image preserved plus description attached");
    assert_eq!(
        output[0].get("type").and_then(Value::as_str),
        Some("input_text"),
        "the stripped image placeholder must be text"
    );
    assert!(
        output[0]
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.contains("image content omitted")),
        "tool output must retain the original image internally until serialization"
    );
    assert!(
        output[1]
            .get("text")
            .and_then(Value::as_str)
            .is_some_and(|text| text.contains("[Image described by vision model]")),
        "tool output must contain the attached vision description"
    );
    assert!(
        !has_input_image(&body),
        "text-only main provider must never receive image content"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn image_capable_model_bypasses_vision_bridge() -> anyhow::Result<()> {
    let main_server = start_mock_server().await;
    let vision_server = start_mock_server().await;
    let vision_base_url = format!("{}/v1", vision_server.uri());

    // Default test model (gpt-5.5) is image-capable: even with a vision
    // provider configured, the bridge must not fire.
    let mut builder = test_codex().with_config(move |config| {
        config.vision_provider_id = Some("luna".to_string());
        config.vision_provider = Some(ModelProviderInfo {
            name: "Luna".to_string(),
            base_url: Some(vision_base_url),
            experimental_bearer_token: Some("test-token".to_string()),
            requires_openai_auth: false,
            ..Default::default()
        });
        config.vision_model = Some("gpt-5.6-luna".to_string());
    });
    let test = builder.build_with_auto_env(&main_server).await?;
    let codex = test.codex.clone();
    let model = test.session_configured.model.clone();

    let main_sse = sse(vec![
        ev_response_created("main-1"),
        ev_assistant_message("msg-1", "ok"),
        ev_completed("main-1"),
    ]);
    let main_mock = mount_sse_once(&main_server, main_sse).await;

    codex
        .start_or_steer_turn(user_turn_with_image(&test, model, png_data_url()))
        .await?;
    wait_for_event_with_timeout(
        &codex,
        |event| matches!(event, EventMsg::TurnComplete(_)),
        VISION_TURN_COMPLETE_TIMEOUT,
    )
    .await;

    let vision_requests = vision_server
        .received_requests()
        .await
        .unwrap_or_default();
    assert!(
        vision_requests.is_empty(),
        "image-capable models must bypass the vision bridge"
    );

    let main_body = main_mock.single_request().body_json();
    assert!(
        has_input_image(&main_body),
        "image-capable models receive the original image"
    );
    assert!(
        !has_text_containing(&main_body, "[Image described by vision model]"),
        "no description should be generated when the model can see images"
    );

    Ok(())
}
