//! Vision bridge: lets a text-only "brain" model (for example DeepSeek) use a
//! vision-capable provider (for example GPT-5.6 Luna) as its "eyes".
//!
//! The bridge intercepts user-supplied images before they are written to
//! history, sends all images in one batched Responses request to the
//! configured vision provider, and replaces the images with a textual
//! description. The active model never receives image content, so the
//! session keeps a single provider and `for_prompt`'s normal modality
//! stripping never has to drop the image after the fact.

use std::sync::Arc;

use codex_api::ResponseEvent;
use codex_api::ResponsesApiRequest;
use codex_api::ResponsesClient;
use codex_api::ResponsesOptions;
use codex_api::ReqwestTransport;
use codex_login::AuthManager;
use codex_login::default_client::create_client;
use codex_model_provider::SharedModelProvider;
use codex_model_provider::create_model_provider;
use codex_model_provider_info::ModelProviderInfo;
use codex_protocol::error::CodexErr;
use codex_protocol::error::Result as CodexResult;
use codex_protocol::models::ContentItem;
use codex_protocol::models::DEFAULT_IMAGE_DETAIL;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ImageDetail;
use codex_protocol::models::ResponseItem;
use codex_protocol::user_input::UserInput;
use codex_utils_image::data_url_from_bytes;
use codex_utils_image::load_data_url_for_prompt;
use codex_utils_image::PromptImageMode;
use futures::StreamExt;

use crate::config::Config;

/// Instructions sent alongside the images to the vision provider.
///
/// The provider is asked to describe each image separately so a batched
/// request still produces per-image sections a text-only model can consume.
pub(crate) const VISION_DESCRIPTION_INSTRUCTIONS: &str = "\
Describe each image for a coding agent. Be precise about visible text, error \
messages, terminal output, filenames, UI state, controls, and spatial \
relationships. Number each description as [Image 1], [Image 2], and so on. \
Do not speculate about information that is not visible.";

/// Marker prefix prepended to descriptions so DeepSeek can tell generated
/// descriptions from user text in later turns.
pub(crate) const VISION_DESCRIPTION_PREFIX: &str = "[Image described by vision model]";

/// Explicit text attached when the vision provider fails, so the text-only
/// model never hallucinates that it saw the image.
pub(crate) const VISION_BRIDGE_FAILURE_MESSAGE: &str =
    "Unable to analyze the attached image with the configured vision provider.";

/// Rewrites a non-image data URL into an image data URL the vision provider
/// accepts. `view_image` and the local-image path label their payloads as
/// `application/octet-stream`, and OpenAI-compatible Responses endpoints
/// reject that MIME type for `input_image`. Remote URLs and correctly labeled
/// `image/*` data URLs pass through unchanged.
fn normalize_image_url(image_url: &str) -> CodexResult<String> {
    if !image_url
        .get(.."data:".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:"))
    {
        return Ok(image_url.to_string());
    }
    if image_url
        .get(.."data:image/".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("data:image/"))
    {
        return Ok(image_url.to_string());
    }
    let image = load_data_url_for_prompt(image_url, PromptImageMode::ResizeToFit).map_err(
        |err| CodexErr::InvalidRequest(format!("invalid image for vision provider: {err}")),
    )?;
    Ok(image.into_data_url())
}

/// One-off client used to describe images through a secondary provider.
///
/// The session's main provider is left untouched: this bridge owns its own
/// `SharedModelProvider` built from the configured `vision_provider`, so one
/// session can legitimately talk to two providers without making the session
/// provider mutable.
#[derive(Clone)]
pub(crate) struct VisionBridge {
    provider: SharedModelProvider,
    model: String,
}

impl VisionBridge {
    pub(crate) fn new(
        provider_info: ModelProviderInfo,
        auth_manager: Option<Arc<AuthManager>>,
        model: String,
    ) -> Self {
        Self {
            provider: create_model_provider(provider_info, auth_manager),
            model,
        }
    }

    /// Sends all images in one batched Responses request and returns the
    /// combined description.
    async fn describe_images(
        &self,
        images: Vec<(String, Option<ImageDetail>)>,
    ) -> CodexResult<String> {
        if images.is_empty() {
            return Ok(String::new());
        }

        let api_provider = self.provider.api_provider().await?;
        let api_auth = self.provider.api_auth().await?;
        let client = ResponsesClient::new(
            ReqwestTransport::from_http_client(create_client()),
            api_provider,
            api_auth,
        );

        let mut content = Vec::with_capacity(images.len() + 1);
        content.push(ContentItem::InputText {
            text: VISION_DESCRIPTION_INSTRUCTIONS.to_string(),
        });
        for (image_url, detail) in images {
            content.push(ContentItem::InputImage {
                image_url: normalize_image_url(&image_url)?,
                detail: Some(detail.unwrap_or(DEFAULT_IMAGE_DETAIL)),
            });
        }

        let request = ResponsesApiRequest {
            model: self.model.clone(),
            instructions: String::new(),
            input: vec![ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content,
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            }],
            tools: None,
            tool_choice: "none".to_string(),
            parallel_tool_calls: false,
            reasoning: None,
            store: false,
            stream: true,
            stream_options: None,
            include: Vec::new(),
            service_tier: None,
            prompt_cache_key: None,
            text: None,
            client_metadata: None,
        };

        let mut stream = client
            .stream_request(request, ResponsesOptions::default())
            .await
            .map_err(|err| CodexErr::Stream(format!("vision request failed: {err}")))?;

        let mut description = String::new();
        while let Some(event) = stream.next().await {
            match event.map_err(|err| {
                CodexErr::Stream(format!("vision stream failed: {err}"))
            })? {
                ResponseEvent::OutputTextDelta(delta) => description.push_str(&delta),
                ResponseEvent::Completed { .. } => break,
                _ => {}
            }
        }

        Ok(description)
    }
}

/// Returns a `VisionBridge` when the session configures one.
pub(crate) fn vision_bridge_from_config(
    config: &Config,
    auth_manager: Option<Arc<AuthManager>>,
) -> Option<VisionBridge> {
    let provider_info = config.vision_provider.as_ref()?.clone();
    let model = config.vision_model.clone()?;
    Some(VisionBridge::new(provider_info, auth_manager, model))
}

/// Whether any user input item carries an image that needs describing.
pub(crate) fn user_input_has_images(input: &[UserInput]) -> bool {
    input.iter().any(|item| {
        matches!(
            item,
            UserInput::Image { .. } | UserInput::LocalImage { .. }
        )
    })
}

/// Whether any response item (for example a tool output such as `view_image`)
/// carries image content that needs describing.
pub(crate) fn response_items_have_images(items: &[ResponseItem]) -> bool {
    items.iter().any(|item| match item {
        ResponseItem::FunctionCallOutput { output, .. }
        | ResponseItem::CustomToolCallOutput { output, .. } => output
            .content_items()
            .is_some_and(|content| {
                content
                    .iter()
                    .any(|item| matches!(item, FunctionCallOutputContentItem::InputImage { .. }))
            }),
        _ => false,
    })
}

/// Replaces image content inside tool outputs with a single batched vision
/// description.
///
/// All images across the recorded items are sent to the vision provider in one
/// request. The original images are preserved; the description is attached
/// after the first image in each affected output, and serialization decides
/// what the active model receives. On failure an explicit error message is
/// attached instead of silently dropping the image.
pub(crate) async fn bridge_response_item_images(
    items: &mut Vec<ResponseItem>,
    vision: &VisionBridge,
) -> CodexResult<()> {
    let mut images = Vec::new();
    for item in items.iter() {
        if let ResponseItem::FunctionCallOutput { output, .. }
        | ResponseItem::CustomToolCallOutput { output, .. } = item
        {
            if let Some(content) = output.content_items() {
                for content_item in content {
                    if let FunctionCallOutputContentItem::InputImage {
                        image_url,
                        detail,
                    } = content_item
                    {
                        images.push((image_url.clone(), *detail));
                    }
                }
            }
        }
    }
    if images.is_empty() {
        return Ok(());
    }

    match vision.describe_images(images).await {
        Ok(description) if !description.trim().is_empty() => {
            attach_response_item_description(items, &description);
            Ok(())
        }
        Ok(_) => Ok(()),
        Err(err) => {
            tracing::warn!(error = %err, "vision bridge failed for tool output");
            attach_response_item_description(items, VISION_BRIDGE_FAILURE_MESSAGE);
            Ok(())
        }
    }
}

/// Attaches `description` to tool outputs without removing the images.
///
/// The description is inserted immediately after the first image in each
/// affected output; images and non-image content keep their order, so a
/// vision-capable model can still see the original image later.
pub(crate) fn attach_response_item_description(
    items: &mut Vec<ResponseItem>,
    description: &str,
) {
    for item in items.iter_mut() {
        if let ResponseItem::FunctionCallOutput { output, .. }
        | ResponseItem::CustomToolCallOutput { output, .. } = item
        {
            if let Some(content) = output.content_items_mut() {
                if let Some(index) = content.iter().position(|content_item| {
                    matches!(
                        content_item,
                        FunctionCallOutputContentItem::InputImage { .. }
                    )
                }) {
                    content.insert(
                        index + 1,
                        FunctionCallOutputContentItem::InputText {
                            text: format!("{VISION_DESCRIPTION_PREFIX}\n{description}"),
                        },
                    );
                }
            }
        }
    }
}

/// Describes images in `input` through the vision provider without removing
/// them.
///
/// Local images are read into data URLs first. On success the images are
/// kept and one `UserInput::Text` carrying the description is inserted after
/// the first image. On failure an explicit error message is attached instead,
/// so the model never hallucinates that it saw the image.
pub(crate) async fn bridge_user_input_images(
    input: &mut Vec<UserInput>,
    vision: &VisionBridge,
) -> CodexResult<()> {
    if !user_input_has_images(input) {
        return Ok(());
    }

    let mut images = Vec::new();
    for item in input.iter() {
        match item {
            UserInput::Image { image_url, detail } => images.push((image_url.clone(), *detail)),
            UserInput::LocalImage { path, detail } => match std::fs::read(path) {
                Ok(bytes) => images.push((
                    data_url_from_bytes("application/octet-stream", &bytes),
                    *detail,
                )),
                Err(err) => tracing::warn!(
                    error = %err,
                    path = %path.display(),
                    "vision bridge could not read local image; omitting it from the batch"
                ),
            },
            _ => {}
        }
    }
    if images.is_empty() {
        return Ok(());
    }

    match vision.describe_images(images).await {
        Ok(description) if !description.trim().is_empty() => {
            attach_vision_description(input, &description);
            Ok(())
        }
        Ok(_) => Ok(()),
        Err(err) => {
            tracing::warn!(error = %err, "vision bridge failed for user input");
            attach_vision_description(input, VISION_BRIDGE_FAILURE_MESSAGE);
            Ok(())
        }
    }
}

/// Attaches `description` to `input` without removing the images.
///
/// The description is inserted immediately after the first image; images and
/// non-image items keep their order, so a vision-capable model can still see
/// the original image later.
pub(crate) fn attach_vision_description(input: &mut Vec<UserInput>, description: &str) {
    let mut output = Vec::with_capacity(input.len());
    let mut inserted = false;
    for item in input.iter() {
        output.push(item.clone());
        if !inserted
            && matches!(item, UserInput::Image { .. } | UserInput::LocalImage { .. })
        {
            output.push(UserInput::Text {
                text: format!("{VISION_DESCRIPTION_PREFIX}\n{description}"),
                text_elements: Vec::new(),
            }
            );
            inserted = true;
        }
    }
    *input = output;
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::FunctionCallOutputBody;
    use codex_protocol::models::FunctionCallOutputPayload;
    use codex_protocol::models::ImageDetail;

    fn text_item(text: &str) -> UserInput {
        UserInput::Text {
            text: text.to_string(),
            text_elements: Vec::new(),
        }
    }

    fn image_item(image_url: &str) -> UserInput {
        UserInput::Image {
            image_url: image_url.to_string(),
            detail: Some(ImageDetail::High),
        }
    }

    const TINY_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

    #[test]
    fn normalizes_octet_stream_data_url_to_image_mime() {
        let octet_stream_url = format!("data:application/octet-stream;base64,{TINY_PNG_BASE64}");
        let normalized = normalize_image_url(&octet_stream_url)
            .expect("octet-stream data URL should be normalized");
        assert!(
            normalized.starts_with("data:image/png;base64,"),
            "expected an image/png data URL, got: {normalized}"
        );
    }

    #[test]
    fn leaves_supported_image_urls_unchanged() {
        assert_eq!(
            normalize_image_url("data:image/png;base64,abc")
                .expect("png data URL should pass through"),
            "data:image/png;base64,abc"
        );
        assert_eq!(
            normalize_image_url("https://example.com/screenshot.png")
                .expect("remote URL should pass through"),
            "https://example.com/screenshot.png"
        );
    }

    fn local_image_item(path: &std::path::Path) -> UserInput {
        UserInput::LocalImage {
            path: path.to_path_buf(),
            detail: None,
        }
    }

    fn function_call_output_with_images() -> ResponseItem {
        ResponseItem::FunctionCallOutput {
            id: None,
            call_id: "call-1".to_string(),
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::ContentItems(vec![
                    FunctionCallOutputContentItem::InputImage {
                        image_url: "data:image/png;base64,abc".to_string(),
                        detail: Some(ImageDetail::High),
                    },
                    FunctionCallOutputContentItem::InputImage {
                        image_url: "data:image/png;base64,def".to_string(),
                        detail: None,
                    },
                    FunctionCallOutputContentItem::InputText {
                        text: "tail text".to_string(),
                    },
                ]),
                success: Some(true),
            },
            internal_chat_message_metadata_passthrough: None,
        }
    }

    #[test]
    fn detects_images() {
        assert!(!user_input_has_images(&[text_item("hello")]));
        assert!(user_input_has_images(&[image_item("data:image/png;base64,abc")]));
        assert!(user_input_has_images(&[
            text_item("hello"),
            local_image_item(std::path::Path::new("/tmp/x.png")),
        ]));
    }

    #[test]
    fn keeps_images_and_attaches_description() {
        let mut input = vec![
            text_item("here is the screenshot:"),
            image_item("data:image/png;base64,abc"),
            text_item("what went wrong?"),
            image_item("data:image/png;base64,def"),
        ];
        attach_vision_description(&mut input, "terminal shows a compile error");

        assert_eq!(
            input,
            vec![
                text_item("here is the screenshot:"),
                image_item("data:image/png;base64,abc"),
                text_item("[Image described by vision model]\nterminal shows a compile error"),
                text_item("what went wrong?"),
                image_item("data:image/png;base64,def"),
            ]
        );
    }

    #[test]
    fn leaves_input_untouched_without_images() {
        let mut input = vec![text_item("plain text")];
        attach_vision_description(&mut input, "description");
        assert_eq!(input, vec![text_item("plain text")]);
    }

    #[test]
    fn detects_response_item_images() {
        assert!(!response_items_have_images(&[
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![ContentItem::InputText {
                    text: "no image".to_string(),
                }],
                phase: None,
                internal_chat_message_metadata_passthrough: None,
            },
        ]));
        assert!(response_items_have_images(&[function_call_output_with_images()]));
    }

    #[test]
    fn keeps_tool_images_and_attaches_description() {
        let mut items = vec![function_call_output_with_images()];
        attach_response_item_description(&mut items, "a terminal with a compile error");

        let ResponseItem::FunctionCallOutput { output, .. } = &items[0] else {
            panic!("expected function call output");
        };
        let content = output.content_items().expect("content items");
        assert_eq!(content.len(), 4);
        assert_eq!(
            content[0],
            FunctionCallOutputContentItem::InputImage {
                image_url: "data:image/png;base64,abc".to_string(),
                detail: Some(ImageDetail::High),
            }
        );
        assert_eq!(
            content[1],
            FunctionCallOutputContentItem::InputText {
                text: "[Image described by vision model]\na terminal with a compile error"
                    .to_string(),
            }
        );
        assert_eq!(
            content[2],
            FunctionCallOutputContentItem::InputImage {
                image_url: "data:image/png;base64,def".to_string(),
                detail: None,
            }
        );
        assert_eq!(
            content[3],
            FunctionCallOutputContentItem::InputText {
                text: "tail text".to_string(),
            }
        );
    }

    /// Standalone bridge test: a `VisionBridge` pointed at a mock Responses
    /// endpoint describes an image with no Codex session or main provider
    /// involved. Failures here are Luna-bridge problems, not DeepSeek
    /// problems.
    #[tokio::test]
    async fn describes_image_through_configured_provider_without_deepseek() {
        use wiremock::Mock;
        use wiremock::MockServer;
        use wiremock::ResponseTemplate;
        use wiremock::matchers::method;
        use wiremock::matchers::path;

        let server = MockServer::start().await;
        let description = "terminal shows a compile error";
        let sse = format!(
            "data: {{\"type\":\"response.created\",\"response\":{{\"id\":\"r1\"}}}}\n\n\
             data: {{\"type\":\"response.output_text.delta\",\"delta\":\"{description}\"}}\n\n\
             data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"r1\",\"usage\":{{\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}}}}}}\n\n"
        );
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-type", "text/event-stream")
                    .set_body_string(sse),
            )
            .mount(&server)
            .await;

        let vision = VisionBridge::new(
            ModelProviderInfo {
                name: "Luna".to_string(),
                base_url: Some(format!("{}/v1", server.uri())),
                experimental_bearer_token: Some("test-token".to_string()),
                requires_openai_auth: false,
                ..Default::default()
            },
            /*auth_manager*/ None,
            "gpt-5.6-luna".to_string(),
        );

        let result = vision
            .describe_images(vec![("data:image/png;base64,abc".to_string(), None)])
            .await
            .expect("vision bridge should describe the image");
        assert_eq!(result, description);
    }

    /// A failed vision call must attach an explicit error message so the
    /// text-only model never hallucinates that it saw the image.
    #[tokio::test]
    async fn failed_vision_call_attaches_explicit_error_message() {
        use wiremock::Mock;
        use wiremock::MockServer;
        use wiremock::ResponseTemplate;
        use wiremock::matchers::method;
        use wiremock::matchers::path;

        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/responses"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;

        let vision = VisionBridge::new(
            ModelProviderInfo {
                name: "Luna".to_string(),
                base_url: Some(format!("{}/v1", server.uri())),
                experimental_bearer_token: Some("test-token".to_string()),
                requires_openai_auth: false,
                ..Default::default()
            },
            /*auth_manager*/ None,
            "gpt-5.6-luna".to_string(),
        );

        let mut input = vec![
            text_item("here is the screenshot:"),
            image_item("data:image/png;base64,abc"),
        ];
        bridge_user_input_images(&mut input, &vision)
            .await
            .expect("bridge should not fail the turn");

        assert_eq!(
            input,
            vec![
                text_item("here is the screenshot:"),
                image_item("data:image/png;base64,abc"),
                text_item(&format!(
                    "{VISION_DESCRIPTION_PREFIX}\n{VISION_BRIDGE_FAILURE_MESSAGE}"
                )),
            ]
        );
    }
}
