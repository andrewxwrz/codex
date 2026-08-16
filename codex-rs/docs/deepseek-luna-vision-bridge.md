# DeepSeek brain + GPT-5.6 Luna eyes

This branch adds a "vision bridge" to Codex: when the active model cannot
consume images (`input_modalities` does not include `image`), attached images
are sent to a separately configured vision provider (for example GPT-5.6
Luna) over the Responses wire, and the returned textual description is
attached to history. The original image is preserved internally; serialization
decides what the active model receives. The session provider is never mutated.

The main provider (DeepSeek) speaks `wire_api = "chat"` natively — no proxy or
adapter is involved.

## Config

```toml
model_provider = "deepseek"
model = "deepseek-v4-pro"

# Luna as eyes: required for the bridge to activate.
vision_provider = "luna"
vision_model = "gpt-5.6-luna"

[model_providers.deepseek]
name = "DeepSeek V4"
base_url = "https://api.deepseek.com"
env_key = "DEEPSEEK_API_KEY"
wire_api = "chat"

[model_providers.luna]
name = "OpenAI Luna"
base_url = "https://api.openai.com/v1"
env_key = "OPENAI_API_KEY"
```

`vision_provider` must be a key in `model_providers`, and `vision_model` is
required whenever `vision_provider` is set. `deepseek-v4-pro`,
`deepseek-v4-flash`, and `gpt-5.6-luna` are bundled catalog models, so no
`model_catalog_json` is needed. DeepSeek entries are text-only; Luna accepts
`["text", "image"]`.

## How it works

```
DeepSeek (chat, text-only)
   │
   │ image encountered
   ▼
VisionBridge (own SharedModelProvider, built from the luna provider)
   │
   ▼
GPT-5.6 Luna (responses)  ── image ──▶ description
   │
   ▼
image + description preserved in history
   │
   ▼ DeepSeek serialization (text-only): image stripped, description kept
```

The routing decision is generic: the bridge fires when the active model's
`input_modalities` lack `Image` — not when the model happens to be DeepSeek.
Image-capable models bypass it entirely and receive the original image.

## What the patch changes

- `config/src/config_toml.rs` and `core/src/config/mod.rs`: new `vision_provider`
  and `vision_model` config keys, resolved against `model_providers` at load.
- `core/src/vision_bridge.rs` (new): `VisionBridge` builds its own
  `SharedModelProvider` from the vision provider (session provider untouched)
  and sends one Responses request, accumulating the `output_text.delta` stream.
  `bridge_user_input_images` and `bridge_response_item_images` attach the
  description after the first image without removing the images. On failure an
  explicit error message ("Unable to analyze the attached image...") is
  attached, so the text-only model never hallucinates that it saw the image.
- `core/src/session/mod.rs`: `record_user_prompt_and_emit_turn_item` bridges
  images before history insertion whenever the active model is text-only and a
  vision provider is configured; `record_conversation_items` bridges tool
  outputs the same way.
- `core/src/tools/handlers/view_image.rs`: the hard "you do not support image
  inputs" rejection is relaxed when a vision provider is configured, so a
  text-only brain can still take screenshots.

## Behavior notes

- The description is written to history next to the image, so later turns can
  ask follow-up questions about the screenshot without calling Luna again.
- Multiple images in one prompt are described in one Luna request; the
  instructions ask Luna to number descriptions `[Image 1]`, `[Image 2]`, ...
- The TUI still disables image paste for text-only models
  (`tui/src/chatwidget/settings.rs`). Pasting through the app-server or CLI
  paths triggers the bridge; relaxing the TUI gate is a follow-up.
- Tool-returned images (`view_image` and custom/MCP tool outputs carrying
  `input_image` content) are bridged when the active model is text-only.

## Testing

Automated regression tests (standalone Luna mock plus two-provider suite):

```sh
cargo test -p codex-core --lib vision_bridge
RUST_MIN_STACK=8388608 cargo test -p codex-core --test all vision_bridge
```

The stack size matches the repo's CI/justfile setting. Manual end-to-end tests
need a client that can attach images to a text-only model (the TUI currently
blocks paste for text-only models) plus a DeepSeek API key; see
`deepseek-chat-manual-smoke.md` for the Chat-wire smoke procedure.
