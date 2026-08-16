# DeepSeek Chat wire — manual smoke test

This is a credential-dependent manual/integration smoke test. It is not part of
the automated suite and no API keys are committed anywhere in this repo.

## Prerequisites

- A patched `codex` binary built from the `deepseek-v4-compat` commit onward:

  ```sh
  cd codex-rs
  cargo build --bin codex
  ```

- A DeepSeek API key: `export DEEPSEEK_API_KEY=...`

## Config

Add to `~/.codex/config.toml`:

```toml
model_provider = "deepseek"
model = "deepseek-v4-pro"

[model_providers.deepseek]
name = "DeepSeek V4"
base_url = "https://api.deepseek.com"
env_key = "DEEPSEEK_API_KEY"
wire_api = "chat"
```

`deepseek-v4-pro` is a bundled catalog model in this build (text-only), so no
`model_catalog_json` is required.

## Scenario 1 — plain text response

```sh
RUST_LOG=info target/debug/codex "Say hello in one sentence."
```

Expected:

- Logs show a `POST` to `https://api.deepseek.com/chat/completions`
  (`chat.stream_request`, `api.path = "chat/completions"`).
- The reply is a normal Codex assistant message.

If this fails, capture the request body (`RUST_LOG=trace`) and check for
protocol mismatches: message roles, tool schema fields, or headers.

## Scenario 2 — tool call + tool result + follow-up

```sh
RUST_LOG=info target/debug/codex "Run \`echo deepseek-tool-ok\` and tell me what it printed."
```

Expected:

- DeepSeek streams a `tool_calls` delta for `exec_command`.
- Codex executes the command through the normal tool machinery.
- A follow-up `POST /chat/completions` carries the assistant `tool_calls`
  message (with `reasoning_content` when reasoning is present) and the
  `role: "tool"` result.
- The final reply repeats `deepseek-tool-ok`.

## Scenario 3 — reasoning round-trip (DeepSeek thinking mode)

Use a prompt that triggers reasoning (a non-trivial task):

```sh
RUST_LOG=info target/debug/codex "Compare bubble sort and quicksort complexity in two sentences."
```

Expected:

- Logs show `response.output_text.delta` / reasoning events streamed from the
  chat SSE (`delta.reasoning_content` → `ReasoningContentDelta`).
- If the model issues a tool call after reasoning, the follow-up request's
  assistant message includes `reasoning_content` anchored to the tool call.

## Debugging aids

- `RUST_LOG=trace` prints the raw SSE lines the Chat SSE parser receives.
- `~/.codex/sessions/rollout-*.jsonl` records the turn history; confirm the
  user message is stored as text and no image content is present.

Do not commit API keys or environment configuration derived from this test.
