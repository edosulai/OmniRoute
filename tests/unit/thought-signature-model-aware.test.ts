import { test } from "node:test";
import assert from "node:assert/strict";
import { openaiToAntigravityRequest } from "../../open-sse/translator/request/openai-to-gemini.ts";
import { geminiToOpenAIResponse } from "../../open-sse/translator/response/gemini-to-openai.ts";
import { initState } from "../../open-sse/translator/index.ts";
import { FORMATS } from "../../open-sse/translator/formats.ts";
import {
  buildGeminiThoughtSignatureNamespace,
  clearGeminiThoughtSignatureMemoryForTests,
  thoughtSignatureModelFamily,
} from "../../open-sse/services/geminiThoughtSignatureStore.ts";

// A thought signature only validates on the model that minted it. The cache was keyed
// by connection + tool call id only, so a chat that made a tool call on
// antigravity/claude-opus-5-5 and continued on gemini-3.8-flash on the same account
// replayed the Claude signature on a Gemini functionCall: 400 "Corrupted thought
// signature", on every turn, for the 30 days the signature stays persisted.

const CLAUDE_SIG = "claude#SIG";

function mintToolCall(namespace: string | null, model: string, signature: string) {
  // Same state shape stream.ts builds for an Antigravity → OpenAI stream.
  const state = { ...initState(FORMATS.OPENAI), signatureNamespace: namespace, toolNameMap: null };
  const chunks = geminiToOpenAIResponse(
    {
      response: {
        responseId: `resp-${model}`,
        modelVersion: model,
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                {
                  thoughtSignature: signature,
                  functionCall: { name: "terminal", args: { command: "ls" } },
                },
              ],
            },
          },
        ],
      },
    },
    state
  ) as Array<{ choices: Array<{ delta: { tool_calls?: Array<{ id: string }> } }> }>;
  const call = chunks.flatMap((c) => c.choices[0].delta.tool_calls ?? [])[0];
  assert.ok(call, "expected a streamed tool call");
  return call.id;
}

function replayedSignature(namespace: string | null, model: string, toolCallId: string) {
  const request = openaiToAntigravityRequest(
    model,
    {
      messages: [
        { role: "user", content: "list files" },
        {
          role: "assistant",
          content: null,
          tool_calls: [
            {
              id: toolCallId,
              type: "function",
              function: { name: "terminal", arguments: '{"command":"ls"}' },
            },
          ],
        },
        { role: "tool", tool_call_id: toolCallId, content: "a.txt" },
        { role: "user", content: "continue" },
      ],
    },
    true,
    namespace ? { _signatureNamespace: namespace } : {}
  ) as {
    request: {
      contents: Array<{ parts: Array<{ functionCall?: unknown; thoughtSignature?: string }> }>;
    };
  };
  const part = request.request.contents.flatMap((c) => c.parts).find((p) => p.functionCall);
  return part?.thoughtSignature;
}

test("model family drops provider prefix and reasoning tier", () => {
  assert.equal(
    thoughtSignatureModelFamily("antigravity/gemini-3.8-flash-tiered"),
    "gemini-3.8-flash"
  );
  assert.equal(thoughtSignatureModelFamily("gemini-3.8-flash-high"), "gemini-3.8-flash");
  assert.equal(thoughtSignatureModelFamily("claude-opus-5-5-high"), "claude-opus-5-5");
  assert.equal(thoughtSignatureModelFamily(""), null);
  assert.equal(buildGeminiThoughtSignatureNamespace("conn", null), "conn");
  assert.equal(buildGeminiThoughtSignatureNamespace(null, "gemini-3.8-flash"), null);
});

test("a Claude signature is never replayed on a Gemini model of the same connection", () => {
  const conn = "conn-model-aware-1";
  const id = mintToolCall(
    buildGeminiThoughtSignatureNamespace(conn, "claude-opus-5-5-high"),
    "claude-opus-5-5-high",
    CLAUDE_SIG
  );
  clearGeminiThoughtSignatureMemoryForTests(); // also after an OmniRoute restart (persisted copy)
  const sig = replayedSignature(
    buildGeminiThoughtSignatureNamespace(conn, "gemini-3.8-flash-tiered"),
    "gemini-3.8-flash-tiered",
    id
  );
  assert.notEqual(sig, CLAUDE_SIG);
});

test("the minting model and its other tiers still get the real signature back", () => {
  const conn = "conn-model-aware-2";
  const id = mintToolCall(
    buildGeminiThoughtSignatureNamespace(conn, "gemini-3.8-flash-tiered"),
    "gemini-3.8-flash-tiered",
    "EGEMINISIG"
  );
  for (const model of ["gemini-3.8-flash-tiered", "gemini-3.8-flash-high"]) {
    assert.equal(
      replayedSignature(buildGeminiThoughtSignatureNamespace(conn, model), model, id),
      "EGEMINISIG"
    );
  }
});
