import "dotenv/config";

import { streamText, type ModelMessage } from "ai";
import { google } from "@ai-sdk/google";
import type { AgentCallbacks, ToolCallInfo } from "../types.ts";
import {estimateMessagesTokens,getModelLimits,isOverThreshold,calculateUsagePercentage,DEFAULT_THRESHOLD,compactConversation} from "./context/index.ts"

import { SYSTEM_PROMPT } from "./system/prompt.ts";
import { executeTool } from "./executeTool.ts";

import { tools } from "./tools/index.ts";

import { getTracer, Laminar } from "@lmnr-ai/lmnr";
import { filterCompatibleMessages } from "./system/filterMessages.ts";

// =========================
// MODEL
// =========================
const MODEL_INSTANCE = google("gemini-2.5-flash");

// =========================
// Laminar Init (SAFE)
// =========================
const lmnrKey = process.env.LMNR_PROJECT_API_KEY;

if (lmnrKey) {
  try {
    Laminar.initialize({
      projectApiKey: lmnrKey,
    });
  } catch (error) {
    console.warn("Failed to initialize Laminar:", error);
  }
}

// =========================
// MAIN AGENT
// =========================
export async function runAgent(
  userMessage: string,
  conversationHistory: ModelMessage[],
  callbacks: AgentCallbacks,
): Promise<ModelMessage[]> {
  const modelLimits = getModelLimits("gemini-2.5-flash");

  const workingHistory = filterCompatibleMessages(conversationHistory);

  let  messages: ModelMessage[] = [
    ...workingHistory,
    { role: "user", content: userMessage },
  ];

  const precheckMessagesTokens = estimateMessagesTokens(messages);

  if(isOverThreshold(precheckMessagesTokens.total,modelLimits.contextWindow)){

    messages = await compactConversation(workingHistory,MODEL_INSTANCE);
    

  }

  let FullResponse = "";

  while (true) {
    //  IMPORTANT: streamText must be awaited stream object usage only
    const result = streamText({
      model: MODEL_INSTANCE,
      system: SYSTEM_PROMPT,
      messages,
      tools,

      temperature: 0.3,

      experimental_telemetry: {
        isEnabled: true,
        tracer: getTracer(),
      },
    });
    const reportTokensUsage = ()=> {
      if(callbacks.onTokenUsage){
        const usage = precheckMessagesTokens
        callbacks.onTokenUsage({
          inputTokens: usage.input,
          outputTokens: usage.output,
          totalTokens: usage.total,
          contextWindow:modelLimits.contextWindow,
          threshold: DEFAULT_THRESHOLD,
          percentage:calculateUsagePercentage(usage.total,modelLimits.contextWindow)
        })
      }
    }

    console.log("TOOLS SENT TO MODEL:", Object.keys(tools));

    const toolCalls: ToolCallInfo[] = [];
    let currentText = "";
    let streamError: Error | null = null;

    try {
      for await (const chunk of result.fullStream) {
        if (chunk.type === "text-delta") {
          currentText += chunk.text;
          callbacks.onToken?.(chunk.text);
        }

        if (chunk.type === "tool-call") {
          const input = "input" in chunk ? chunk.input : {};

          toolCalls.push({
            toolCallId: chunk.toolCallId,
            toolName: chunk.toolName,
            args: input as any,
          });

          callbacks.onToolCallStart?.(chunk.toolName, input);
        }
      }
    } catch (error) {
      streamError = error as Error;
      console.error("Stream error:", streamError);

      // allow graceful failure
      if (!currentText) break;
    }

    FullResponse += currentText;

    // =========================
    // FAIL SAFE
    // =========================
    if (streamError && !currentText) {
      const msg = "Sorry, the agent failed to generate a response.";
      callbacks.onToken?.(msg);
      return [
        { role: "system", content: SYSTEM_PROMPT },
        ...messages,
        { role: "assistant", content: msg },
      ];
    }

    // =========================
    // CHECK FINISH
    // =========================
    const finishReason = await result.finishReason;

    console.log("Finish reason:", finishReason);
    console.log("Tool calls:", toolCalls.length);

    // Add assistant message first
    const responseMessage = await result.response;
    messages.push(...responseMessage.messages);
    reportTokensUsage()

    // If no tools → exit loop
    if (finishReason !== "tool-calls" || toolCalls.length === 0) {
      break;
    }

    // =========================
    // EXECUTE TOOLS
    // =========================
    for (const call of toolCalls) {
      const toolResult = await executeTool(call.toolName, call.args);

      callbacks.onToolCallEnd?.(call.toolName, toolResult);

      messages.push({
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: call.toolCallId,
            toolName: call.toolName,
            output: { type: "text", value: toolResult },
          },
        ],
      } as any);
      reportTokensUsage()
    }

    callbacks.onComplete?.(FullResponse);
  };

  return messages
}