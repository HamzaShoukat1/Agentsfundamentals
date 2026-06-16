import "dotenv/config";

import {
  generateText,
  stepCountIs,
  tool,
  type ModelMessage,
  type ToolSet,
} from "ai";

import { google } from "@ai-sdk/google";
import z from "zod";

import { SYSTEM_PROMPT } from "../src/agent/system/prompt.ts";
import type {
  EvalData,
  SingleTurnResult,
  MultiTurnEvalData,
  MultiTurnResult,
} from "./types.ts";

import { buildMessages, buildMockedTools } from "./utils.ts";

/**
 * ------------------------
 * TOOL DEFINITIONS
 * ------------------------
 */
const TOOL_DEFINATIONS: Record<string, any> = {
  readFile: {
    description: "Read the content of a file at specified path",
    parameters: z.object({
      path: z.string().describe("Path of the file to read"),
    }),
  },

  writeFile: {
    description: "Write content to a file at specified path",
    parameters: z.object({
      path: z.string().describe("Path of the file to write"),
      content: z.string().describe("Content to write to the file"),
    }),
  },

  listFiles: {
    description: "List all files in a directory",
    parameters: z.object({
      path: z.string().describe("Path of directory"),
    }),
  },

  deleteFile: {
    description: "Delete a file at specified path",
    parameters: z.object({
      path: z.string().describe("Path of file to delete"),
    }),
  },

  runCommand: {
    description: "Execute shell command",
    parameters: z.object({
      command: z.string().describe("Shell command to run"),
    }),
  },
};

/**
 * ------------------------
 * SINGLE TURN EVAL
 * ------------------------
 */
//Did model choose correct tool?
export async function singleTurnExecutor(
  data: EvalData,
): Promise<SingleTurnResult> {
  const messages = buildMessages(data);

  const tools: ToolSet = {};

  for (const toolName of data.tools) {
    const def = TOOL_DEFINATIONS[toolName]; // ✅ FIXED

    if (!def) continue;

    tools[toolName] = tool({
      description: def.description,
      inputSchema: def.parameters,
    });
  }

  const result = await generateText({
    model: google(data.config?.model ?? "gemini-2.5-flash"), // ✅ FIXED
    messages,
    tools,
    stopWhen: stepCountIs(1),
    temperature: data.config?.temperature ?? undefined,
  });

  const toolCalls = (result.toolCalls ?? []).map((tc) => ({
    toolName: tc.toolName,
    args: "args" in tc ? tc.args : {},
  }));

  return {
    toolCalls,
    toolNames: toolCalls.map((t) => t.toolName),
    selectedAny: toolCalls.length > 0,
  };
};

/**
 * 
 * ------------------------
 * MULTI TURN EVAL (MOCKS)
 * ------------------------
 */
//Can model finish the whole task?
export async function multiTurnWithMocks(
  data: MultiTurnEvalData,
): Promise<MultiTurnResult> {
  const tools = buildMockedTools(data.mockTools);

  const messages: ModelMessage[] =
    data.messages ?? [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: data.prompt! },
    ];

  const result = await generateText({
    model: google(data.config?.model ?? "gemini-2.5-flash"), // ✅ FIXED
    messages,
    tools,
    stopWhen: stepCountIs(data.config?.maxSteps ?? 20),
  });

  const allToolCalls: string[] = [];

  const steps = result.steps.map((step) => {
    const stepToolCalls = (step.toolCalls ?? []).map((tc) => {
      allToolCalls.push(tc.toolName);

      return {
        toolName: tc.toolName,
        args: "args" in tc ? tc.args : {},

      };
    });

    const stepToolResults = (step.toolResults ?? []).map((tr) => ({
      toolName: tr.toolName,
      result: "result" in tr ? tr.result : tr,
    }));

    return {
      toolCalls: stepToolCalls.length ? stepToolCalls : undefined,
      toolResults: stepToolResults.length ? stepToolResults : undefined,
      text: step.text || undefined,
    };
  });

  return {
    text: result.text,
    steps,
    toolsUsed: [...new Set(allToolCalls)],
    toolCallOrder: allToolCalls,
  };
};