import "dotenv/config";

import { GoogleGenAI, type Content, type Tool } from "@google/genai";
import type { ModelMessage } from "ai";
import { fileURLToPath } from "node:url";
import path from "node:path";
import type { AgentCallbacks } from "../types.ts";
import { SYSTEM_PROMPT } from "./system/prompt.ts";
import { extractMessageText } from "./context/tokenEstimator.ts";
import { executeTool } from "./executeTool.ts";
import { tools } from "./tools/index.ts";

const MODEL_NAME = "gemini-2.5-flash";

const apiKey = process.env.GEMINI_API_KEY;

if (!apiKey) {
  throw new Error("GEMINI_API_KEY is not set.");
}
const ai = new GoogleGenAI({ apiKey });

/**
 * Convert AI SDK tools to Gemini function declarations
 */
function buildGeminiTools(): Tool[] {
  const geminiTools: Tool[] = [];

  for (const [toolName, toolDef] of Object.entries(tools)) {
    // Type guard: check if it's a Tool object with required properties
    if (
      !toolDef ||
      typeof toolDef !== "object" ||
      !("description" in toolDef) ||
      !("inputSchema" in toolDef)
    ) {
      continue;
    }

    const desc = toolDef.description as string;
    const zodSchema = toolDef.inputSchema as any;

    // Extract properties from Zod schema
    const properties: Record<string, any> = {};
    const required: string[] = [];

    if (zodSchema._def?.shape) {
      // For z.object()
      for (const [key, fieldSchema] of Object.entries(
        zodSchema._def.shape,
      )) {
        const field = fieldSchema as any;
        let fieldType = "string";
        let fieldDesc = "";

        // Infer type from Zod
        if (field._def?.typeName === "ZodString") {
          fieldType = "string";
        } else if (field._def?.typeName === "ZodNumber") {
          fieldType = "number";
        } else if (field._def?.typeName === "ZodBoolean") {
          fieldType = "boolean";
        }

        // Get description
        if (field.description) {
          fieldDesc = field.description;
        }

        properties[key] = {
          type: fieldType,
          description: fieldDesc,
        };

        // All fields are required unless explicitly optional
        required.push(key);
      }
    }

    geminiTools.push({
      functionDeclarations: [
        {
          name: toolName,
          description: desc,
          parameters: {
            type: "OBJECT" as any,
            properties,
            required,
          },
        },
      ],
    });
  }

  return geminiTools;
}

/**
 * Convert AI SDK message format → Gemini format
 */
function toGeminiContent(message: ModelMessage): Content | null {
  if (message.role !== "user" && message.role !== "assistant") {
    return null;
  }

  const text = extractMessageText(message).trim();
  if (!text) return null;

  return {
    role: message.role === "assistant" ? "model" : "user",
    parts: [{ text }],
  };
}

/**
 * MAIN AGENT LOOP
 */
export async function runAgent(
  userMessage: string,
  conversationHistory: ModelMessage[],
  callbacks: AgentCallbacks,
): Promise<ModelMessage[]> {
  // Build conversation
  const contents: Content[] = conversationHistory
    .map(toGeminiContent)
    .filter((m): m is Content => m !== null);

  contents.push({
    role: "user",
    parts: [{ text: userMessage }],
  });

  const geminiTools = buildGeminiTools();
  let responseText = "";

  // Agent loop: keep calling model until no more tool calls
  while (true) {
    // CALL MODEL
    const response = await ai.models.generateContent({
      model: MODEL_NAME,
      contents,
      config: {
        systemInstruction: SYSTEM_PROMPT,
        tools: geminiTools.length > 0 ? geminiTools : undefined,
      },
    });

    // Get text response
    const modelText = response.text?.trim() ?? "";
    if (modelText) {
      responseText = modelText;
      callbacks?.onToken?.(modelText);
    }

    // CHECK FOR FUNCTION CALLS
    const functionCalls = response.functionCalls;

    if (!functionCalls || functionCalls.length === 0) {
      // No tool calls - done
      break;
    }

    // Add model's response to history
    contents.push({
      role: "model",
      parts: [{ text: modelText || "Processing tool calls..." }],
    });

    // EXECUTE TOOLS
    const toolResults: Content[] = [];

    for (const call of functionCalls) {
      const toolName = call.name ?? "";
      if (!toolName) {
        continue;
      }

      const toolArgs = (call.args as Record<string, unknown>) ?? {};

      // Callbacks
      callbacks?.onToolCallStart?.(toolName, toolArgs);

      // Execute
      const result = await executeTool(toolName, toolArgs);
      callbacks?.onToolCallEnd?.(toolName, result);

      // Add tool result for Gemini
      toolResults.push({
        role: "user",
        parts: [
          {
            functionResponse: {
              name: toolName,
              response: {
                result: result,
              },
            },
          } as any,
        ],
      });
    }

    // Add all tool results to contents
    contents.push(...toolResults);
  }

  // Callbacks
  callbacks?.onComplete?.(responseText);

  // Return updated conversation
  return [
    ...conversationHistory,
    {
      role: "user",
      content: userMessage,
    },
    {
      role: "assistant",
      content: responseText || "Tool execution completed.",
    },
  ];
}

/**
 * TEST: Run agent directly in terminal
 */
async function testAgent() {
  console.log("\n🤖 Testing Agent...\n");

  try {
    const result = await runAgent(
      "What is the current date and time?",
      [],
      {
        onToken: (token) => {
          process.stdout.write(token);
        },
        onComplete: () => {
          console.log("\n\n✅ Agent completed!\n");
        },
        onToolCallStart: (toolName, args) => {
          console.log(`\n🔧 [TOOL START] ${toolName}`);
          console.log(`   Args: ${JSON.stringify(args)}`);
        },
        onToolCallEnd: (toolName, result) => {
          console.log(`\n✅ [TOOL END] ${toolName}`);
          console.log(`   Result: ${result}\n`);
        },
        onToolApproval: async (toolName, args) => {
          console.log(`\n❓ [APPROVAL] Use ${toolName}? (auto-approving)`);
          return true;
        },
        onTokenUsage: (usage) => {
          console.log(`\n📊 Tokens - Input: ${usage.inputTokens}, Output: ${usage.outputTokens}`);
        },
      },
    );

    console.log("\n📝 Final conversation:\n");
    result.forEach((msg, i) => {
      console.log(`[${i}] ${msg.role.toUpperCase()}: ${msg.content}`);
    });
  } catch (error) {
    console.error("\n❌ Error:", error);
    process.exit(1);
  }
}

// Run test if this file is executed directly
const isRunDirectly =
  process.argv[1] !== undefined &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (isRunDirectly) {
  void testAgent();
}