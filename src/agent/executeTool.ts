import { tools } from "./tools/index.ts";

export type ToolName = keyof typeof tools;

export async function executeTool(
  name: string,
  args: Record<string, unknown>,
): Promise<string> {
  const tool = tools[name as ToolName];

  if (!tool) {
    return `Unknown tool: ${name}`;
  }

  // Type guard: check if tool has execute method
  if (
    !tool ||
    typeof tool !== "object" ||
    !("execute" in tool) ||
    typeof tool.execute !== "function"
  ) {
    return `Tool ${name} - executed by model provider`;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const result = await (tool.execute as any)(args, {
    toolCallId: "",
    messages: [],
  });

  return String(result);
}