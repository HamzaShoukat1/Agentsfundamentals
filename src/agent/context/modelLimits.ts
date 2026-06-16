import type { ModelLimits } from "../../types.ts";

/**
 * Default threshold for context window usage (80%)
 */
export const DEFAULT_THRESHOLD = 0.8;
const MODEL_NAME = "gemini-2.5-flash";
const MINI_MODEL_NAME = "gemini-2.0-mini";
/**
 * Model limits registry
 * Currently only includes GPT-5 models
 */
const MODEL_LIMITS: Record<string, ModelLimits> = {
  [MODEL_NAME]: {
    inputLimit: 272000,
    outputLimit: 128000,
    contextWindow: 400000,
  },
  [MINI_MODEL_NAME]: {
    inputLimit: 272000,
    outputLimit: 128000,
    contextWindow: 100000,
  },
};

/**
 * Default limits used when model is not found in registry
 */
const DEFAULT_LIMITS: ModelLimits = {
  inputLimit: 128000,
  outputLimit: 16000,
  contextWindow: 900000,
};

/**
 * Get token limits for a specific model.
 * Falls back to default limits if model not found.
 * Matches GPT-5 variants (gpt-5, gpt-5-mini, etc.)
 */
export function getModelLimits(model: string): ModelLimits {
  // Direct match
  if (MODEL_LIMITS[model]) {
    return MODEL_LIMITS[model];
  }

  // Check for gpt-5 variants
  if (model.startsWith(MODEL_NAME)) {
    return MODEL_LIMITS[MODEL_NAME];
  }

  return DEFAULT_LIMITS;
}

/**
 * Check if token usage exceeds the threshold
 */
export function isOverThreshold(
  totalTokens: number,
  contextWindow: number,
  threshold: number = DEFAULT_THRESHOLD,
): boolean {
  return totalTokens > contextWindow * threshold;
}

/**
 * Calculate usage percentage
 */
export function calculateUsagePercentage(
  totalTokens: number,
  contextWindow: number,
): number {
  return (totalTokens / contextWindow) * 100;
}
