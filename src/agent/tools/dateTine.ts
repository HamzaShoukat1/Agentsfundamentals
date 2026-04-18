import { tool } from "ai";
import { z } from "zod";

export const getDateTime = tool({
    description: "Get the current date and time.",
    inputSchema: z.object({}),
    execute: async () => {
        return new Date().toString();
    },
});
