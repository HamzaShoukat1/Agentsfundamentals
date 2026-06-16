import { google } from "@ai-sdk/google"
import { GoogleGenAI } from "@google/genai";




// const ModelName = "gemini-2.5-flash";
// const ai = new GoogleGenAI({})

// const groundingTool  = {
//     googleSearch:{}
// }
// const config = {
//     tools:[groundingTool]
// };
// export const webSearch = await ai.models.generateContent({
//     model:ModelName,
//     contents:"What is the capital of France?",
//     config
// })

export const webSearch = google.tools.googleSearch({});
