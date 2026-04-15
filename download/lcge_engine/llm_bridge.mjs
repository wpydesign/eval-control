/**
 * llm_bridge.mjs — Node.js bridge for z-ai-web-dev-sdk
 *
 * This script is called by the Python LLM execution layer.
 * It reads a JSON input file, calls the z-ai-web-dev-sdk,
 * and outputs a JSON response to stdout.
 *
 * Usage: node llm_bridge.mjs <input_json_path>
 *
 * Input JSON: { "prompt": "...", "role": "primary|baseline", "temperature": 0.7 }
 * Output JSON: { "content": "...", "token_count": N, "finish_reason": "..." }
 *
 * v1.3: Added temperature parameter for stochastic sampling.
 */

import ZAI from 'z-ai-web-dev-sdk';
import { readFileSync } from 'fs';

const inputPath = process.argv[2];
if (!inputPath) {
  console.error('Usage: node llm_bridge.mjs <input_json_path>');
  process.exit(1);
}

async function main() {
  try {
    const input = JSON.parse(readFileSync(inputPath, 'utf-8'));
    const { prompt, role, temperature } = input;

    const zai = await ZAI.create();

    const completion = await zai.chat.completions.create({
      messages: [
        {
          role: 'system',
          content: 'You are a helpful assistant. Answer directly and concisely.'
        },
        {
          role: 'user',
          content: prompt
        }
      ],
      temperature: typeof temperature === 'number' ? temperature : undefined,
    });

    const choice = completion.choices?.[0];
    const content = choice?.message?.content || '';
    const finishReason = choice?.finish_reason || 'stop';

    // Estimate token count from content
    const tokenCount = content.split(/\s+/).filter(Boolean).length;

    const output = {
      content: content,
      token_count: tokenCount,
      finish_reason: finishReason
    };

    console.log(JSON.stringify(output));

  } catch (error) {
    console.error(JSON.stringify({
      content: `[BRIDGE ERROR] ${error.message}`,
      token_count: 0,
      finish_reason: 'error'
    }));
    process.exit(1);
  }
}

main();
