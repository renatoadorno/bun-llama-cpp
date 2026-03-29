import type { LibLlama } from './ffi.ts'

// EOT text markers — covers Qwen3, Llama3, Mistral, Phi, etc.
const EOT_TEXTS = new Set([
  '<|im_end|>', '</s>', '<|endoftext|>', '<|eot_id|>', '<|end|>', '<|EOT|>',
])

/** Tokenize text into an Int32Array of token IDs. */
export function tokenize(L: LibLlama, vocabPtr: number, text: string): Int32Array {
  const textBuf = Buffer.from(text + '\0', 'utf8')
  const tokBuf = new Int32Array(textBuf.length)
  const n = L.llama_tokenize(
    vocabPtr, textBuf, textBuf.length - 1,
    tokBuf, tokBuf.length,
    true,  // add_special (BOS)
    true,  // parse_special — so <|im_start|>/<|im_end|> become real tokens
  )
  if (n < 0) throw new Error(`Tokenization failed (code ${n})`)
  return tokBuf.subarray(0, n)
}

/** Render a single token ID to its text representation. */
export function tokenPiece(L: LibLlama, vocabPtr: number, token: number): string {
  const buf = Buffer.alloc(64)
  const n = L.llama_token_to_piece(vocabPtr, token, buf, 64, 0, true)
  if (n <= 0) return ''
  return buf.subarray(0, n).toString('utf8')
}

/** Check if a token or its text marks end-of-generation. */
export function isEndOfGeneration(L: LibLlama, vocabPtr: number, token: number, piece: string): boolean {
  if (Boolean(L.llama_vocab_is_eog(vocabPtr, token))) return true
  return EOT_TEXTS.has(piece)
}

/** Check if a piece is a special control token (e.g. <|im_start|>). */
export function isSpecialToken(piece: string): boolean {
  return piece.startsWith('<|') && piece.endsWith('|>')
}
