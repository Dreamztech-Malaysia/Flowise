/**
 * Self-hosted customer memory helpers for the Mem0 node (Dreamztech fork).
 *
 * Pure helpers (config parsing, formatting, redaction, backfill dedupe) live at the top and are unit tested.
 * The mem0ai OSS `Memory` instance factory lives at the bottom and is loaded lazily so the node pool
 * never pulls mem0ai (and its native deps) in unless a self-hosted flow actually runs.
 *
 * NOTE: this file has no `nodeClass` export on purpose; NodesPool skips it.
 */
import { createHash } from 'crypto'
import { AsyncLocalStorage } from 'async_hooks'

export const CUSTOMER_MEMORY_DEFAULT_COLLECTION = 'customer_memory'
export const CUSTOMER_MEMORY_DEFAULT_BASE_URL = 'https://api.openai.com/v1'
export const CUSTOMER_MEMORY_DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
export const CUSTOMER_MEMORY_DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'
export const CUSTOMER_MEMORY_DEFAULT_EMBEDDING_DIMS = 1536
export const CUSTOMER_MEMORY_FACTS_HEADER =
    'Known facts about this customer from earlier conversations (they may be outdated or wrong; confirm before relying on them):'

/** Shape the caller sends as `overrideConfig.customerMemory`. */
export interface CustomerMemoryRequest {
    enabled?: boolean | string
    userId?: string
    ownerId?: number | string
    salesmanId?: number | string
    apiKey?: string
    baseUrl?: string
    llmModel?: string
    embeddingModel?: string
    backfill?: boolean | string
}

/** Normalised config. `null` from the parser means "memory off, behave as plain buffer memory". */
export interface CustomerMemoryConfig {
    userId: string
    ownerId?: number
    salesmanId?: string
    apiKey: string
    baseUrl: string
    llmModel: string
    embeddingModel: string
    backfill: boolean
}

const toBool = (v: unknown): boolean => v === true || v === 'true' || v === 1 || v === '1'

/**
 * Flowise resolves node inputs through its variable resolver, which stringifies object values, so the
 * override can arrive as an object or as a JSON string. Anything invalid => null (memory off).
 */
export const parseCustomerMemoryConfig = (raw: unknown): CustomerMemoryConfig | null => {
    if (raw === undefined || raw === null || raw === '') return null
    let obj: CustomerMemoryRequest
    if (typeof raw === 'string') {
        try {
            obj = JSON.parse(raw)
        } catch {
            return null
        }
    } else if (typeof raw === 'object' && !Array.isArray(raw)) {
        obj = raw as CustomerMemoryRequest
    } else {
        return null
    }
    if (!obj || typeof obj !== 'object' || !toBool(obj.enabled)) return null

    const userId = typeof obj.userId === 'string' ? obj.userId.trim() : ''
    const apiKey = typeof obj.apiKey === 'string' ? obj.apiKey.trim() : ''
    // mem0 rejects ids containing whitespace; treat as "memory off" rather than failing the reply.
    if (!userId || !apiKey || userId.length > 256 || /\s/.test(userId)) return null

    let ownerId: number | undefined
    if (obj.ownerId !== undefined && obj.ownerId !== null && obj.ownerId !== '') {
        const n = Number(obj.ownerId)
        if (Number.isFinite(n)) ownerId = n
    }
    let salesmanId: string | undefined
    if ((typeof obj.salesmanId === 'string' || typeof obj.salesmanId === 'number') && String(obj.salesmanId).trim()) {
        salesmanId = String(obj.salesmanId).trim().slice(0, 128)
    }

    const str = (v: unknown, d: string) => (typeof v === 'string' && v.trim() ? v.trim() : d)
    return {
        userId,
        ownerId,
        ...(salesmanId !== undefined ? { salesmanId } : {}),
        apiKey,
        baseUrl: str(obj.baseUrl, CUSTOMER_MEMORY_DEFAULT_BASE_URL).replace(/\/+$/, ''),
        llmModel: str(obj.llmModel, CUSTOMER_MEMORY_DEFAULT_LLM_MODEL),
        embeddingModel: str(obj.embeddingModel, CUSTOMER_MEMORY_DEFAULT_EMBEDDING_MODEL),
        backfill: toBool(obj.backfill)
    }
}

/** Render recalled facts as one prompt block; empty string when there is nothing to inject. */
export const formatFacts = (facts: { memory?: string }[], limit: number): string => {
    const lines = facts
        .map((f) => (typeof f?.memory === 'string' ? f.memory.trim() : ''))
        .filter(Boolean)
        .slice(0, Math.max(0, limit))
    if (!lines.length) return ''
    return [CUSTOMER_MEMORY_FACTS_HEADER, ...lines.map((l) => `- ${l}`)].join('\n')
}

/** Strip anything key-like out of provider error messages before logging. */
export const redactSecrets = (message: unknown): string => {
    const s = message instanceof Error ? message.message : String(message ?? '')
    return s
        .replace(/(sk|pk|rk|gsk|xai)-[A-Za-z0-9_\-*]{6,}/g, '$1-***')
        .replace(/(Bearer\s+)[A-Za-z0-9._\-]+/gi, '$1***')
        .replace(/(api[_-]?key["'\s:=]+)[^\s"',}]+/gi, '$1***')
}

export interface Mem0ChatMessage {
    role: 'user' | 'assistant'
    content: string
}

/** Map Flowise chat_message rows (oldest first) to mem0 messages, dropping the turn currently being saved. */
export const buildBackfillMessages = (rows: { role: string; content: string }[], current: Mem0ChatMessage[]): Mem0ChatMessage[] => {
    const mapped: Mem0ChatMessage[] = rows
        .filter((r) => typeof r?.content === 'string' && r.content.trim())
        .map((r) => ({ role: r.role === 'userMessage' ? 'user' : 'assistant', content: r.content }))
    // The server may already have persisted all or part of the current exchange (user row first, then the
    // reply). Drop the longest tail of `mapped` that equals a prefix of `current` so nothing is fed twice.
    const same = (a: Mem0ChatMessage, b: Mem0ChatMessage) => a.role === b.role && a.content === b.content
    for (let k = Math.min(current.length, mapped.length); k > 0; k--) {
        const tail = mapped.slice(mapped.length - k)
        if (tail.every((m, i) => same(m, current[i]))) return mapped.slice(0, mapped.length - k)
    }
    return mapped
}

// ---------------------------------------------------------------------------------------------------------
// Category + label tagging (piggybacks on mem0's own extraction LLM call; no extra request)
// ---------------------------------------------------------------------------------------------------------

export const FACT_CATEGORIES = ['interest', 'objection', 'budget', 'location', 'timing', 'family', 'other'] as const
export type FactCategory = typeof FACT_CATEGORIES[number]
export interface FactTag {
    category: FactCategory
    label: string
}

/**
 * Appended to mem0's additive extraction prompt via `customInstructions`. mem0 3.x drops unknown keys when it
 * zod-parses the response, so the raw response is captured (see `extractionCapture`) and parsed here instead.
 * Cost: ~120 extra prompt tokens + ~10 output tokens per fact, same request.
 */
export const CUSTOMER_MEMORY_TAG_INSTRUCTIONS =
    'Only facts the customer stated about themselves; each "text" max 8 words, no subject, no dates ' +
    '(e.g. "Budget RM500", "Has 2 kids", "Lives in Johor Bahru"). ' +
    'For every item in "memory" also output "category" (exactly one of: interest, objection, budget, location, timing, family, other) ' +
    'and "label" (a short normalized 1-4 word Title-case phrase, e.g. "Family Package", "Price Too High", "Budget RM50k"). Keep "text" unchanged. ' +
    'If "memory" is non-empty, also output a top-level "summary": an "at a glance" description of the customer in 1-2 sentences (max 40 words), ' +
    'rewriting the Current Customer Summary to include the new facts. Omit "summary" when "memory" is empty.'

/**
 * Appended to mem0's extraction SYSTEM prompt (mem0's default extracts from both roles, writes long sentences and
 * stamps dates). Sales use needs a short list of what the customer said about themselves.
 */
export const CUSTOMER_MEMORY_FACT_RULES = [
    '# CUSTOMER MEMORY OVERRIDES (these rules take precedence over everything above)',
    'The "user" is a customer chatting with a sales assistant. Memories are read by the salesperson and the assistant later.',
    '- Extract ONLY facts the customer stated about themselves in "user" messages: needs, budget, family, location, availability/timing, preferences, objections, decisions.',
    '- "assistant" messages are context only. NEVER extract what the assistant offered, recommended, quoted, answered or promised. attributed_to is always "user".',
    '- Each "text" is ONE short fact, max 8 words, without a subject: "Budget RM500", "Has 2 kids", "Lives in Johor Bahru", "Finds Premium package too expensive", "Prefers contact after 6pm".',
    '- NO dates or time stamps (never "as of October 2026", never the observation/current date). Keep a date only if the customer gave it as part of the fact ("Moving house in December").',
    '- No duplicates: skip anything already in Existing Memories or already in this output. Skip greetings, plain questions and guesses.',
    '- If the customer stated nothing new about themselves, return {"memory": []}.'
].join('\n')

const MONTH =
    '(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)'
const DATE = `(?:\\d{4}-\\d{2}-\\d{2}|(?:early |mid[- ]|late )?${MONTH}\\.?(?: \\d{1,2}(?:st|nd|rd|th)?,?)? \\d{4}|\\d{1,2}(?:st|nd|rd|th)? ${MONTH}\\.?,? \\d{4})`
/** "as of October 2026", "as mentioned on 2026-10-23", "(2026-10-23)" -- stamps, not facts */
const DATE_STAMP_RE = new RegExp(
    `[,(]?\\s*\\b(?:as of|as at|as mentioned (?:on|in)|mentioned (?:on|in)|stated (?:on|in)|noted (?:on|in)|reported (?:on|in))\\s+${DATE}\\)?|\\(\\s*${DATE}\\s*\\)`,
    'gi'
)
const SUBJECT_RE = /^(?:the\s+)?(?:user|customer|client|lead)(?:'s|’s)?\s+/i
const ASSISTANT_FACT_RE =
    /^(?:was|were|has been|have been)\s+(?:recommended|offered|told|informed|advised|quoted|sent|given)\b|\b(?:assistant|ai|agent|salesman|bot)\s+(?:recommended|offered|suggested|said|quoted|promised|explained)\b/i

/** Short, subject-less, date-free fact text. '' when nothing is left. */
export const cleanFactText = (text: unknown): string => {
    if (typeof text !== 'string') return ''
    let s = text.replace(/\s+/g, ' ').trim()
    s = s
        .replace(DATE_STAMP_RE, '')
        .replace(/\s+([,.;])/g, '$1')
        .replace(/\s{2,}/g, ' ')
        .trim()
    const stripped = s.replace(SUBJECT_RE, '')
    if (stripped && stripped !== s) {
        // "User is ..." -> "Is ..." reads badly; drop a leading copula too
        s = stripped.replace(/^(?:is|are)\s+/i, '')
    }
    s = s.replace(/[\s.;,]+$/g, '').trim()
    return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''
}

/**
 * Enforce the fact rules on mem0's raw extraction JSON before mem0 parses it: drop assistant-attributed and
 * assistant-offer facts, clean each text, drop duplicates. Extra keys (category/label/summary) are kept.
 * Returns the input unchanged when it is not parseable JSON.
 */
export const sanitizeExtraction = (raw: unknown): unknown => {
    if (typeof raw !== 'string' || !raw.trim()) return raw
    let json: any
    try {
        const cleaned = raw.replace(/```(?:json)?/gi, '').trim()
        const start = cleaned.indexOf('{')
        const end = cleaned.lastIndexOf('}')
        if (start < 0 || end <= start) return raw
        json = JSON.parse(cleaned.slice(start, end + 1))
    } catch {
        return raw
    }
    if (!json || !Array.isArray(json.memory)) return raw
    const seen = new Set<string>()
    const memory: any[] = []
    for (const m of json.memory) {
        if (!m || typeof m !== 'object') continue
        if (typeof m.attributed_to === 'string' && m.attributed_to.toLowerCase() === 'assistant') continue
        const text = cleanFactText(m.text)
        if (!text || ASSISTANT_FACT_RE.test(text)) continue
        const key = normText(text)
        if (seen.has(key)) continue
        seen.add(key)
        memory.push({ ...m, text, attributed_to: 'user' })
    }
    return JSON.stringify({ ...json, memory })
}

/** Append the fact rules to the system message of an extraction request (copy; input untouched). */
export const injectFactRules = (messages: any[]): any[] => {
    if (!Array.isArray(messages) || !messages.length) return messages
    const idx = messages.findIndex((m) => m?.role === 'system')
    if (idx < 0 || typeof messages[idx]?.content !== 'string' || messages[idx].content.includes(CUSTOMER_MEMORY_FACT_RULES)) return messages
    const copy = messages.slice()
    copy[idx] = { ...messages[idx], content: `${messages[idx].content}\n\n${CUSTOMER_MEMORY_FACT_RULES}` }
    return copy
}

export const normalizeCategory = (v: unknown): FactCategory | null => {
    const s = typeof v === 'string' ? v.trim().toLowerCase() : ''
    return (FACT_CATEGORIES as readonly string[]).includes(s) ? (s as FactCategory) : null
}

export const normalizeLabel = (v: unknown): string | null => {
    if (typeof v !== 'string') return null
    const s = v
        .replace(/\s+/g, ' ')
        .replace(/^[\s"'.:-]+|[\s"'.:-]+$/g, '')
        .trim()
    if (!s) return null
    const short = s.length > 48 ? s.slice(0, 48).trim() : s
    return short.charAt(0).toUpperCase() + short.slice(1)
}

/** Cheap keyword fallback when the LLM omitted/garbled category or label. Order matters (first hit wins). */
const CATEGORY_RULES: { category: FactCategory; label: string; re: RegExp }[] = [
    { category: 'objection', label: 'Price Too High', re: /\b(too (expensive|pricey|costly)|price (is )?too high|can'?t afford|mahal)\b/i },
    { category: 'objection', label: 'Not Interested', re: /\b(not interested|no longer interested|tak (nak|mahu)|declined?)\b/i },
    { category: 'objection', label: 'Has Concerns', re: /\b(concern(ed)?|worried|hesitant|doubt|complain(ed|t)?|unhappy|objection)\b/i },
    {
        category: 'budget',
        label: 'Budget',
        re: /\b(budget|afford|rm ?\d|\$ ?\d|usd|myr|price range|loan|financing|installment|down ?payment)\b/i
    },
    {
        category: 'family',
        label: 'Family',
        re: /\b(wife|husband|spouse|kids?|children|child|son|daughter|family|parents?|married|baby)\b/i
    },
    {
        category: 'timing',
        label: 'Timing',
        re: /\b(next (week|month|year)|this (week|month|year)|tomorrow|today|soon|asap|urgent|by (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*|deadline|\d{4}-\d{2}-\d{2}|q[1-4])\b/i
    },
    {
        category: 'location',
        label: 'Location',
        re: /\b(lives? in|located|based in|area|city|town|near|address|kuala lumpur|kl|selangor|penang|johor|singapore)\b/i
    },
    {
        category: 'interest',
        label: 'Interest',
        re: /\b(interested|wants?|looking for|prefers?|likes?|asked about|enquir|inquir|considering|keen)\b/i
    }
]

export const ruleTag = (text: string): FactTag => {
    for (const r of CATEGORY_RULES) if (r.re.test(text)) return { category: r.category, label: r.label }
    return { category: 'other', label: 'Other' }
}

const normText = (t: string) => t.trim().replace(/\s+/g, ' ').toLowerCase()

export interface ExtractionExtras {
    tags: Map<string, Partial<FactTag>>
    summary: string | null
}

/** Parse the raw mem0 extraction response for our extra per-fact keys. Never throws. */
export const parseExtractionExtras = (raw: unknown): ExtractionExtras => {
    const out: ExtractionExtras = { tags: new Map(), summary: null }
    if (typeof raw !== 'string' || !raw.trim()) return out
    let json: any
    try {
        const cleaned = raw.replace(/```(?:json)?/gi, '').trim()
        const start = cleaned.indexOf('{')
        const end = cleaned.lastIndexOf('}')
        if (start < 0 || end <= start) return out
        json = JSON.parse(cleaned.slice(start, end + 1))
    } catch {
        return out
    }
    for (const m of Array.isArray(json?.memory) ? json.memory : []) {
        if (typeof m?.text !== 'string' || !m.text.trim()) continue
        const tag: Partial<FactTag> = {}
        const c = normalizeCategory(m.category)
        const l = normalizeLabel(m.label)
        if (c) tag.category = c
        if (l) tag.label = l
        out.tags.set(normText(m.text), tag)
    }
    if (typeof json?.summary === 'string' && json.summary.trim()) out.summary = json.summary.replace(/\s+/g, ' ').trim().slice(0, 600)
    return out
}

/** LLM tag when valid, keyword rules for whatever is missing. */
export const resolveTag = (text: string, extras: ExtractionExtras): FactTag => {
    const llm = extras.tags.get(normText(text)) ?? {}
    const rule = ruleTag(text)
    const category = llm.category ?? rule.category
    // a rule label only makes sense for the rule's own category
    const label = llm.label ?? (category === rule.category ? rule.label : normalizeLabel(category) ?? 'Other')
    return { category, label }
}

/** Raw extraction response captured per add() call (AsyncLocalStorage, so concurrent customers never mix). */
export interface ExtractionCapture {
    raw?: string
    /** Injected into the extraction prompt so the model can rewrite (not re-derive) the summary. */
    previousSummary?: string | null
    /** Token usage of every provider call made inside the capture (LLM + embeddings). */
    usage?: TokenUsage
}

export interface TokenUsage {
    llmCalls: number
    promptTokens: number
    completionTokens: number
    embeddingCalls: number
    embeddingTokens: number
}

export const emptyUsage = (): TokenUsage => ({ llmCalls: 0, promptTokens: 0, completionTokens: 0, embeddingCalls: 0, embeddingTokens: 0 })

/** mem0's extraction call is the one whose system prompt is the additive extractor. */
const isExtractionCall = (messages: any[]): boolean =>
    Array.isArray(messages) &&
    messages.some((m) => m?.role === 'system' && typeof m.content === 'string' && /Memory Extractor/.test(m.content))

export const SUMMARY_PROMPT_HEADER = '## Current Customer Summary'

/** Append the stored summary to the last user message of the extraction request (mutates a copy). */
export const injectSummary = (messages: any[], previousSummary: string | null | undefined): any[] => {
    if (!Array.isArray(messages) || !messages.length) return messages
    const idx = messages.map((m) => m?.role).lastIndexOf('user')
    if (idx < 0 || typeof messages[idx]?.content !== 'string') return messages
    const copy = messages.slice()
    copy[idx] = {
        ...messages[idx],
        content: `${messages[idx].content}\n\n${SUMMARY_PROMPT_HEADER}\n${previousSummary?.trim() || '(none yet)'}`
    }
    return copy
}

/** Deterministic UUID for the per-customer summary point (same derivation as the server). */
export const summaryPointId = (userId: string): string => {
    const h = createHash('sha256').update(`customer-summary:${userId}`).digest('hex')
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20, 32)}`
}

export const getSummaryCollectionName = (): string => `${getCollectionName()}_summaries`
export const extractionCapture = new AsyncLocalStorage<ExtractionCapture>()

/** Wrap mem0's LLM so the raw JSON of the extraction call is visible to us. Idempotent. */
export const wrapLlmForCapture = (memory: any) => {
    const llm = memory?.llm
    if (!llm || typeof llm.generateResponse !== 'function' || llm.__customerMemoryWrapped) return
    const orig = llm.generateResponse.bind(llm)
    llm.generateResponse = async (...args: any[]) => {
        const store = extractionCapture.getStore()
        const extraction = isExtractionCall(args[0])
        if (extraction) args[0] = injectFactRules(args[0])
        if (store && 'previousSummary' in store) args[0] = injectSummary(args[0], store.previousSummary)
        let res = await orig(...args)
        if (extraction && typeof res === 'string') res = sanitizeExtraction(res)
        if (store && typeof res === 'string') store.raw = res
        return res
    }
    llm.__customerMemoryWrapped = true
    wrapUsage(memory)
}

const addUsage = (kind: 'llm' | 'embedding', usage: any) => {
    const u = extractionCapture.getStore()?.usage
    if (!u || !usage) return
    if (kind === 'llm') {
        u.llmCalls++
        u.promptTokens += Number(usage.prompt_tokens) || 0
        u.completionTokens += Number(usage.completion_tokens) || 0
    } else {
        u.embeddingCalls++
        u.embeddingTokens += Number(usage.prompt_tokens ?? usage.total_tokens) || 0
    }
}

/** Count tokens of the OpenAI-compatible clients mem0 uses (only inside an extractionCapture with `usage`). */
const wrapUsage = (memory: any) => {
    const wrap = (target: any, kind: 'llm' | 'embedding') => {
        if (!target || typeof target.create !== 'function' || target.__customerMemoryUsage) return
        const orig = target.create.bind(target)
        target.create = async (...args: any[]) => {
            const res = await orig(...args)
            addUsage(kind, res?.usage)
            return res
        }
        target.__customerMemoryUsage = true
    }
    wrap(memory?.llm?.openai?.chat?.completions, 'llm')
    wrap(memory?.embedder?.openai?.embeddings, 'embedding')
}

export const getCollectionName = (): string => process.env.CUSTOMER_MEMORY_COLLECTION || CUSTOMER_MEMORY_DEFAULT_COLLECTION

/** Embedding models that accept OpenAI's `dimensions` param (OpenAI v3, Gemini via its OpenAI-compatible API). */
export const supportsDimensions = (model: string): boolean => /^(text-embedding-3|gemini-embedding)/.test(model)

export const getEmbeddingDims = (): number => {
    const n = Number(process.env.CUSTOMER_MEMORY_EMBEDDING_DIMS)
    return Number.isFinite(n) && n > 0 ? n : CUSTOMER_MEMORY_DEFAULT_EMBEDDING_DIMS
}

/** Run a promise with a timeout so recall can never hang a reply. */
export const withTimeout = <T>(p: Promise<T>, ms: number, label: string): Promise<T> =>
    new Promise<T>((resolve, reject) => {
        const t = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
        p.then(
            (v) => {
                clearTimeout(t)
                resolve(v)
            },
            (e) => {
                clearTimeout(t)
                reject(e)
            }
        )
    })

// ---------------------------------------------------------------------------------------------------------
// mem0ai OSS Memory factory (lazy)
// ---------------------------------------------------------------------------------------------------------

export interface Mem0Like {
    add(messages: Mem0ChatMessage[], config: Record<string, any>): Promise<any>
    search(query: string, config: Record<string, any>): Promise<{ results: any[] }>
    getAll(config: Record<string, any>): Promise<{ results: any[] }>
    /** Merge payload keys into existing points (category/label after extraction). Optional for test doubles. */
    setPayload?(ids: string[], payload: Record<string, any>): Promise<void>
    getSummary?(userId: string): Promise<string | null>
    saveSummary?(userId: string, summary: string, payload: Record<string, any>): Promise<void>
}

const MAX_CACHED_INSTANCES = 200
const instanceCache = new Map<string, Mem0Like>()

const cacheKey = (cfg: CustomerMemoryConfig) =>
    createHash('sha256')
        .update([cfg.apiKey, cfg.baseUrl, cfg.llmModel, cfg.embeddingModel, getCollectionName()].join('\u0000'))
        .digest('hex')

/**
 * One Memory per (customer key, baseUrl, models). The key only lives in memory and is hashed for the cache key.
 * Qdrant connection comes from QDRANT_URL / QDRANT_API_KEY; no company LLM/embedding key is ever used.
 */
export const getMem0Instance = async (cfg: CustomerMemoryConfig): Promise<Mem0Like> => {
    const key = cacheKey(cfg)
    const cached = instanceCache.get(key)
    if (cached) {
        // refresh LRU order
        instanceCache.delete(key)
        instanceCache.set(key, cached)
        return cached
    }

    const qdrantUrl = process.env.QDRANT_URL
    if (!qdrantUrl) throw new Error('QDRANT_URL is not set; self-hosted customer memory is unavailable')

    // mem0 reads MEM0_TELEMETRY at module load; default it off so customer data/usage is not reported.
    if (process.env.MEM0_TELEMETRY === undefined) process.env.MEM0_TELEMETRY = 'false'
    const { Memory } = await import('mem0ai-oss/oss')
    const { QdrantClient } = await import('@qdrant/js-client-rest')

    const dims = getEmbeddingDims()
    const client = new QdrantClient({ url: qdrantUrl, apiKey: process.env.QDRANT_API_KEY || undefined })
    const embedderConfig: Record<string, any> = { apiKey: cfg.apiKey, baseURL: cfg.baseUrl, model: cfg.embeddingModel }
    // text-embedding-3-* and gemini-embedding-* accept the `dimensions` param (Gemini's default is 3072, the collection
    // is 1536); others must natively match the collection size.
    if (supportsDimensions(cfg.embeddingModel)) embedderConfig.embeddingDims = dims

    const collectionName = getCollectionName()
    const memory = new Memory({
        version: 'v1.1',
        disableHistory: true,
        customInstructions: CUSTOMER_MEMORY_TAG_INSTRUCTIONS,
        embedder: { provider: 'openai', config: embedderConfig },
        llm: { provider: 'openai', config: { apiKey: cfg.apiKey, baseURL: cfg.baseUrl, model: cfg.llmModel } },
        vectorStore: {
            provider: 'qdrant',
            config: { client, collectionName, embeddingModelDims: dims, dimension: dims }
        }
    }) as unknown as Mem0Like
    wrapLlmForCapture(memory)
    memory.setPayload = async (ids: string[], payload: Record<string, any>) => {
        if (!ids.length) return
        await client.setPayload(collectionName, { points: ids, payload, wait: true })
    }
    // Summaries live in `<collection>_summaries`: one point per customer, 1-dim dummy vector, payload only.
    const summaryCollection = getSummaryCollectionName()
    let summaryReady: Promise<void> | null = null
    const ensureSummaryCollection = () =>
        (summaryReady ??= (async () => {
            const { exists } = await client.collectionExists(summaryCollection)
            if (!exists) {
                try {
                    await client.createCollection(summaryCollection, { vectors: { size: 1, distance: 'Dot' } })
                } catch (e: any) {
                    if (!/already exists/i.test(String(e?.message ?? e))) throw e
                }
            }
        })().catch((e) => {
            summaryReady = null
            throw e
        }))
    memory.getSummary = async (userId: string) => {
        const { exists } = await client.collectionExists(summaryCollection)
        if (!exists) return null
        const pts = await client.retrieve(summaryCollection, { ids: [summaryPointId(userId)], with_payload: true, with_vector: false })
        const v = pts?.[0]?.payload?.summary
        return typeof v === 'string' ? v : null
    }
    memory.saveSummary = async (userId: string, summary: string, payload: Record<string, any>) => {
        await ensureSummaryCollection()
        await client.upsert(summaryCollection, {
            wait: true,
            points: [
                {
                    id: summaryPointId(userId),
                    vector: [1],
                    payload: { ...payload, user_id: userId, summary, updatedAt: new Date().toISOString() }
                }
            ]
        })
    }

    instanceCache.set(key, memory)
    if (instanceCache.size > MAX_CACHED_INSTANCES) {
        const oldest = instanceCache.keys().next().value
        if (oldest) instanceCache.delete(oldest)
    }
    return memory
}

/** Test hook. */
export const __clearMem0InstanceCache = () => instanceCache.clear()
