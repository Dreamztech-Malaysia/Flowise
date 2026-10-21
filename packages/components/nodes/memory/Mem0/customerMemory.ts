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

    const str = (v: unknown, d: string) => (typeof v === 'string' && v.trim() ? v.trim() : d)
    return {
        userId,
        ownerId,
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
export const buildBackfillMessages = (
    rows: { role: string; content: string }[],
    current: Mem0ChatMessage[]
): Mem0ChatMessage[] => {
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

export const getCollectionName = (): string => process.env.CUSTOMER_MEMORY_COLLECTION || CUSTOMER_MEMORY_DEFAULT_COLLECTION

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
}

const MAX_CACHED_INSTANCES = 200
const instanceCache = new Map<string, Mem0Like>()

const cacheKey = (cfg: CustomerMemoryConfig) =>
    createHash('sha256').update([cfg.apiKey, cfg.baseUrl, cfg.llmModel, cfg.embeddingModel, getCollectionName()].join('\u0000')).digest('hex')

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
    // Only text-embedding-3-* accepts the `dimensions` param; others must natively match the collection size.
    if (/^text-embedding-3/.test(cfg.embeddingModel)) embedderConfig.embeddingDims = dims

    const memory = new Memory({
        version: 'v1.1',
        disableHistory: true,
        embedder: { provider: 'openai', config: embedderConfig },
        llm: { provider: 'openai', config: { apiKey: cfg.apiKey, baseURL: cfg.baseUrl, model: cfg.llmModel } },
        vectorStore: {
            provider: 'qdrant',
            config: { client, collectionName: getCollectionName(), embeddingModelDims: dims, dimension: dims }
        }
    }) as unknown as Mem0Like

    instanceCache.set(key, memory)
    if (instanceCache.size > MAX_CACHED_INSTANCES) {
        const oldest = instanceCache.keys().next().value
        if (oldest) instanceCache.delete(oldest)
    }
    return memory
}

/** Test hook. */
export const __clearMem0InstanceCache = () => instanceCache.clear()
