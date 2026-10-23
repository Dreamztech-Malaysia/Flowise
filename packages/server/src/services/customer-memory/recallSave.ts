import { StatusCodes } from 'http-status-codes'
import {
    addCustomerFacts,
    AddCustomerFactsResult,
    CustomerMemoryConfig,
    getMem0Instance,
    Mem0ChatMessage,
    Mem0Like,
    parseCustomerMemoryConfig,
    queueForUser,
    recallCustomerFacts,
    RecallResult,
    redactSecrets
} from 'flowise-components'
import { InternalFlowiseError } from '../../errors/internalFlowiseError'
import logger from '../../utils/logger'

/**
 * Recall/save for flows that cannot host the Mem0 memory node (agentflows such as Test Processor 3).
 * Same mem0 instance, fact rules, tagging and summary as the node; the caller passes the customerMemory block
 * (plain, API-key auth) or a sealed token (see sealedToken.ts).
 */

export const MAX_MESSAGES = 20
export const MAX_MESSAGE_CHARS = 4000
export const DEFAULT_RECALL_LIMIT = 15
export const MAX_RECALL_LIMIT = 50

let mem0Factory: (cfg: CustomerMemoryConfig) => Promise<Mem0Like> = getMem0Instance
/** Test hook. */
export const __setMem0Factory = (f?: (cfg: CustomerMemoryConfig) => Promise<Mem0Like>) => {
    mem0Factory = f ?? getMem0Instance
}

export const toConfig = (raw: unknown): CustomerMemoryConfig => {
    const cfg = parseCustomerMemoryConfig(raw)
    if (!cfg) throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'customerMemory is missing, disabled or invalid')
    return cfg
}

/** Accepts [{role, content}] or {input, output}; anything else => 400. Trims and caps sizes. */
export const toMessages = (body: any): Mem0ChatMessage[] => {
    let list: any[] = []
    if (Array.isArray(body?.messages)) list = body.messages
    else if (typeof body?.input === 'string' || typeof body?.output === 'string') {
        list = [
            { role: 'user', content: body.input ?? '' },
            { role: 'assistant', content: body.output ?? '' }
        ]
    }
    const out: Mem0ChatMessage[] = []
    for (const m of list.slice(-MAX_MESSAGES)) {
        const role =
            m?.role === 'assistant' || m?.role === 'apiMessage' || m?.role === 'ai'
                ? 'assistant'
                : m?.role === 'user' || m?.role === 'userMessage' || m?.role === 'human'
                ? 'user'
                : null
        const content = typeof m?.content === 'string' ? m.content.trim().slice(0, MAX_MESSAGE_CHARS) : ''
        if (role && content) out.push({ role, content })
    }
    if (!out.some((m) => m.role === 'user'))
        throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'messages must contain at least one user message')
    return out
}

const providerError = (what: string, e: unknown): InternalFlowiseError => {
    if (e instanceof InternalFlowiseError) return e
    return new InternalFlowiseError(StatusCodes.BAD_GATEWAY, `customer memory ${what} failed: ${redactSecrets(e)}`)
}

export const recall = async (cfg: CustomerMemoryConfig, query: unknown, limit: unknown): Promise<RecallResult> => {
    const q = typeof query === 'string' ? query.slice(0, MAX_MESSAGE_CHARS) : ''
    const n = Math.min(MAX_RECALL_LIMIT, Math.max(1, Number(limit) || DEFAULT_RECALL_LIMIT))
    try {
        const mem0 = await mem0Factory(cfg)
        return await recallCustomerFacts(mem0, cfg, q, n)
    } catch (e) {
        throw providerError('recall', e)
    }
}

export const save = async (
    cfg: CustomerMemoryConfig,
    messages: Mem0ChatMessage[],
    ctx: { chatflowId?: string; sessionId?: string; source?: string }
): Promise<AddCustomerFactsResult> => {
    const warn = (msg: string, err?: unknown) =>
        logger.warn(`[CustomerMemory] user=${cfg.userId} ${msg}${err !== undefined ? `: ${redactSecrets(err)}` : ''}`)
    try {
        return await queueForUser(cfg.userId, async () => {
            const mem0 = await mem0Factory(cfg)
            const res = await addCustomerFacts(mem0, cfg, messages, ctx, warn)
            const u = res.usage
            logger.info(
                `[CustomerMemory] user=${cfg.userId} saved ${res.added.length} fact(s); tokens llm ${u.promptTokens}+${u.completionTokens} (${u.llmCalls} call), embeddings ${u.embeddingTokens} (${u.embeddingCalls} call)`
            )
            return res
        })
    } catch (e) {
        throw providerError('save', e)
    }
}

/** Fire-and-forget variant: never rejects, logs instead. */
export const saveInBackground = (
    cfg: CustomerMemoryConfig,
    messages: Mem0ChatMessage[],
    ctx: { chatflowId?: string; sessionId?: string; source?: string }
) => {
    void save(cfg, messages, ctx).catch((e) =>
        logger.warn(`[CustomerMemory] user=${cfg.userId} background save failed: ${redactSecrets(e?.message ?? e)}`)
    )
}

const cleanId = (v: unknown, max = 256): string | undefined => (typeof v === 'string' && v.trim() ? v.trim().slice(0, max) : undefined)
export const toContext = (body: any, source: string) => ({
    chatflowId: cleanId(body?.chatflowId),
    sessionId: cleanId(body?.sessionId),
    source
})
