/**
 * Customer memory operations shared by the Mem0 node (SelfHostedCustomerMemory) and the server's
 * /api/v1/customer-memory recall/save endpoints (agentflows). Kept free of Flowise/langchain imports so the
 * server can load it without the whole components index.
 */
import {
    CustomerMemoryConfig,
    emptyUsage,
    ExtractionCapture,
    extractionCapture,
    ExtractionExtras,
    formatFacts,
    Mem0ChatMessage,
    Mem0Like,
    parseExtractionExtras,
    resolveTag,
    TokenUsage,
    withTimeout
} from './customerMemory'

export const RECALL_TIMEOUT_MS = 8000
export const RECALL_THRESHOLD = 0

/** Serialises extraction per mem0 user so two quick messages cannot both run the backfill / extract concurrently. */
export const userQueues = new Map<string, Promise<void>>()

export interface AddedFact {
    id: string
    memory: string
    category: string
    label: string
}

export interface AddCustomerFactsResult {
    added: AddedFact[]
    summary: string | null
    usage: TokenUsage
}

/**
 * One mem0 extraction for a customer: add facts (fact rules + sanitiser applied by the LLM wrapper), tag them with
 * category/label from the same response and rewrite the summary. Shared by the Mem0 node and the
 * /api/v1/customer-memory save endpoint (agentflows). Throws on provider errors; callers decide.
 */
export const addCustomerFacts = async (
    mem0: Mem0Like,
    cfg: CustomerMemoryConfig,
    messages: Mem0ChatMessage[],
    ctx: { chatflowId?: string; sessionId?: string; source?: string },
    warn: (msg: string, err?: unknown) => void
): Promise<AddCustomerFactsResult> => {
    const metadata: Record<string, any> = { source: ctx.source ?? 'flowise' }
    if (ctx.chatflowId) metadata.chatflow_id = ctx.chatflowId
    // owner_id (snake) is what the server filters on; ownerId/salesmanId (camel) are the reporting keys.
    if (cfg.ownerId !== undefined) {
        metadata.owner_id = cfg.ownerId
        metadata.ownerId = cfg.ownerId
    }
    if (cfg.salesmanId !== undefined) metadata.salesmanId = cfg.salesmanId
    if (ctx.sessionId) metadata.session_id = ctx.sessionId

    const capture: ExtractionCapture = { usage: emptyUsage() }
    if (mem0.getSummary) {
        try {
            capture.previousSummary = await mem0.getSummary(cfg.userId)
        } catch (e) {
            warn('summary lookup failed', e)
            capture.previousSummary = null
        }
    }
    const res = await extractionCapture.run(capture, () => mem0.add(messages, { userId: cfg.userId, metadata }))
    const facts: { id: string; memory: string }[] = (res?.results ?? []).filter(
        (r: any) => r && typeof r.id === 'string' && typeof r.memory === 'string'
    )
    const usage = capture.usage ?? emptyUsage()
    // Fact set unchanged => keep the stored summary as is.
    if (!facts.length) return { added: [], summary: null, usage }
    const extras = parseExtractionExtras(capture.raw)
    const added = await tagFacts(mem0, facts, extras, warn)
    let summary: string | null = null
    if (extras.summary && mem0.saveSummary) {
        const payload: Record<string, any> = {}
        if (cfg.ownerId !== undefined) payload.owner_id = cfg.ownerId
        if (cfg.salesmanId !== undefined) payload.salesmanId = cfg.salesmanId
        try {
            await mem0.saveSummary(cfg.userId, extras.summary, payload)
            summary = extras.summary
        } catch (e) {
            warn('saving summary failed', e)
        }
    }
    return { added, summary, usage }
}

/** category/label come from the same extraction response (no extra LLM call); keyword rules fill gaps. */
const tagFacts = async (
    mem0: Mem0Like,
    facts: { id: string; memory: string }[],
    extras: ExtractionExtras,
    warn: (msg: string, err?: unknown) => void
): Promise<AddedFact[]> => {
    const tagged = facts.map((f) => ({ ...f, ...resolveTag(f.memory, extras) }))
    if (!mem0.setPayload) return tagged
    // group by identical tag so most turns cost one Qdrant set_payload request
    const groups = new Map<string, { tag: { category: string; label: string }; ids: string[] }>()
    for (const f of tagged) {
        const k = `${f.category}\u0000${f.label}`
        const g = groups.get(k) ?? { tag: { category: f.category, label: f.label }, ids: [] }
        g.ids.push(f.id)
        groups.set(k, g)
    }
    for (const g of groups.values()) {
        try {
            await mem0.setPayload(g.ids, { category: g.tag.category, label: g.tag.label })
        } catch (e) {
            warn('tagging facts failed (facts kept untagged)', e)
        }
    }
    return tagged
}

/** Serialise one job per mem0 user (same queue the node uses) so two quick turns never extract concurrently. */
export const queueForUser = <T>(userId: string, job: () => Promise<T>): Promise<T> => {
    const prev = userQueues.get(userId) ?? Promise.resolve()
    const run = prev.then(job, job)
    const settled = run.then(
        () => undefined,
        () => undefined
    )
    userQueues.set(userId, settled)
    void settled.finally(() => {
        if (userQueues.get(userId) === settled) userQueues.delete(userId)
    })
    return run
}

export interface RecallResult {
    facts: { id: string; memory: string; category: string | null; label: string | null }[]
    summary: string | null
    /** Ready-to-inject prompt block ('' when nothing is known) */
    text: string
    usage: TokenUsage
}

/** Facts most relevant to `query` (all facts ranked; no score cutoff) + stored summary. Throws on provider errors. */
export const recallCustomerFacts = async (
    mem0: Mem0Like,
    cfg: CustomerMemoryConfig,
    query: string,
    limit: number
): Promise<RecallResult> => {
    const capture: ExtractionCapture = { usage: emptyUsage() }
    const filters = { user_id: cfg.userId }
    const res = await extractionCapture.run(capture, () =>
        query && query.trim()
            ? withTimeout(mem0.search(query, { topK: limit, threshold: RECALL_THRESHOLD, filters }), RECALL_TIMEOUT_MS, 'mem0 search')
            : withTimeout(mem0.getAll({ topK: limit, filters }), RECALL_TIMEOUT_MS, 'mem0 getAll')
    )
    const facts = (res?.results ?? [])
        .filter((r: any) => r && typeof r.memory === 'string' && r.memory.trim())
        .slice(0, limit)
        .map((r: any) => ({
            id: String(r.id),
            memory: r.memory.trim(),
            category: typeof r.metadata?.category === 'string' ? r.metadata.category : null,
            label: typeof r.metadata?.label === 'string' ? r.metadata.label : null
        }))
    let summary: string | null = null
    if (facts.length && mem0.getSummary) {
        try {
            summary = await mem0.getSummary(cfg.userId)
        } catch {
            summary = null
        }
    }
    const factBlock = formatFacts(facts, limit)
    const text = factBlock ? (summary ? `${factBlock}\nAt a glance: ${summary}` : factBlock) : ''
    return { facts, summary, text, usage: capture.usage ?? emptyUsage() }
}
