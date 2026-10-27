import { StatusCodes } from 'http-status-codes'
import { InternalFlowiseError } from '../../errors/internalFlowiseError'

/**
 * Customer memory (mem0 self-hosted) read/delete API.
 *
 * Talks to Qdrant's REST API directly (no LLM / embedding call, no customer key needed) using the same
 * QDRANT_URL / QDRANT_API_KEY + collection as the Mem0 node in "Self-hosted" mode.
 *
 * Payload written by mem0ai OSS 3.x: { data, hash, createdAt, updatedAt?, user_id, owner_id, session_id,
 * chatflow_id, source, textLemmatized, ... }.
 */

export interface CustomerMemoryFact {
    id: string
    memory: string
    created_at: string | null
    updated_at: string | null
    user_id: string | null
    owner_id: number | string | null
    category: string | null
    label: string | null
    metadata: Record<string, any>
}

export interface CustomerMemoryFactsResult {
    facts: CustomerMemoryFact[]
    summary: string | null
    summaryUpdatedAt: string | null
}

export interface CustomerMemorySummaryRow {
    userId: string
    salesmanId: string | null
    category: string
    label: string
    /** Monday (UTC) of the ISO week the fact was created, YYYY-MM-DD */
    week: string
    count: number
}

export interface CustomerMemoryUser {
    userId: string
    ownerId: number | string | null
    count: number
    lastUpdatedAt: string | null
}

const INTERNAL_KEYS = new Set(['data', 'hash', 'createdAt', 'updatedAt', 'textLemmatized', 'user_id', 'owner_id', 'category', 'label'])
const PAGE = 256
const MAX_POINTS = 100_000

export const getCollectionName = () => process.env.CUSTOMER_MEMORY_COLLECTION || 'customer_memory'
/** Written by the Mem0 node: one point per customer with payload { user_id, summary, updatedAt, owner_id?, salesmanId? }. */
export const getSummaryCollectionName = () => `${getCollectionName()}_summaries`

const getQdrantConfig = () => {
    const url = process.env.QDRANT_URL
    if (!url) throw new InternalFlowiseError(StatusCodes.SERVICE_UNAVAILABLE, 'Customer memory is not configured (QDRANT_URL is not set)')
    return { url: url.replace(/\/+$/, ''), apiKey: process.env.QDRANT_API_KEY }
}

const qdrant = async (method: string, path: string, body?: any, allow404 = false): Promise<any> => {
    const { url, apiKey } = getQdrantConfig()
    const headers: Record<string, string> = { 'Content-Type': 'application/json' }
    if (apiKey) headers['api-key'] = apiKey
    let res: Response
    try {
        res = await fetch(`${url}${path}`, { method, headers, body: body ? JSON.stringify(body) : undefined })
    } catch (e: any) {
        throw new InternalFlowiseError(StatusCodes.BAD_GATEWAY, `Qdrant unreachable: ${e?.message ?? e}`)
    }
    if (res.status === 404 && allow404) return null
    if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new InternalFlowiseError(
            StatusCodes.BAD_GATEWAY,
            `Qdrant ${method} ${path.split('?')[0]} failed: ${res.status} ${text.slice(0, 300)}`
        )
    }
    const json: any = await res.json()
    return json?.result
}

const collectionPath = (name = getCollectionName()) => `/collections/${encodeURIComponent(name)}`

/** owner_id is stored as a number when the caller sent an int; match both representations. */
const ownerCondition = (ownerId: string) => {
    const n = Number(ownerId)
    if (ownerId.trim() !== '' && Number.isInteger(n)) {
        return {
            should: [
                { key: 'owner_id', match: { value: n } },
                { key: 'owner_id', match: { value: ownerId } }
            ]
        }
    }
    return { must: [{ key: 'owner_id', match: { value: ownerId } }] }
}

const userFilter = (userId: string) => ({ must: [{ key: 'user_id', match: { value: userId } }] })

/** Scroll every point matching `filter` (bounded by MAX_POINTS). Missing collection => []. */
const scrollAll = async (filter: any, withPayload: any = true, collection = getCollectionName()): Promise<any[]> => {
    const points: any[] = []
    let offset: any = undefined
    do {
        const result = await qdrant(
            'POST',
            `${collectionPath(collection)}/points/scroll`,
            { filter, limit: PAGE, with_payload: withPayload, with_vector: false, ...(offset !== undefined ? { offset } : {}) },
            true
        )
        if (!result) return points
        points.push(...(result.points ?? []))
        offset = result.next_page_offset ?? undefined
    } while (offset !== undefined && offset !== null && points.length < MAX_POINTS)
    return points
}

const toIso = (v: unknown): string | null => {
    if (v === undefined || v === null || v === '') return null
    const d = typeof v === 'number' ? new Date(v < 1e12 ? v * 1000 : v) : new Date(String(v))
    return isNaN(d.getTime()) ? null : d.toISOString()
}

export const toFact = (point: any): CustomerMemoryFact => {
    const p = point?.payload ?? {}
    const metadata: Record<string, any> = {}
    for (const [k, v] of Object.entries(p)) if (!INTERNAL_KEYS.has(k)) metadata[k] = v
    return {
        id: String(point.id),
        memory: typeof p.data === 'string' ? p.data : '',
        created_at: toIso(p.createdAt ?? p.created_at),
        updated_at: toIso(p.updatedAt ?? p.updated_at),
        user_id: p.user_id ?? null,
        owner_id: p.owner_id ?? null,
        category: p.category ?? (Array.isArray(p.categories) ? p.categories[0] ?? null : null),
        label: typeof p.label === 'string' ? p.label : null,
        metadata
    }
}

const validateUserId = (userId: string) => {
    if (!userId || typeof userId !== 'string' || userId.length > 256 || /\s/.test(userId)) {
        throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'Invalid userId')
    }
}

const getFacts = async (userId: string): Promise<CustomerMemoryFactsResult> => {
    validateUserId(userId)
    const [points, summaryPoints] = await Promise.all([
        scrollAll(userFilter(userId)),
        qdrant(
            'POST',
            `${collectionPath(getSummaryCollectionName())}/points/scroll`,
            { filter: userFilter(userId), limit: 1, with_payload: ['summary', 'updatedAt'], with_vector: false },
            true
        )
    ])
    const facts = points.map(toFact).sort((a, b) => (b.updated_at ?? b.created_at ?? '').localeCompare(a.updated_at ?? a.created_at ?? ''))
    const sp = summaryPoints?.points?.[0]?.payload ?? {}
    return {
        facts,
        summary: typeof sp.summary === 'string' ? sp.summary : null,
        summaryUpdatedAt: toIso(sp.updatedAt)
    }
}

/** Monday (UTC) of the ISO week containing `iso`, as YYYY-MM-DD. */
export const weekStart = (iso: string): string => {
    const d = new Date(iso)
    const day = (d.getUTCDay() + 6) % 7 // Mon=0
    const m = new Date(Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate() - day))
    return m.toISOString().slice(0, 10)
}

/**
 * Counts facts per (userId, salesmanId, category, label, week). Pure Qdrant scroll + counting, no AI.
 * Untagged (pre-tagging) facts count as category 'other' / label 'Other'.
 */
const getSummary = async (ownerId?: string, since?: string): Promise<CustomerMemorySummaryRow[]> => {
    let sinceTs: number | undefined
    if (since) {
        if (!/^\d{4}-\d{2}-\d{2}$/.test(since)) throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'since must be YYYY-MM-DD')
        sinceTs = new Date(`${since}T00:00:00.000Z`).getTime()
        if (isNaN(sinceTs)) throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'since must be YYYY-MM-DD')
    }
    const filter = ownerId ? ownerCondition(ownerId) : undefined
    const points = await scrollAll(filter, ['user_id', 'salesmanId', 'category', 'label', 'createdAt'])
    const rows = new Map<string, CustomerMemorySummaryRow>()
    for (const pt of points) {
        const p = pt.payload ?? {}
        if (!p.user_id) continue
        const created = toIso(p.createdAt)
        if (!created) continue
        if (sinceTs !== undefined && new Date(created).getTime() < sinceTs) continue
        const salesmanId = p.salesmanId === undefined || p.salesmanId === null || p.salesmanId === '' ? null : String(p.salesmanId)
        const category = typeof p.category === 'string' && p.category ? p.category : 'other'
        const label = typeof p.label === 'string' && p.label ? p.label : 'Other'
        const week = weekStart(created)
        const key = [p.user_id, salesmanId ?? '', category, label, week].join('\u0000')
        const row = rows.get(key) ?? { userId: p.user_id, salesmanId, category, label, week, count: 0 }
        row.count++
        rows.set(key, row)
    }
    return [...rows.values()].sort(
        (a, b) =>
            b.week.localeCompare(a.week) ||
            a.userId.localeCompare(b.userId) ||
            a.category.localeCompare(b.category) ||
            a.label.localeCompare(b.label)
    )
}

const deleteAllFacts = async (userId: string): Promise<{ deleted: number }> => {
    validateUserId(userId)
    const points = await scrollAll(userFilter(userId), false)
    if (points.length) {
        await qdrant('POST', `${collectionPath()}/points/delete?wait=true`, { filter: userFilter(userId) }, true)
    }
    // mem0 also links entities per user in `<collection>_entities`; purge those too (PDPA "forget").
    await qdrant(
        'POST',
        `${collectionPath(`${getCollectionName()}_entities`)}/points/delete?wait=true`,
        { filter: userFilter(userId) },
        true
    )
    // ...and the stored at-a-glance summary.
    await qdrant('POST', `${collectionPath(getSummaryCollectionName())}/points/delete?wait=true`, { filter: userFilter(userId) }, true)
    return { deleted: points.length }
}

const deleteFact = async (userId: string, memoryId: string): Promise<{ deleted: number }> => {
    validateUserId(userId)
    if (!memoryId || !/^[0-9a-fA-F-]{1,64}$|^\d+$/.test(memoryId))
        throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'Invalid memoryId')
    const id = /^\d+$/.test(memoryId) ? Number(memoryId) : memoryId
    const result = await qdrant('POST', `${collectionPath()}/points`, { ids: [id], with_payload: ['user_id'], with_vector: false }, true)
    const point = Array.isArray(result) ? result[0] : undefined
    // Never let a caller delete another customer's fact by guessing ids.
    if (!point || point.payload?.user_id !== userId) {
        throw new InternalFlowiseError(StatusCodes.NOT_FOUND, `Memory ${memoryId} not found for this user`)
    }
    await qdrant('POST', `${collectionPath()}/points/delete?wait=true`, { points: [id] })
    // The stored summary may quote the forgotten fact: drop it. The next extraction writes a fresh one and
    // callers fall back to the remaining facts meanwhile.
    await qdrant('POST', `${collectionPath(getSummaryCollectionName())}/points/delete?wait=true`, { filter: userFilter(userId) }, true)
    return { deleted: 1 }
}

/**
 * List customers (mem0 user ids) with facts. `ownerId` scopes to one salesman/owner; `updatedBefore` keeps only users
 * whose most recent fact is older than the given ISO date (retention cron).
 */
const listUsers = async (ownerId?: string, updatedBefore?: string): Promise<CustomerMemoryUser[]> => {
    let before: number | undefined
    if (updatedBefore) {
        before = new Date(updatedBefore).getTime()
        if (isNaN(before)) throw new InternalFlowiseError(StatusCodes.BAD_REQUEST, 'updatedBefore must be an ISO date')
    }
    const filter = ownerId ? ownerCondition(ownerId) : undefined
    const points = await scrollAll(filter, ['user_id', 'owner_id', 'createdAt', 'updatedAt'])
    const users = new Map<string, CustomerMemoryUser & { _ts: number }>()
    for (const pt of points) {
        const p = pt.payload ?? {}
        if (!p.user_id) continue
        const iso = toIso(p.updatedAt) ?? toIso(p.createdAt)
        const ts = iso ? new Date(iso).getTime() : 0
        const u = users.get(p.user_id) ?? { userId: p.user_id, ownerId: p.owner_id ?? null, count: 0, lastUpdatedAt: null, _ts: -1 }
        u.count++
        if (ts > u._ts) {
            u._ts = ts
            u.lastUpdatedAt = iso
        }
        users.set(p.user_id, u)
    }
    return [...users.values()]
        .filter((u) => before === undefined || u._ts < before)
        .sort((a, b) => b._ts - a._ts)
        .map(({ _ts, ...u }) => u)
}

export default { getFacts, getSummary, deleteAllFacts, deleteFact, listUsers }
