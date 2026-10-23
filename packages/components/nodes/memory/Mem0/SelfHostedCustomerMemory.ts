import { BaseMessage, AIMessage } from '@langchain/core/messages'
import { InputValues, MemoryVariables, OutputValues } from '@langchain/core/memory'
import { DataSource } from 'typeorm'
import { FlowiseMemory, IDatabaseEntity, IMessage, MemoryMethods, MessageType } from '../../../src/Interface'
import { mapChatMessageToBaseMessage } from '../../../src/utils'
import {
    buildBackfillMessages,
    CustomerMemoryConfig,
    formatFacts,
    getMem0Instance,
    Mem0ChatMessage,
    Mem0Like,
    parseCustomerMemoryConfig,
    redactSecrets,
    withTimeout
} from './customerMemory'
import { addCustomerFacts, RECALL_THRESHOLD, RECALL_TIMEOUT_MS, userQueues } from './customerMemoryOps'

export { addCustomerFacts, queueForUser, recallCustomerFacts } from './customerMemoryOps'
export type { AddCustomerFactsResult, AddedFact, RecallResult } from './customerMemoryOps'

const MAX_MESSAGE_CHARS = 4000

export interface SelfHostedCustomerMemoryInput {
    rawCustomerMemory: unknown
    sessionId: string
    appDataSource: DataSource
    databaseEntities: IDatabaseEntity
    chatflowid: string
    orgId: string
    input: string
    searchOnly: boolean
    recallLimit: number
    historyWindow: number
    backfillMessages: number
    memoryKey: string
    inputKey: string
    logger?: { warn: (...a: any[]) => void; error: (...a: any[]) => void; debug?: (...a: any[]) => void }
    /** test hook */
    mem0Factory?: (cfg: CustomerMemoryConfig) => Promise<Mem0Like>
}

/**
 * Mem0 "Self-hosted" mode.
 *
 * - Chat history always comes from Flowise's chat_message table (same as Buffer Memory); the server persists it.
 * - When overrideConfig.customerMemory is enabled with an apiKey, facts are recalled from Qdrant before the LLM
 *   call and new facts are extracted after the reply (fire-and-forget, never throws into the reply path).
 * - Otherwise it is plain buffer memory and mem0 is never touched.
 */
export class SelfHostedCustomerMemory extends FlowiseMemory implements MemoryMethods {
    sessionId: string
    appDataSource: DataSource
    databaseEntities: IDatabaseEntity
    chatflowid: string
    orgId: string
    input: string
    searchOnly: boolean
    recallLimit: number
    historyWindow: number
    backfillMessages: number
    customerMemory: CustomerMemoryConfig | null
    private log: NonNullable<SelfHostedCustomerMemoryInput['logger']>
    private mem0Factory: (cfg: CustomerMemoryConfig) => Promise<Mem0Like>

    constructor(fields: SelfHostedCustomerMemoryInput) {
        super({ returnMessages: true, memoryKey: fields.memoryKey, inputKey: fields.inputKey, outputKey: 'output' })
        this.sessionId = fields.sessionId
        this.appDataSource = fields.appDataSource
        this.databaseEntities = fields.databaseEntities
        this.chatflowid = fields.chatflowid
        this.orgId = fields.orgId
        this.input = fields.input
        this.searchOnly = fields.searchOnly
        this.recallLimit = fields.recallLimit
        this.historyWindow = fields.historyWindow
        this.backfillMessages = fields.backfillMessages
        this.customerMemory = parseCustomerMemoryConfig(fields.rawCustomerMemory)
        this.log = fields.logger ?? console
        this.mem0Factory = fields.mem0Factory ?? getMem0Instance
    }

    get memoryKeys(): string[] {
        return [this.memoryKey]
    }

    private warn(msg: string, err?: unknown) {
        const who = this.customerMemory ? ` user=${this.customerMemory.userId}` : ''
        this.log.warn(`[CustomerMemory]${who} ${msg}${err !== undefined ? `: ${redactSecrets(err)}` : ''}`)
    }

    private async loadRecentRows(sessionId: string, take: number): Promise<any[]> {
        if (!sessionId || !this.appDataSource) return []
        const rows = await this.appDataSource.getRepository(this.databaseEntities['ChatMessage']).find({
            where: { sessionId, chatflowid: this.chatflowid },
            order: { createdDate: 'DESC' },
            take
        })
        return rows.reverse()
    }

    /** Recall facts for the current input. Returns '' on any failure or when memory is off. */
    async recallFacts(query: string): Promise<string> {
        const cfg = this.customerMemory
        if (!cfg || !query || !query.trim()) return ''
        try {
            const mem0 = await this.mem0Factory(cfg)
            const filters: Record<string, any> = { user_id: cfg.userId }
            // mem0 3.x drops hits below a 0.1 semantic score by default; generic turns ("which package suits me?")
            // score below that against fact text, so most facts were lost. A customer has few facts: rank, don't cut.
            const res = await withTimeout(
                mem0.search(query, { topK: this.recallLimit, threshold: RECALL_THRESHOLD, filters }),
                RECALL_TIMEOUT_MS,
                'mem0 search'
            )
            return formatFacts(res?.results ?? [], this.recallLimit)
        } catch (e) {
            this.warn('recall failed, continuing without facts', e)
            return ''
        }
    }

    async getChatMessages(
        overrideSessionId = '',
        returnBaseMessages = false,
        prependMessages?: IMessage[]
    ): Promise<IMessage[] | BaseMessage[]> {
        const id = overrideSessionId || this.sessionId
        const rows: any[] = await this.loadRecentRows(id, this.historyWindow)
        if (prependMessages?.length) rows.unshift(...(prependMessages as any[]))

        if (!returnBaseMessages) {
            return rows.map((m) => ({ message: (m.content ?? m.message) as string, type: (m.role ?? m.type) as MessageType }))
        }

        const messages = await mapChatMessageToBaseMessage(rows, this.orgId)
        const facts = await this.recallFacts(this.input)
        // Same convention as the platform Mem0 node: the recalled block is prepended as an assistant-side message,
        // which every chat model accepts mid-history (a SystemMessage would be rejected by e.g. Anthropic).
        if (facts) messages.unshift(new AIMessage(facts))
        return messages
    }

    async loadMemoryVariables(values: InputValues): Promise<MemoryVariables> {
        const query = (values?.[this.inputKey ?? 'input'] as string) ?? this.input ?? ''
        const rows = await this.loadRecentRows(this.sessionId, this.historyWindow)
        const messages = await mapChatMessageToBaseMessage(rows, this.orgId)
        const facts = await this.recallFacts(query)
        if (this.returnMessages) {
            return { [this.memoryKey]: facts ? [new AIMessage(facts), ...messages] : messages }
        }
        const transcript = messages.map((m) => `${m._getType() === 'human' ? 'Human' : 'AI'}: ${m.content}`).join('\n')
        return { [this.memoryKey]: [facts, transcript].filter(Boolean).join('\n\n') }
    }

    /** Fire-and-forget: returns immediately, extraction runs in the background. */
    async saveContext(inputValues: InputValues, outputValues: OutputValues): Promise<void> {
        const input = inputValues?.[this.inputKey ?? 'input']
        const output = outputValues?.output ?? outputValues?.text ?? Object.values(outputValues ?? {})[0]
        this.scheduleAdd(String(input ?? ''), String(output ?? ''), this.sessionId)
    }

    async addChatMessages(msgArray: { text: string; type: MessageType }[], overrideSessionId = ''): Promise<void> {
        // Chat history rows are persisted by the server; only mem0 extraction happens here.
        const input = msgArray.find((m) => m.type === 'userMessage')
        const output = msgArray.find((m) => m.type === 'apiMessage')
        this.scheduleAdd(input?.text ?? '', output?.text ?? '', overrideSessionId || this.sessionId)
    }

    async clearChatMessages(): Promise<void> {
        // Chat history is cleared at server level (like Buffer Memory). Customer facts are NOT wiped here:
        // use DELETE /api/v1/customer-memory/:userId so "clear chat" and "forget customer" stay separate actions.
        return
    }

    /** Exposed for tests: resolves when the background add for this call is finished. */
    lastAdd: Promise<void> = Promise.resolve()

    private scheduleAdd(input: string, output: string, sessionId: string) {
        const cfg = this.customerMemory
        if (!cfg || this.searchOnly) return
        const current: Mem0ChatMessage[] = []
        if (input.trim()) current.push({ role: 'user', content: input.slice(0, MAX_MESSAGE_CHARS) })
        if (output.trim()) current.push({ role: 'assistant', content: output.slice(0, MAX_MESSAGE_CHARS) })
        if (!current.length) return

        const prev = userQueues.get(cfg.userId) ?? Promise.resolve()
        const job = prev.then(() => this.runAdd(cfg, current, sessionId)).catch((e) => this.warn('add failed (reply unaffected)', e))
        userQueues.set(cfg.userId, job)
        void job.finally(() => {
            if (userQueues.get(cfg.userId) === job) userQueues.delete(cfg.userId)
        })
        this.lastAdd = job
    }

    private async runAdd(cfg: CustomerMemoryConfig, current: Mem0ChatMessage[], sessionId: string): Promise<void> {
        const mem0 = await this.mem0Factory(cfg)
        let messages = current

        if (cfg.backfill && sessionId) {
            try {
                const existing = await mem0.getAll({ topK: 1, filters: { user_id: cfg.userId } })
                if (!existing?.results?.length) {
                    const rows = await this.loadRecentRows(sessionId, this.backfillMessages)
                    const old = buildBackfillMessages(rows, current).map((m) => ({ ...m, content: m.content.slice(0, MAX_MESSAGE_CHARS) }))
                    if (old.length) {
                        messages = [...old, ...current]
                        this.log.debug?.(`[CustomerMemory] user=${cfg.userId} backfilling ${old.length} past messages`)
                    }
                }
            } catch (e) {
                this.warn('backfill lookup failed, adding current turn only', e)
            }
        }

        await addCustomerFacts(mem0, cfg, messages, { chatflowId: this.chatflowid, sessionId }, (m, e) => this.warn(m, e))
    }
}
