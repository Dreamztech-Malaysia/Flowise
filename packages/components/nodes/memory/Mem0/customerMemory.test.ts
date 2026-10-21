import { AIMessage } from '@langchain/core/messages'
import {
    buildBackfillMessages,
    CUSTOMER_MEMORY_FACTS_HEADER,
    formatFacts,
    parseCustomerMemoryConfig,
    redactSecrets
} from './customerMemory'
import { SelfHostedCustomerMemory } from './SelfHostedCustomerMemory'

describe('parseCustomerMemoryConfig', () => {
    const base = { enabled: true, userId: '12:60123', ownerId: 12, apiKey: 'sk-test-123456' }

    it('returns null when missing, disabled, or without key/userId', () => {
        expect(parseCustomerMemoryConfig(undefined)).toBeNull()
        expect(parseCustomerMemoryConfig('')).toBeNull()
        expect(parseCustomerMemoryConfig('not json')).toBeNull()
        expect(parseCustomerMemoryConfig({ ...base, enabled: false })).toBeNull()
        expect(parseCustomerMemoryConfig({ ...base, apiKey: '  ' })).toBeNull()
        expect(parseCustomerMemoryConfig({ ...base, userId: '' })).toBeNull()
        expect(parseCustomerMemoryConfig({ ...base, userId: 'has space' })).toBeNull()
    })

    it('applies defaults and accepts JSON strings (Flowise variable resolver may stringify)', () => {
        const cfg = parseCustomerMemoryConfig(JSON.stringify({ ...base, enabled: 'true', ownerId: '12', baseUrl: 'https://x.ai/v1/' }))
        expect(cfg).toEqual({
            userId: '12:60123',
            ownerId: 12,
            apiKey: 'sk-test-123456',
            baseUrl: 'https://x.ai/v1',
            llmModel: 'gpt-4.1-mini',
            embeddingModel: 'text-embedding-3-small',
            backfill: false
        })
        expect(parseCustomerMemoryConfig({ ...base, backfill: true })?.baseUrl).toBe('https://api.openai.com/v1')
        expect(parseCustomerMemoryConfig({ ...base, backfill: true })?.backfill).toBe(true)
    })
})

describe('formatFacts / redactSecrets / buildBackfillMessages', () => {
    it('formats facts with the outdated-warning header and a limit', () => {
        expect(formatFacts([], 15)).toBe('')
        const out = formatFacts([{ memory: 'Likes red' }, { memory: ' ' }, { memory: 'Budget 50k' }, { memory: 'x' }], 2)
        expect(out).toBe(`${CUSTOMER_MEMORY_FACTS_HEADER}\n- Likes red\n- Budget 50k`)
    })

    it('redacts keys from error messages', () => {
        const s = redactSecrets(new Error('Incorrect API key provided: sk-proj-abcdefghijkl. Bearer abc.def'))
        expect(s).not.toContain('abcdefghijkl')
        expect(s).not.toContain('abc.def')
    })

    it('drops the already-persisted current turn from backfill rows', () => {
        const rows = [
            { role: 'userMessage', content: 'hi' },
            { role: 'apiMessage', content: 'hello' },
            { role: 'userMessage', content: 'price?' },
            { role: 'apiMessage', content: 'RM10' }
        ]
        const current = [
            { role: 'user' as const, content: 'price?' },
            { role: 'assistant' as const, content: 'RM10' }
        ]
        expect(buildBackfillMessages(rows, current)).toEqual([
            { role: 'user', content: 'hi' },
            { role: 'assistant', content: 'hello' }
        ])
        // only the user row persisted so far
        expect(buildBackfillMessages(rows.slice(0, 3), current)).toHaveLength(2)
        // nothing persisted yet
        expect(buildBackfillMessages(rows.slice(0, 2), current)).toHaveLength(2)
    })
})

describe('SelfHostedCustomerMemory', () => {
    const rows = [
        { role: 'userMessage', content: 'old question', createdDate: 1 },
        { role: 'apiMessage', content: 'old answer', createdDate: 2 }
    ]
    const appDataSource: any = {
        getRepository: () => ({ find: jest.fn(async ({ take }: any) => [...rows].reverse().slice(0, take)) })
    }
    const silent = { warn: jest.fn(), error: jest.fn(), debug: jest.fn() }

    const make = (customerMemory: any, mem0: any, extra: any = {}) =>
        new SelfHostedCustomerMemory({
            rawCustomerMemory: customerMemory,
            sessionId: 's1',
            appDataSource,
            databaseEntities: { ChatMessage: {} } as any,
            chatflowid: 'cf1',
            orgId: 'o1',
            input: 'what colour?',
            searchOnly: false,
            recallLimit: 15,
            historyWindow: 10,
            backfillMessages: 50,
            memoryKey: 'chat_history',
            inputKey: 'input',
            logger: silent,
            mem0Factory: async () => mem0,
            ...extra
        })

    const cm = { enabled: true, userId: '7:abc', ownerId: 7, apiKey: 'sk-test-aaaaaaaa', backfill: true }

    it('acts as plain buffer memory when customerMemory is missing (no mem0 calls)', async () => {
        const mem0 = { search: jest.fn(), add: jest.fn(), getAll: jest.fn() }
        const m = make(undefined, mem0)
        const msgs: any[] = (await m.getChatMessages('s1', true)) as any[]
        expect(msgs).toHaveLength(2)
        await m.addChatMessages([
            { text: 'q', type: 'userMessage' },
            { text: 'a', type: 'apiMessage' }
        ])
        await m.lastAdd
        expect(mem0.search).not.toHaveBeenCalled()
        expect(mem0.add).not.toHaveBeenCalled()
    })

    it('injects recalled facts before history', async () => {
        const mem0 = { search: jest.fn(async () => ({ results: [{ memory: 'Prefers blue' }] })), add: jest.fn(), getAll: jest.fn() }
        const msgs: any[] = (await make(cm, mem0).getChatMessages('s1', true)) as any[]
        expect(mem0.search).toHaveBeenCalledWith('what colour?', { topK: 15, filters: { user_id: '7:abc' } })
        expect(msgs[0]).toBeInstanceOf(AIMessage)
        expect(msgs[0].content).toContain('Prefers blue')
        expect(msgs[0].content).toContain('outdated')
        expect(msgs).toHaveLength(3)
    })

    it('recall failure does not break history', async () => {
        const mem0 = { search: jest.fn(async () => Promise.reject(new Error('401 sk-test-aaaaaaaa'))), add: jest.fn(), getAll: jest.fn() }
        const msgs: any[] = (await make(cm, mem0).getChatMessages('s1', true)) as any[]
        expect(msgs).toHaveLength(2)
        const calls = silent.warn.mock.calls
        expect(String(calls[calls.length - 1]?.[0])).not.toContain('aaaaaaaa')
    })

    it('add runs in background, backfills when user has no memories, and swallows errors', async () => {
        const mem0 = {
            search: jest.fn(),
            getAll: jest.fn(async () => ({ results: [] })),
            add: jest.fn(async () => {
                throw new Error('insufficient_quota')
            })
        }
        const m = make(cm, mem0)
        await expect(
            m.addChatMessages([
                { text: 'new q', type: 'userMessage' },
                { text: 'new a', type: 'apiMessage' }
            ])
        ).resolves.toBeUndefined()
        await m.lastAdd
        expect(mem0.add).toHaveBeenCalledTimes(1)
        const [messages, opts] = (mem0.add.mock.calls[0] as any[]) ?? []
        expect(messages.map((x: any) => x.content)).toEqual(['old question', 'old answer', 'new q', 'new a'])
        expect(opts).toEqual({ userId: '7:abc', metadata: { source: 'flowise', chatflow_id: 'cf1', owner_id: 7, session_id: 's1' } })
        expect(silent.warn).toHaveBeenCalled()
    })

    it('skips backfill when memories already exist and never writes in searchOnly', async () => {
        const mem0 = { search: jest.fn(), getAll: jest.fn(async () => ({ results: [{ id: 'x' }] })), add: jest.fn(async () => ({})) }
        const m = make(cm, mem0)
        await m.addChatMessages([
            { text: 'q', type: 'userMessage' },
            { text: 'a', type: 'apiMessage' }
        ])
        await m.lastAdd
        expect(((mem0.add.mock.calls[0] as any[]) ?? [])[0]).toHaveLength(2)

        const ro = { search: jest.fn(), getAll: jest.fn(), add: jest.fn() }
        const m2 = make(cm, ro, { searchOnly: true })
        await m2.addChatMessages([
            { text: 'q', type: 'userMessage' },
            { text: 'a', type: 'apiMessage' }
        ])
        await m2.lastAdd
        expect(ro.add).not.toHaveBeenCalled()
    })
})
