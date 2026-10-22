import { AIMessage } from '@langchain/core/messages'
import {
    buildBackfillMessages,
    CUSTOMER_MEMORY_FACTS_HEADER,
    extractionCapture,
    formatFacts,
    injectSummary,
    parseCustomerMemoryConfig,
    parseExtractionExtras,
    redactSecrets,
    resolveTag,
    ruleTag,
    summaryPointId,
    wrapLlmForCapture
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
        expect(parseCustomerMemoryConfig({ ...base, salesmanId: 33 })?.salesmanId).toBe('33')
        expect(parseCustomerMemoryConfig({ ...base, salesmanId: ' ' })).not.toHaveProperty('salesmanId')
    })
})

describe('category/label tagging', () => {
    it('parses per-fact category/label from the raw extraction JSON and falls back to keyword rules', () => {
        const raw =
            '```json\n{"memory":[{"id":"0","text":"Wants the family package","category":"Interest","label":"family package"},' +
            '{"id":"1","text":"Thinks price is too high","category":"bogus"}]}\n```'
        const ex = parseExtractionExtras(raw)
        expect(resolveTag('Wants the family package', ex)).toEqual({ category: 'interest', label: 'Family package' })
        expect(resolveTag('Thinks price is too high', ex)).toEqual({ category: 'objection', label: 'Price Too High' })
        expect(resolveTag('Has two kids', parseExtractionExtras('garbage'))).toEqual({ category: 'family', label: 'Family' })
        expect(ruleTag('Blue sky')).toEqual({ category: 'other', label: 'Other' })
    })

    it('captures the raw LLM response only inside extractionCapture.run', async () => {
        const memory: any = { llm: { generateResponse: async () => '{"memory":[]}' } }
        wrapLlmForCapture(memory)
        wrapLlmForCapture(memory)
        const cap: any = {}
        await extractionCapture.run(cap, () => memory.llm.generateResponse([]))
        expect(cap.raw).toBe('{"memory":[]}')
        await expect(memory.llm.generateResponse([])).resolves.toBe('{"memory":[]}')
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

    const cm = { enabled: true, userId: '7:abc', ownerId: 7, salesmanId: 's9', apiKey: 'sk-test-aaaaaaaa', backfill: true }

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
        expect(opts).toEqual({
            userId: '7:abc',
            metadata: { source: 'flowise', chatflow_id: 'cf1', owner_id: 7, ownerId: 7, salesmanId: 's9', session_id: 's1' }
        })
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

    it('tags newly added facts with category/label from the same extraction response (no extra LLM call)', async () => {
        const llm = { generateResponse: jest.fn(async () => '{"memory":[{"id":"0","text":"Wants a family package","category":"interest","label":"Family package"},{"id":"1","text":"Has 3 kids"}]}') }
        const mem0: any = {
            llm,
            search: jest.fn(),
            getAll: jest.fn(async () => ({ results: [{ id: 'x' }] })),
            add: jest.fn(async () => {
                await mem0.llm.generateResponse([])
                return { results: [{ id: 'f1', memory: 'Wants a family package' }, { id: 'f2', memory: 'Has 3 kids' }] }
            }),
            setPayload: jest.fn(async () => undefined)
        }
        wrapLlmForCapture(mem0)
        const m = make(cm, mem0)
        await m.addChatMessages([
            { text: 'q', type: 'userMessage' },
            { text: 'a', type: 'apiMessage' }
        ])
        await m.lastAdd
        expect(llm.generateResponse).toHaveBeenCalledTimes(1)
        expect(mem0.setPayload).toHaveBeenCalledWith(['f1'], { category: 'interest', label: 'Family package' })
        expect(mem0.setPayload).toHaveBeenCalledWith(['f2'], { category: 'family', label: 'Family' })
    })

    it('rewrites the summary in the same extraction call only when new facts were added', async () => {
        const llm = {
            generateResponse: jest.fn(async (msgs: any[]) => {
                expect(msgs[1].content).toContain('## Current Customer Summary\nOld summary')
                return '{"memory":[{"id":"0","text":"Budget RM500k","category":"budget","label":"Budget RM500k"}],"summary":"Buyer with RM500k budget."}'
            })
        }
        const results = [{ id: 'f1', memory: 'Budget RM500k' }]
        const mem0: any = {
            llm,
            search: jest.fn(),
            getAll: jest.fn(async () => ({ results: [{ id: 'x' }] })),
            add: jest.fn(async () => {
                await mem0.llm.generateResponse([{ role: 'system', content: 's' }, { role: 'user', content: 'u' }])
                return { results }
            }),
            setPayload: jest.fn(async () => undefined),
            getSummary: jest.fn(async () => 'Old summary'),
            saveSummary: jest.fn(async () => undefined)
        }
        wrapLlmForCapture(mem0)
        const m = make(cm, mem0)
        await m.addChatMessages([{ text: 'q', type: 'userMessage' }])
        await m.lastAdd
        expect(llm.generateResponse).toHaveBeenCalledTimes(1)
        expect(mem0.saveSummary).toHaveBeenCalledWith('7:abc', 'Buyer with RM500k budget.', { owner_id: 7, salesmanId: 's9' })

        results.length = 0
        await m.addChatMessages([{ text: 'q2', type: 'userMessage' }])
        await m.lastAdd
        expect(mem0.saveSummary).toHaveBeenCalledTimes(1)
    })
})

describe('summary helpers', () => {
    it('injects the previous summary into the last user message without mutating input', () => {
        const msgs = [{ role: 'system', content: 's' }, { role: 'user', content: 'u' }]
        const out = injectSummary(msgs, null)
        expect(out[1].content).toBe('u\n\n## Current Customer Summary\n(none yet)')
        expect(msgs[1].content).toBe('u')
    })

    it('derives a stable uuid per user', () => {
        expect(summaryPointId('7:a')).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/)
        expect(summaryPointId('7:a')).toBe(summaryPointId('7:a'))
        expect(summaryPointId('7:a')).not.toBe(summaryPointId('7:b'))
    })
})
