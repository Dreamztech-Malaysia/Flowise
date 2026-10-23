import { AIMessage } from '@langchain/core/messages'
import {
    buildBackfillMessages,
    cleanFactText,
    CUSTOMER_MEMORY_FACT_RULES,
    CUSTOMER_MEMORY_FACTS_HEADER,
    emptyUsage,
    extractionCapture,
    formatFacts,
    injectFactRules,
    injectSummary,
    parseCustomerMemoryConfig,
    parseExtractionExtras,
    redactSecrets,
    resolveTag,
    ruleTag,
    sanitizeExtraction,
    summaryPointId,
    supportsDimensions,
    wrapLlmForCapture
} from './customerMemory'
import { addCustomerFacts, queueForUser, recallCustomerFacts, SelfHostedCustomerMemory } from './SelfHostedCustomerMemory'

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
        expect(mem0.search).toHaveBeenCalledWith('what colour?', { topK: 15, threshold: 0, filters: { user_id: '7:abc' } })
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
        const gen = jest.fn(async () => '{"memory":[{"id":"0","text":"Wants a family package","category":"interest","label":"Family package"},{"id":"1","text":"Has 3 kids"}]}')
        const llm = { generateResponse: gen }
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
        expect(gen).toHaveBeenCalledTimes(1)
        expect(mem0.setPayload).toHaveBeenCalledWith(['f1'], { category: 'interest', label: 'Family package' })
        expect(mem0.setPayload).toHaveBeenCalledWith(['f2'], { category: 'family', label: 'Family' })
    })

    it('rewrites the summary in the same extraction call only when new facts were added', async () => {
        const gen = jest.fn(async (msgs: any[]) => {
                expect(msgs[1].content).toContain('## Current Customer Summary\nOld summary')
                return '{"memory":[{"id":"0","text":"Budget RM500k","category":"budget","label":"Budget RM500k"}],"summary":"Buyer with RM500k budget."}'
        })
        const llm = { generateResponse: gen }
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
        expect(gen).toHaveBeenCalledTimes(1)
        expect(mem0.saveSummary).toHaveBeenCalledWith('7:abc', 'Buyer with RM500k budget.', { owner_id: 7, salesmanId: 's9' })

        results.length = 0
        await m.addChatMessages([{ text: 'q2', type: 'userMessage' }])
        await m.lastAdd
        expect(mem0.saveSummary).toHaveBeenCalledTimes(1)
    })
})

describe('fact quality (customer-only, short, no dates, no duplicates)', () => {
    it('cleans subjects, date stamps and trailing punctuation', () => {
        expect(cleanFactText('User has 2 kids.')).toBe('Has 2 kids')
        expect(cleanFactText("Customer's budget is RM500 as of October 2026")).toBe('Budget is RM500')
        expect(cleanFactText('User is based in Johor Bahru (2026-10-23)')).toBe('Based in Johor Bahru')
        expect(cleanFactText('Prefers contact after 6pm, as mentioned on October 23, 2026')).toBe('Prefers contact after 6pm')
        // a date that is part of the fact itself stays
        expect(cleanFactText('Moving house in December 2026')).toBe('Moving house in December 2026')
        expect(cleanFactText('   ')).toBe('')
        expect(cleanFactText(42)).toBe('')
    })

    it('drops assistant facts, assistant offers and duplicates; keeps tags and summary', () => {
        const raw = JSON.stringify({
            memory: [
                { id: '0', text: 'User has 2 kids', attributed_to: 'user', category: 'family', label: 'Has Kids' },
                { id: '1', text: 'User was recommended the Basic package', attributed_to: 'user' },
                { id: '2', text: 'Assistant offered a 10% discount', attributed_to: 'assistant' },
                { id: '3', text: 'has 2 kids.', attributed_to: 'user' },
                { id: '4', text: 'Budget RM500 as of October 2026' },
                { id: '5', text: '' }
            ],
            summary: 'Parent of two in JB.'
        })
        const out = JSON.parse(sanitizeExtraction('```json\n' + raw + '\n```') as string)
        expect(out.memory.map((m: any) => m.text)).toEqual(['Has 2 kids', 'Budget RM500'])
        expect(out.memory[0]).toMatchObject({ category: 'family', label: 'Has Kids', attributed_to: 'user' })
        expect(out.summary).toBe('Parent of two in JB.')
        expect(sanitizeExtraction('not json')).toBe('not json')
        expect(sanitizeExtraction(undefined)).toBeUndefined()
    })

    it('appends the fact rules to the extraction system prompt only, once', () => {
        const msgs = [
            { role: 'system', content: 'You are a Memory Extractor' },
            { role: 'user', content: 'u' }
        ]
        const once = injectFactRules(msgs)
        expect(once[0].content).toContain(CUSTOMER_MEMORY_FACT_RULES)
        expect(msgs[0].content).toBe('You are a Memory Extractor')
        expect(injectFactRules(once)).toBe(once)
        expect(CUSTOMER_MEMORY_FACT_RULES).toMatch(/NEVER extract what the assistant offered/)
        expect(CUSTOMER_MEMORY_FACT_RULES).toMatch(/max 8 words/)
    })

    it('wrapped LLM: rules in, sanitised JSON out, only for the extraction call; usage counted', async () => {
        const seen: any[] = []
        const create = jest.fn(async () => ({ usage: { prompt_tokens: 100, completion_tokens: 20 } }))
        const embCreate = jest.fn(async () => ({ usage: { prompt_tokens: 7, total_tokens: 7 } }))
        const memory: any = {
            llm: {
                openai: { chat: { completions: { create } } },
                generateResponse: async (m: any[]) => {
                    seen.push(m)
                    await memory.llm.openai.chat.completions.create({})
                    return '{"memory":[{"id":"0","text":"User lives in JB as of October 2026"},{"id":"1","text":"x","attributed_to":"assistant"}]}'
                }
            },
            embedder: { openai: { embeddings: { create: embCreate } } }
        }
        wrapLlmForCapture(memory)
        const cap: any = { usage: emptyUsage() }
        const res = await extractionCapture.run(cap, async () => {
            await memory.embedder.openai.embeddings.create({})
            return memory.llm.generateResponse([
                { role: 'system', content: 'You are a Memory Extractor' },
                { role: 'user', content: 'u' }
            ])
        })
        expect(JSON.parse(res).memory).toEqual([{ id: '0', text: 'Lives in JB', attributed_to: 'user' }])
        expect(seen[0][0].content).toContain(CUSTOMER_MEMORY_FACT_RULES)
        expect(cap.usage).toEqual({ llmCalls: 1, promptTokens: 100, completionTokens: 20, embeddingCalls: 1, embeddingTokens: 7 })

        // non-extraction calls pass through untouched
        const other = await memory.llm.generateResponse([{ role: 'system', content: 'something else' }])
        expect(JSON.parse(other).memory).toHaveLength(2)
        expect(seen[1][0].content).toBe('something else')
    })
})

describe('addCustomerFacts / recallCustomerFacts / queueForUser (shared with the agentflow endpoints)', () => {
    const cfg: any = { userId: '7:abc', ownerId: 7, salesmanId: '509', apiKey: 'k', baseUrl: 'b', llmModel: 'm', embeddingModel: 'e', backfill: false }

    it('returns tagged facts, the new summary and usage', async () => {
        const mem0: any = {
            llm: {
                generateResponse: async () =>
                    '{"memory":[{"id":"0","text":"Budget RM500","category":"budget","label":"Budget RM500"}],"summary":"Parent, RM500 budget."}'
            },
            search: jest.fn(),
            getAll: jest.fn(),
            add: jest.fn(async (_m: any, opts: any) => {
                expect(opts.metadata).toEqual({ source: 'flowise-agentflow', chatflow_id: 'tp3', owner_id: 7, ownerId: 7, salesmanId: '509', session_id: 's' })
                await mem0.llm.generateResponse([{ role: 'system', content: 'You are a Memory Extractor' }])
                return { results: [{ id: 'f1', memory: 'Budget RM500' }] }
            }),
            setPayload: jest.fn(async () => undefined),
            getSummary: jest.fn(async () => null),
            saveSummary: jest.fn(async () => undefined)
        }
        wrapLlmForCapture(mem0)
        const r = await addCustomerFacts(mem0, cfg, [{ role: 'user', content: 'my budget is RM500' }], { chatflowId: 'tp3', sessionId: 's', source: 'flowise-agentflow' }, jest.fn())
        expect(r.added).toEqual([{ id: 'f1', memory: 'Budget RM500', category: 'budget', label: 'Budget RM500' }])
        expect(r.summary).toBe('Parent, RM500 budget.')
        expect(mem0.setPayload).toHaveBeenCalledWith(['f1'], { category: 'budget', label: 'Budget RM500' })
        expect(mem0.saveSummary).toHaveBeenCalledWith('7:abc', 'Parent, RM500 budget.', { owner_id: 7, salesmanId: '509' })
    })

    it('recall ranks without a score cutoff and builds the prompt block with the summary', async () => {
        const mem0: any = {
            search: jest.fn(async () => ({ results: [{ id: 'a', memory: 'Has 2 kids', metadata: { category: 'family', label: 'Has Kids' } }] })),
            getAll: jest.fn(async () => ({ results: [] })),
            add: jest.fn(),
            getSummary: jest.fn(async () => 'Parent of two.')
        }
        const r = await recallCustomerFacts(mem0, cfg, 'what suits me?', 15)
        expect(mem0.search).toHaveBeenCalledWith('what suits me?', { topK: 15, threshold: 0, filters: { user_id: '7:abc' } })
        expect(r.facts).toEqual([{ id: 'a', memory: 'Has 2 kids', category: 'family', label: 'Has Kids' }])
        expect(r.text).toBe(`${CUSTOMER_MEMORY_FACTS_HEADER}\n- Has 2 kids\nAt a glance: Parent of two.`)
        const empty = await recallCustomerFacts(mem0, cfg, '', 15)
        expect(mem0.getAll).toHaveBeenCalled()
        expect(empty.text).toBe('')
    })

    it('queueForUser runs jobs for one user in order, even after a failure', async () => {
        const order: string[] = []
        const slow = queueForUser('u', async () => {
            await new Promise((r) => setTimeout(r, 20))
            order.push('a')
            throw new Error('boom')
        })
        const fast = queueForUser('u', async () => {
            order.push('b')
            return 2
        })
        await expect(slow).rejects.toThrow('boom')
        await expect(fast).resolves.toBe(2)
        expect(order).toEqual(['a', 'b'])
    })
})

describe('embedding dimensions', () => {
    it('asks OpenAI v3 and Gemini embeddings for the collection size, leaves others alone', () => {
        expect(supportsDimensions('text-embedding-3-small')).toBe(true)
        expect(supportsDimensions('gemini-embedding-001')).toBe(true)
        expect(supportsDimensions('text-embedding-ada-002')).toBe(false)
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
