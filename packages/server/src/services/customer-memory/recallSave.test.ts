import { openCustomerMemory, sealCustomerMemory } from './sealedToken'
import { __setMem0Factory, recall, save, toConfig, toMessages } from './recallSave'

// The full components index pulls jsdom (ESM) which jest cannot load here; the customer memory parts are self-contained.
jest.mock('flowise-components', () => ({
    ...jest.requireActual('../../../../components/nodes/memory/Mem0/customerMemory'),
    ...jest.requireActual('../../../../components/nodes/memory/Mem0/customerMemoryOps')
}))
jest.mock('../../utils/logger', () => ({ __esModule: true, default: { info: jest.fn(), warn: jest.fn(), error: jest.fn() } }))

const cm = { enabled: true, userId: '211:p:60100000001', ownerId: 211, salesmanId: 509, apiKey: 'sk-test-abcdefgh', llmModel: 'gpt-4.1' }

describe('sealed customer memory token', () => {
    beforeEach(() => {
        process.env.CUSTOMER_MEMORY_TOKEN_SECRET = 'unit-test-secret-0123456789'
    })
    afterEach(() => {
        delete process.env.CUSTOMER_MEMORY_TOKEN_SECRET
    })

    it('round-trips and hides the key', () => {
        const t = sealCustomerMemory(cm, 600)
        expect(t.startsWith('cm1.')).toBe(true)
        expect(t).not.toContain('sk-test')
        expect(openCustomerMemory(t)).toEqual(cm)
    })

    it('rejects tampered, expired, over-long-lived, wrong-secret and malformed tokens with 401', () => {
        const t = sealCustomerMemory(cm, 600)
        const flipped = t.slice(0, -3) + (t.slice(-3, -2) === 'A' ? 'B' : 'A') + t.slice(-2)
        for (const bad of [flipped, 'cm1.abc', 'nope', undefined, sealCustomerMemory(cm, -5), sealCustomerMemory(cm, 3 * 24 * 3600)]) {
            expect(() => openCustomerMemory(bad)).toThrow(expect.objectContaining({ statusCode: 401 }))
        }
        process.env.CUSTOMER_MEMORY_TOKEN_SECRET = 'another-secret-0123456789'
        expect(() => openCustomerMemory(t)).toThrow(expect.objectContaining({ statusCode: 401 }))
    })

    it('is unavailable (503) without a secret', () => {
        delete process.env.CUSTOMER_MEMORY_TOKEN_SECRET
        expect(() => openCustomerMemory('cm1.x')).toThrow(expect.objectContaining({ statusCode: 503 }))
    })

    it('matches the PHP minting format (sha256(secret) key, iv|ct|tag, base64url)', () => {
        // produced with: openssl_encrypt(json, 'aes-256-gcm', hash('sha256', secret, true), OPENSSL_RAW_DATA, iv, tag)
        const crypto = require('crypto')
        const iv = Buffer.alloc(12, 1)
        const k = crypto.createHash('sha256').update('unit-test-secret-0123456789').digest()
        const c = crypto.createCipheriv('aes-256-gcm', k, iv)
        const ct = Buffer.concat([c.update(JSON.stringify({ exp: Math.floor(Date.now() / 1000) + 60, cm })), c.final()])
        const raw = Buffer.concat([iv, ct, c.getAuthTag()]).toString('base64').replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')
        expect(openCustomerMemory('cm1.' + raw).userId).toBe(cm.userId)
    })
})

describe('recall/save service', () => {
    afterEach(() => __setMem0Factory())

    it('validates config and messages', () => {
        expect(() => toConfig({ ...cm, enabled: false })).toThrow(expect.objectContaining({ statusCode: 400 }))
        expect(toConfig(cm).llmModel).toBe('gpt-4.1')
        expect(() => toMessages({ messages: [{ role: 'assistant', content: 'hi' }] })).toThrow(expect.objectContaining({ statusCode: 400 }))
        expect(toMessages({ input: ' I have 2 kids ', output: 'Nice!' })).toEqual([
            { role: 'user', content: 'I have 2 kids' },
            { role: 'assistant', content: 'Nice!' }
        ])
        expect(toMessages({ messages: [{ role: 'userMessage', content: 'a' }, { role: 'system', content: 'x' }] })).toEqual([{ role: 'user', content: 'a' }])
    })

    it('recall returns facts + text; provider errors become 502 without the key', async () => {
        __setMem0Factory(async () => ({
            search: async () => ({ results: [{ id: '1', memory: 'Has 2 kids' }] }),
            getAll: async () => ({ results: [] }),
            add: async () => ({ results: [] })
        }))
        const r = await recall(toConfig(cm), 'what suits me?', 5)
        expect(r.facts.map((f) => f.memory)).toEqual(['Has 2 kids'])
        expect(r.text).toContain('- Has 2 kids')

        __setMem0Factory(async () => {
            throw new Error('401 Incorrect API key provided: sk-test-abcdefgh')
        })
        const err: any = await recall(toConfig(cm), 'q', 5).catch((e) => e)
        expect(err.statusCode).toBe(502)
        expect(String(err.message)).not.toContain('abcdefgh')
    })

    it('save adds with agentflow metadata', async () => {
        const add = jest.fn(async () => ({ results: [{ id: 'f1', memory: 'Budget RM500' }] }))
        __setMem0Factory(async () => ({ search: async () => ({ results: [] }), getAll: async () => ({ results: [] }), add }))
        const r = await save(toConfig(cm), [{ role: 'user', content: 'budget RM500' }], { chatflowId: 'tp3', sessionId: 's', source: 'flowise-agentflow' })
        expect(r.added).toEqual([{ id: 'f1', memory: 'Budget RM500', category: 'budget', label: 'Budget' }])
        expect((add.mock.calls[0] as any[])[1].metadata).toMatchObject({ source: 'flowise-agentflow', chatflow_id: 'tp3', owner_id: 211, salesmanId: '509' })
    })
})
