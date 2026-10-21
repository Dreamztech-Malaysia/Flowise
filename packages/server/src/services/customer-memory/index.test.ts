import customerMemoryService, { toFact } from '.'

describe('customer-memory service', () => {
    const realFetch = global.fetch
    beforeEach(() => {
        process.env.QDRANT_URL = 'http://qdrant.test:6333/'
        process.env.QDRANT_API_KEY = 'qk'
    })
    afterEach(() => {
        global.fetch = realFetch
        delete process.env.QDRANT_URL
        delete process.env.QDRANT_API_KEY
    })

    const ok = (result: any) => ({ ok: true, status: 200, json: async () => ({ result }), text: async () => '' }) as any

    it('maps a mem0 payload to the public fact shape', () => {
        const f = toFact({
            id: 'a-1',
            payload: {
                data: 'Wants a 3-room condo',
                hash: 'h',
                textLemmatized: 'x',
                createdAt: '2026-10-01T00:00:00.000Z',
                updatedAt: '2026-10-02T00:00:00.000Z',
                user_id: '7:abc',
                owner_id: 7,
                session_id: 's1'
            }
        })
        expect(f).toEqual({
            id: 'a-1',
            memory: 'Wants a 3-room condo',
            created_at: '2026-10-01T00:00:00.000Z',
            updated_at: '2026-10-02T00:00:00.000Z',
            user_id: '7:abc',
            owner_id: 7,
            category: null,
            metadata: { session_id: 's1' }
        })
    })

    it('lists users scoped by owner and filtered by updatedBefore, sending the api-key header', async () => {
        const fetchMock = jest.fn(async (_url: string, _init: any) =>
            ok({
                points: [
                    { id: 1, payload: { user_id: '7:a', owner_id: 7, createdAt: '2025-01-01T00:00:00Z' } },
                    { id: 2, payload: { user_id: '7:a', owner_id: 7, createdAt: '2025-02-01T00:00:00Z' } },
                    { id: 3, payload: { user_id: '7:b', owner_id: 7, createdAt: '2026-10-01T00:00:00Z' } }
                ],
                next_page_offset: null
            })
        )
        global.fetch = fetchMock as any
        const users = await customerMemoryService.listUsers('7', '2026-01-01T00:00:00Z')
        expect(users).toEqual([{ userId: '7:a', ownerId: 7, count: 2, lastUpdatedAt: '2025-02-01T00:00:00.000Z' }])
        const [url, init] = fetchMock.mock.calls[0]
        expect(url).toBe('http://qdrant.test:6333/collections/customer_memory/points/scroll')
        expect(init.headers['api-key']).toBe('qk')
        expect(JSON.parse(init.body).filter.should).toHaveLength(2)
    })

    it('refuses to delete a fact that belongs to another user', async () => {
        global.fetch = jest.fn(async () => ok([{ id: 'f-1', payload: { user_id: 'other' } }])) as any
        await expect(customerMemoryService.deleteFact('7:a', 'f-1')).rejects.toThrow(/not found/)
    })

    it('returns an empty list when the collection does not exist yet', async () => {
        global.fetch = jest.fn(async () => ({ ok: false, status: 404, json: async () => ({}), text: async () => '' })) as any
        await expect(customerMemoryService.getFacts('7:a')).resolves.toEqual([])
    })
})
