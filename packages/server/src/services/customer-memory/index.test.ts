import customerMemoryService, { toFact, weekStart } from '.'

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
                session_id: 's1',
                category: 'interest',
                label: 'Condo'
            }
        })
        expect(f).toEqual({
            id: 'a-1',
            memory: 'Wants a 3-room condo',
            created_at: '2026-10-01T00:00:00.000Z',
            updated_at: '2026-10-02T00:00:00.000Z',
            user_id: '7:abc',
            owner_id: 7,
            category: 'interest',
            label: 'Condo',
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

    it('returns facts plus the stored summary', async () => {
        global.fetch = jest.fn(async (url: string) =>
            url.includes('_summaries')
                ? ok({ points: [{ id: 's', payload: { summary: 'Family buyer.', updatedAt: '2026-10-03T00:00:00Z' } }] })
                : ok({ points: [{ id: 'f1', payload: { data: 'Has kids', user_id: '7:a', category: 'family', label: 'Family' } }], next_page_offset: null })
        ) as any
        const r = await customerMemoryService.getFacts('7:a')
        expect(r.summary).toBe('Family buyer.')
        expect(r.summaryUpdatedAt).toBe('2026-10-03T00:00:00.000Z')
        expect(r.facts[0]).toMatchObject({ id: 'f1', memory: 'Has kids', category: 'family', label: 'Family' })
    })

    it('deleteAllFacts also deletes the stored summary', async () => {
        const fetchMock = jest.fn(async (_url: string, _init: any) => ok({ points: [{ id: 1 }], next_page_offset: null }))
        global.fetch = fetchMock as any
        await customerMemoryService.deleteAllFacts('7:a')
        const urls = fetchMock.mock.calls.map((c) => c[0])
        expect(urls).toContain('http://qdrant.test:6333/collections/customer_memory_summaries/points/delete?wait=true')
    })

    it('builds the weekly category/label summary without AI', async () => {
        const fetchMock = jest.fn(async (_url: string, _init: any) =>
            ok({
                points: [
                    { id: 1, payload: { user_id: '7:a', salesmanId: 's1', category: 'objection', label: 'Price Too High', createdAt: '2026-10-20T10:00:00Z' } },
                    { id: 2, payload: { user_id: '7:a', salesmanId: 's1', category: 'objection', label: 'Price Too High', createdAt: '2026-10-22T10:00:00Z' } },
                    { id: 3, payload: { user_id: '7:b', createdAt: '2026-10-14T00:00:00Z' } },
                    { id: 4, payload: { user_id: '7:b', category: 'budget', label: 'Budget', createdAt: '2026-09-01T00:00:00Z' } }
                ],
                next_page_offset: null
            })
        )
        global.fetch = fetchMock as any
        const rows = await customerMemoryService.getSummary('7', '2026-10-01')
        expect(rows).toEqual([
            { userId: '7:a', salesmanId: 's1', category: 'objection', label: 'Price Too High', week: '2026-10-19', count: 2 },
            { userId: '7:b', salesmanId: null, category: 'other', label: 'Other', week: '2026-10-12', count: 1 }
        ])
        expect(JSON.parse(fetchMock.mock.calls[0][1].body).filter.should).toHaveLength(2)
        await expect(customerMemoryService.getSummary(undefined, '22-10-2026')).rejects.toThrow(/YYYY-MM-DD/)
        expect(weekStart('2026-10-25T23:00:00Z')).toBe('2026-10-19')
    })
})
