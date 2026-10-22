import { NextFunction, Request, Response } from 'express'
import customerMemoryService from '../../services/customer-memory'

const getFacts = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const userId = req.params.userId
        const { facts, summary, summaryUpdatedAt } = await customerMemoryService.getFacts(userId)
        // Superset of the previous { userId, count, facts } shape; facts now also carry `label`.
        return res.json({ userId, count: facts.length, summary, summaryUpdatedAt, facts })
    } catch (error) {
        next(error)
    }
}

const deleteAllFacts = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const userId = req.params.userId
        const result = await customerMemoryService.deleteAllFacts(userId)
        return res.json({ userId, ...result })
    } catch (error) {
        next(error)
    }
}

const deleteFact = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const { userId, memoryId } = req.params
        const result = await customerMemoryService.deleteFact(userId, memoryId)
        return res.json({ userId, memoryId, ...result })
    } catch (error) {
        next(error)
    }
}

const listUsers = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const ownerId = typeof req.query.ownerId === 'string' && req.query.ownerId !== '' ? req.query.ownerId : undefined
        const updatedBefore =
            typeof req.query.updatedBefore === 'string' && req.query.updatedBefore !== '' ? req.query.updatedBefore : undefined
        const users = await customerMemoryService.listUsers(ownerId, updatedBefore)
        return res.json({ ownerId: ownerId ?? null, updatedBefore: updatedBefore ?? null, count: users.length, users })
    } catch (error) {
        next(error)
    }
}

const getSummary = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const ownerId = typeof req.query.ownerId === 'string' && req.query.ownerId !== '' ? req.query.ownerId : undefined
        const since = typeof req.query.since === 'string' && req.query.since !== '' ? req.query.since : undefined
        return res.json(await customerMemoryService.getSummary(ownerId, since))
    } catch (error) {
        next(error)
    }
}

export default { getFacts, getSummary, deleteAllFacts, deleteFact, listUsers }
