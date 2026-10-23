import { NextFunction, Request, Response } from 'express'
import type { TokenUsage } from 'flowise-components'
import customerMemoryService from '../../services/customer-memory'
import * as recallSave from '../../services/customer-memory/recallSave'
import { toConfig, toContext, toMessages } from '../../services/customer-memory/recallSave'
import { openCustomerMemory } from '../../services/customer-memory/sealedToken'

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

const usageOut = (u: TokenUsage) => ({
    llmCalls: u.llmCalls,
    promptTokens: u.promptTokens,
    completionTokens: u.completionTokens,
    embeddingCalls: u.embeddingCalls,
    embeddingTokens: u.embeddingTokens
})

/** POST /recall { customerMemory, query, limit? } -> { userId, facts, summary, text, usage } (API key auth) */
const recall = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const cfg = toConfig(req.body?.customerMemory)
        const r = await recallSave.recall(cfg, req.body?.query, req.body?.limit)
        return res.json({ userId: cfg.userId, facts: r.facts, summary: r.summary, text: r.text, usage: usageOut(r.usage) })
    } catch (error) {
        next(error)
    }
}

/**
 * POST /save { customerMemory, messages:[{role,content}] | input+output, sessionId?, chatflowId?, background? }
 * -> { userId, added, summary, usage }   (background=true -> 202 { userId, queued: true })
 */
const save = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const cfg = toConfig(req.body?.customerMemory)
        const messages = toMessages(req.body)
        const ctx = toContext(req.body, 'flowise-api')
        if (req.body?.background === true) {
            recallSave.saveInBackground(cfg, messages, ctx)
            return res.status(202).json({ userId: cfg.userId, queued: true })
        }
        const r = await recallSave.save(cfg, messages, ctx)
        return res.json({ userId: cfg.userId, added: r.added, summary: r.summary, usage: usageOut(r.usage) })
    } catch (error) {
        next(error)
    }
}

/** POST /sealed/recall { token, query, limit? } -- same as /recall, auth = sealed token. Omits userId. */
const sealedRecall = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const cfg = toConfig(openCustomerMemory(req.body?.token))
        const r = await recallSave.recall(cfg, req.body?.query, req.body?.limit)
        return res.json({ facts: r.facts, summary: r.summary, text: r.text, usage: usageOut(r.usage) })
    } catch (error) {
        next(error)
    }
}

/** POST /sealed/save { token, messages | input+output, sessionId?, chatflowId?, background? } -- auth = sealed token. */
const sealedSave = async (req: Request, res: Response, next: NextFunction) => {
    try {
        const cfg = toConfig(openCustomerMemory(req.body?.token))
        const messages = toMessages(req.body)
        const ctx = toContext(req.body, 'flowise-agentflow')
        if (req.body?.background === true) {
            recallSave.saveInBackground(cfg, messages, ctx)
            return res.status(202).json({ queued: true })
        }
        const r = await recallSave.save(cfg, messages, ctx)
        return res.json({ added: r.added, summary: r.summary, usage: usageOut(r.usage) })
    } catch (error) {
        next(error)
    }
}

export default { getFacts, getSummary, deleteAllFacts, deleteFact, listUsers, recall, save, sealedRecall, sealedSave }
