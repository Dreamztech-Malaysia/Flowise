import dotenv from 'dotenv'
import path from 'path'

const envPath = path.join(__dirname, '..', '..', '.env')
dotenv.config({ path: envPath, override: true })

export * from '../evaluation/EvaluationRunner'
export { MCPToolkit } from '../nodes/tools/MCP/core'
export * from './agentflowv2Generator'
export * from './followUpPrompts'
export * from './handler'
export * from './headerValidation'
export * from './httpSecurity'
export * from './Interface'
export * from './recordManagerSecurity'
export * from './sanitizeDataSourceOptions'
export * from './speechToText'
export * from './storage'
export * from './storageUtils'
export * from './textToSpeech'
export { tracingEnvEnabled } from './tracingEnv'
export * from './utils'
export * from './validator'

// Dreamztech fork: self-hosted customer memory, used by the server's /api/v1/customer-memory recall/save endpoints.
export { addCustomerFacts, recallCustomerFacts, queueForUser } from '../nodes/memory/Mem0/customerMemoryOps'
export type { AddCustomerFactsResult, AddedFact, RecallResult } from '../nodes/memory/Mem0/customerMemoryOps'
export { parseCustomerMemoryConfig, getMem0Instance, redactSecrets, emptyUsage } from '../nodes/memory/Mem0/customerMemory'
export type { CustomerMemoryConfig, Mem0ChatMessage, Mem0Like, TokenUsage } from '../nodes/memory/Mem0/customerMemory'
