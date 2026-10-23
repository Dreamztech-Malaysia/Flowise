import express from 'express'
import customerMemoryController from '../../controllers/customer-memory'
import { checkAnyPermission } from '../../enterprise/rbac/PermissionCheck'

const router = express.Router()

const VIEW = 'chatflows:view,chatflows:update,agentflows:view,agentflows:update'
const DELETE = 'chatflows:delete,agentflows:delete'

// RECALL / SAVE for agentflows. '/sealed/*' is whitelisted (auth = sealed token in the body, see
// services/customer-memory/sealedToken.ts). POST routes, so they never clash with GET/DELETE '/:userId'.
router.post('/sealed/recall', customerMemoryController.sealedRecall)
router.post('/sealed/save', customerMemoryController.sealedSave)
router.post('/recall', checkAnyPermission(VIEW), customerMemoryController.recall)
router.post('/save', checkAnyPermission(VIEW), customerMemoryController.save)

// READ
router.get('/', checkAnyPermission(VIEW), customerMemoryController.listUsers)
// must stay above '/:userId' so 'summary' is not taken as a user id
router.get('/summary', checkAnyPermission(VIEW), customerMemoryController.getSummary)
router.get('/:userId', checkAnyPermission(VIEW), customerMemoryController.getFacts)

// DELETE
router.delete('/:userId', checkAnyPermission(DELETE), customerMemoryController.deleteAllFacts)
router.delete('/:userId/:memoryId', checkAnyPermission(DELETE), customerMemoryController.deleteFact)

export default router
